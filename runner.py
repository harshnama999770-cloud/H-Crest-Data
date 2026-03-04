# =========================================================
# runner.py
# Runs cleaning pipeline and Stage 6 scorecard
# =========================================================

import os
import time
import json
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from cleaningStage6 import DataQualityScorecard


# =========================================================
# NEW: Utility helpers
# =========================================================

def _now_ts():
    return time.time()


def _mb(x_bytes: int) -> float:
    return float(x_bytes) / (1024 * 1024)


def _estimate_df_memory_bytes(df: pd.DataFrame) -> int:
    """
    Rough memory estimate of a DataFrame.
    """
    try:
        return int(df.memory_usage(deep=True).sum())
    except Exception:
        return 0


def _default_progress_callback(payload: dict):
    """
    Safe default callback (does nothing).
    GUI can pass its own callback.
    """
    return None


def _emit_progress(cb, stage: str, percent: float, message: str, extra: dict | None = None):
    """
    Sends progress updates in a consistent structure.
    """
    if cb is None:
        return

    payload = {
        "stage": str(stage),
        "percent": float(max(0.0, min(100.0, percent))),
        "message": str(message),
        "timestamp": _now_ts(),
    }
    if extra:
        payload.update(extra)

    try:
        cb(payload)
    except Exception:
        # Never crash runner due to UI callback errors
        pass


def _memory_monitor(df: pd.DataFrame, warn_mb: float = 500.0):
    """
    Warn-only memory monitor.
    """
    mem_bytes = _estimate_df_memory_bytes(df)
    mem_mb = _mb(mem_bytes)

    warnings = []
    if mem_mb >= warn_mb:
        warnings.append(
            f"[MEMORY WARNING] DataFrame is ~{mem_mb:.1f} MB in memory. "
            f"This may be slow or crash on low-RAM systems."
        )

    # also warn for huge row counts
    try:
        n = len(df)
        if n >= 2_000_000:
            warnings.append(
                f"[MEMORY WARNING] Dataset has {n:,} rows. "
                f"Consider chunking / multi-file mode."
            )
    except Exception:
        pass

    return {
        "estimated_df_memory_mb": float(round(mem_mb, 2)),
        "warnings": warnings,
    }


# =========================================================
# Your original helpers (kept)
# =========================================================

def _collect_all_issues(pipe):
    """
    Collect issues from all stages inside either:
    - sklearn Pipeline
    - SafeCleaningPipeline (your custom wrapper)
    """

    all_issues = []

    # Case 1: sklearn Pipeline directly
    if hasattr(pipe, "steps"):
        for step_name, step_obj in pipe.steps:
            if hasattr(step_obj, "issues"):
                its = getattr(step_obj, "issues", [])
                if isinstance(its, list):
                    all_issues.extend(its)

    # Case 2: SafeCleaningPipeline wrapper
    if hasattr(pipe, "pipeline") and hasattr(pipe.pipeline, "steps"):
        for step_name, step_obj in pipe.pipeline.steps:
            if hasattr(step_obj, "issues"):
                its = getattr(step_obj, "issues", [])
                if isinstance(its, list):
                    all_issues.extend(its)

    # Fallback: pipe itself stores issues
    if hasattr(pipe, "issues"):
        its = getattr(pipe, "issues", [])
        if isinstance(its, list):
            all_issues.extend(its)

    # Keep only dict issues
    all_issues = [x for x in all_issues if isinstance(x, dict)]

    # OPTIONAL: remove duplicates (very useful)
    # We deduplicate by (stage, row, column, issue)
    seen = set()
    dedup = []
    for it in all_issues:
        key = (
            str(it.get("stage", "")),
            str(it.get("row", "")),
            str(it.get("column", "")),
            str(it.get("issue", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)

    return dedup


def _split_issues(all_issues: list):
    """
    Splits issues into:
    - row_issues (row != None)
    - global_issues (row == None)
    """
    row_issues = []
    global_issues = []

    for it in all_issues or []:
        if not isinstance(it, dict):
            continue

        if it.get("row") is None:
            global_issues.append(it)
        else:
            row_issues.append(it)

    return row_issues, global_issues


# =========================================================
# NEW: Timeout-safe execution in a separate process
# =========================================================

def _run_pipeline_worker(pipe, df: pd.DataFrame):
    """
    Runs transform + stage6 inside worker process.
    Returns a pure-python dict (pickle safe).
    """
    cleaned_df = pipe.transform(df, strict=True)

    if cleaned_df is None:
        raise ValueError("Pipeline returned None (rejected).")

    if not isinstance(cleaned_df, pd.DataFrame):
        raise TypeError(f"Pipeline returned {type(cleaned_df)}, expected pandas DataFrame.")

    all_issues = _collect_all_issues(pipe)
    row_issues, global_issues = _split_issues(all_issues)

    stage6 = DataQualityScorecard(issues=all_issues)
    stage6.fit(cleaned_df)
    stage6.transform(cleaned_df)

    scorecard = getattr(stage6, "scorecard_", {})

    reports = {
        "pipeline": getattr(pipe, "last_report_", {}),
        "stage6": getattr(stage6, "last_report_", {}),
        "issues_summary": {
            "total_issues": int(len(all_issues)),
            "row_issues": int(len(row_issues)),
            "global_issues": int(len(global_issues)),
        }
    }

    return {
        "cleaned_df": cleaned_df,
        "issues": all_issues,
        "row_issues": row_issues,
        "global_issues": global_issues,
        "scorecard": scorecard,
        "reports": reports,
    }


def _run_with_timeout(pipe, df: pd.DataFrame, timeout_seconds: int | None = None):
    """
    Runs pipeline inside a process so timeout works safely.
    """
    if timeout_seconds is None:
        # no timeout
        return _run_pipeline_worker(pipe, df)

    with ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run_pipeline_worker, pipe, df)
        return fut.result(timeout=int(timeout_seconds))


# =========================================================
# MAIN: Used by GUI
# =========================================================

def run_existing_pipeline_with_scorecard(
    pipe,
    df: pd.DataFrame,

    # NEW
    progress_callback=None,
    timeout_seconds: int | None = None,
    memory_warn_mb: float = 500.0,
):
    """
    Used by GUI.

    Input:
      - pipe: already trained pipeline loaded from pipeline.pkl
      - df: dataframe to clean

    Output:
      {
        "cleaned_df": DataFrame,
        "issues": list[dict],         # ALL issues
        "row_issues": list[dict],     # row-level issues only
        "global_issues": list[dict],  # row=None warnings
        "scorecard": dict,
        "reports": dict,

        # NEW:
        "runner_meta": dict
      }
    """

    cb = progress_callback or _default_progress_callback

    runner_meta = {
        "timeout_seconds": timeout_seconds,
        "memory_warn_mb": float(memory_warn_mb),
        "memory": {},
        "warnings": [],
        "timing": {},
        "status": "started",
    }

    t0 = _now_ts()

    # ---------------------------------------------------------
    # 0) Memory monitoring
    # ---------------------------------------------------------
    _emit_progress(cb, "runner", 2, "Checking memory usage...")
    mem = _memory_monitor(df, warn_mb=memory_warn_mb)
    runner_meta["memory"] = mem
    runner_meta["warnings"].extend(mem.get("warnings", []))

    # ---------------------------------------------------------
    # 1) Run pipeline + scorecard (timeout safe)
    # ---------------------------------------------------------
    _emit_progress(cb, "pipeline", 10, "Running cleaning pipeline...")

    try:
        result = _run_with_timeout(pipe, df, timeout_seconds=timeout_seconds)
    except Exception as e:
        runner_meta["status"] = "failed"
        runner_meta["timing"]["total_seconds"] = float(round(_now_ts() - t0, 3))

        err = str(e)
        tb = traceback.format_exc()

        if "timeout" in err.lower():
            err = f"Pipeline timed out after {timeout_seconds} seconds."

        return {
            "cleaned_df": None,
            "issues": [],
            "row_issues": [],
            "global_issues": [],
            "scorecard": {},
            "reports": {"error": err, "traceback": tb},
            "runner_meta": runner_meta,
        }

    _emit_progress(cb, "pipeline", 75, "Pipeline finished. Building scorecard report...")

    # ---------------------------------------------------------
    # 2) Finish + attach runner metadata
    # ---------------------------------------------------------
    runner_meta["status"] = "success"
    runner_meta["timing"]["total_seconds"] = float(round(_now_ts() - t0, 3))

    # Add runner meta into reports
    reports = result.get("reports", {})
    reports["runner_meta"] = runner_meta
    result["reports"] = reports
    result["runner_meta"] = runner_meta

    _emit_progress(cb, "runner", 100, "Done.")

    return result


# ---------------------------------------------------------
# OPTIONAL: terminal runner
# ---------------------------------------------------------

def run_cleaning_with_scorecard(
    df: pd.DataFrame,

    # NEW
    progress_callback=None,
    timeout_seconds: int | None = None,
    memory_warn_mb: float = 500.0,
):
    """
    Optional terminal runner.
    This version builds a new pipeline fresh.
    """
    cb = progress_callback or _default_progress_callback

    from pipeline import build_pipeline

    _emit_progress(cb, "pipeline", 5, "Building pipeline...")
    pipe = build_pipeline()

    _emit_progress(cb, "pipeline", 10, "Training pipeline...")
    pipe.fit(df)

    _emit_progress(cb, "pipeline", 20, "Running cleaning pipeline...")

    # Run with timeout safe
    result = run_existing_pipeline_with_scorecard(
        pipe,
        df,
        progress_callback=cb,
        timeout_seconds=timeout_seconds,
        memory_warn_mb=memory_warn_mb,
    )

    return result


# =========================================================
# NEW: Multi-file mode (parallel processing)
# =========================================================

def run_multi_file_mode(
    pipe,
    filepaths: list[str],
    read_kwargs: dict | None = None,
    max_workers: int = 2,

    # NEW
    timeout_seconds_per_file: int | None = None,
    progress_callback=None,
):
    """
    Runs pipeline on multiple CSV files in parallel.

    Returns:
      {
        "results": [{"file":..., "result":...}, ...],
        "failures": [{"file":..., "error":...}, ...],
      }
    """

    cb = progress_callback or _default_progress_callback
    read_kwargs = read_kwargs or {}

    results = []
    failures = []

    if not filepaths:
        return {"results": [], "failures": []}

    max_workers = max(1, int(max_workers))

    _emit_progress(cb, "multi_file", 0, f"Starting multi-file mode ({len(filepaths)} files)...")

    def _process_one(fp):
        df = pd.read_csv(fp, **read_kwargs)
        res = run_existing_pipeline_with_scorecard(
            pipe,
            df,
            timeout_seconds=timeout_seconds_per_file,
            progress_callback=None,  # avoid cross-process UI calls
        )
        return fp, res

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_one, fp): fp for fp in filepaths}

        done = 0
        total = len(filepaths)

        for fut in as_completed(futs):
            fp = futs[fut]
            done += 1

            try:
                fp, res = fut.result()
                results.append({"file": fp, "result": res})
            except Exception as e:
                failures.append({"file": fp, "error": str(e)})

            pct = (done / max(total, 1)) * 100
            _emit_progress(cb, "multi_file", pct, f"Processed {done}/{total} files")

    _emit_progress(cb, "multi_file", 100, "Multi-file processing finished.")
    return {"results": results, "failures": failures}


# =========================================================
# Optional: Run directly from terminal
# =========================================================
if __name__ == "__main__":
    input_csv = "input.csv"
    output_csv = "cleaned.csv"

    df = pd.read_csv(input_csv)

    result = run_cleaning_with_scorecard(
        df,
        timeout_seconds=120,   # NEW: prevents hanging forever
        memory_warn_mb=500.0,  # NEW: memory warning
    )

    if result.get("cleaned_df") is None:
        print("\n=== FAILED ===")
        print(result.get("reports", {}))
        raise SystemExit(1)

    print("\n=== SCORECARD ===")
    print(result["scorecard"])

    result["cleaned_df"].to_csv(output_csv, index=False)

    print(f"\nCleaned CSV saved to: {output_csv}")
    print(f"Total issues found: {len(result['issues'])}")
    print(f"Row issues used in score: {len(result['row_issues'])}")
    print(f"Global warnings (row=None): {len(result['global_issues'])}")

    runner_meta = result.get("runner_meta", {})
    if runner_meta.get("warnings"):
        print("\n=== RUNNER WARNINGS ===")
        for w in runner_meta["warnings"]:
            print("-", w)
