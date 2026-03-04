import os
import json
from pipeline import load_pipeline


def safe_print_dict(d, title=None, max_len=2500):
    """
    Prints dict safely without crashing.
    """
    if title:
        print(f"\n==================== {title} ====================")

    try:
        s = json.dumps(d, indent=2, ensure_ascii=False)
        if len(s) > max_len:
            print(s[:max_len] + "\n... (truncated)")
        else:
            print(s)
    except Exception:
        print(str(d))


def summarize_stage(stage_name, stage_report):
    """
    Prints a short summary for a stage.
    """
    print(f"\n--- {stage_name} ---")

    if stage_report is None:
        print("No report found.")
        return

    if not isinstance(stage_report, dict):
        print(f"Report is not dict. Type={type(stage_report)}")
        print(stage_report)
        return

    # Common known keys
    if "status" in stage_report:
        print(f"status: {stage_report.get('status')}")

    if "semantic_issue_count" in stage_report:
        print(f"semantic_issue_count: {stage_report.get('semantic_issue_count')}")

    if "pattern_issue_count" in stage_report:
        print(f"pattern_issue_count: {stage_report.get('pattern_issue_count')}")

    if "row_issues" in stage_report:
        try:
            print(f"row_issues: {len(stage_report.get('row_issues', []))}")
        except Exception:
            pass

    if "column_issues" in stage_report:
        try:
            print(f"column_issues: {len(stage_report.get('column_issues', []))}")
        except Exception:
            pass

    # LLM info inside stage report
    if "llm" in stage_report and isinstance(stage_report["llm"], dict):
        llm = stage_report["llm"]
        print(f"llm.enabled: {llm.get('enabled')}")
        print(f"llm.model: {llm.get('model')}")
        applied = llm.get("applied_aliases", {})
        if isinstance(applied, dict):
            print(f"llm.applied_aliases_count: {len(applied)}")

    # Print drift/stale warnings (Stage3)
    drift = stage_report.get("type_drift_warnings")
    if isinstance(drift, list) and drift:
        print(f"type_drift_warnings: {len(drift)}")
        for w in drift[:5]:
            print("  -", w)

    stale = stage_report.get("stale_history_warnings")
    if isinstance(stale, list) and stale:
        print(f"stale_history_warnings: {len(stale)}")
        for w in stale[:5]:
            print("  -", w)


def main():
    path = "pipeline.pkl"

    if not os.path.exists(path):
        print("❌ pipeline.pkl not found.")
        print("Run training first:")
        print("python train_pipeline.py example_train.csv --output pipeline.pkl --enable-llm")
        return

    print(f"Loading pipeline from: {path}")
    pipe = load_pipeline(path)

    report = getattr(pipe, "last_report_", None)
    llm_report = getattr(pipe, "llm_report_", None)

    # -------------------------
    # Pipeline basic info
    # -------------------------
    print("\n==================== PIPELINE INFO ====================")
    print("is_fitted:", getattr(pipe, "is_fitted", False))
    print("cleaning_mode:", getattr(pipe, "cleaning_mode", "unknown"))
    print("enable_llm:", getattr(pipe, "enable_llm", None))
    print("llm_model:", getattr(pipe, "llm_model", None))

    # -------------------------
    # LLM report
    # -------------------------
    if llm_report is not None:
        safe_print_dict(llm_report, title="LLM REPORT (fit-time)")
    else:
        print("\n[LLM REPORT] Not found (LLM may be disabled or fit not run).")

    # -------------------------
    # Pipeline last_report_
    # -------------------------
    if report is None:
        print("\n❌ last_report_ is missing.")
        print("Run transform at least once before checking report.")
        return

    print("\n==================== PIPELINE STATUS ====================")
    print("status:", report.get("status"))
    print("cleaning_mode:", report.get("cleaning_mode"))

    # -------------------------
    # Stage summaries
    # -------------------------
    print("\n==================== STAGE SUMMARIES ====================")

    stage_order = [
        "cleaningStage0",
        "cleaningStage1",
        "cleaningStage2",
        "semanticValidator",
        "relationshipValidator",
        "patternValidator",
        "outlierAwareImputer",
        "exportNormalizer",
    ]

    for stage_name in stage_order:
        summarize_stage(stage_name, report.get(stage_name))

    # -------------------------
    # Full report (optional)
    # -------------------------
    print("\n==================== FULL REPORT (TRUNCATED) ====================")
    safe_print_dict(report, title="last_report_", max_len=4000)


if __name__ == "__main__":
    main()
