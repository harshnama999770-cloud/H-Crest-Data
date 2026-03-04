# =========================================================
# cleaningStage6.py
# Stage 6: Data Quality Scorecard (SMART + ENTERPRISE REPORT)
# =========================================================

import pandas as pd
import numpy as np
import json
import os
import hashlib
import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from pipeline_utils import _check_df_integrity


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _safe_str(x):
    try:
        return str(x)
    except Exception:
        return "<unprintable>"


def _top_k_dict(d: dict, k: int = 10):
    """
    Returns list of {"name":..., "count":...} sorted desc.
    """
    items = []
    for name, count in (d or {}).items():
        items.append({"name": _safe_str(name), "count": int(count)})
    items.sort(key=lambda x: x["count"], reverse=True)
    return items[:k]


def _count_missing_by_column(df: pd.DataFrame):
    """
    Returns list:
    [{"column":..., "missing":..., "percent":...}, ...]
    sorted by missing desc
    """
    if df is None or df.empty:
        return []

    n = len(df)
    out = []

    for col in df.columns:
        miss = int(df[col].isna().sum())
        pct = float((miss / n) * 100) if n > 0 else 0.0
        out.append({
            "column": str(col),
            "missing": miss,
            "percent": float(round(pct, 2)),
        })

    out.sort(key=lambda x: x["missing"], reverse=True)
    return out


def _count_issues_by_column(issues: list, include_global: bool = False):
    """
    Counts issues by column.

    - include_global=False -> counts only row-level issues
    - include_global=True  -> counts row-level + global issues
    """
    counts = {}

    for it in issues or []:
        if not isinstance(it, dict):
            continue

        is_global = it.get("row") is None

        if is_global and not include_global:
            continue

        col = it.get("column")
        if col is None:
            continue

        col = str(col).strip()
        if not col:
            continue

        counts[col] = counts.get(col, 0) + 1

    return counts


def _count_issue_types(issues: list, mode: str = "all"):
    """
    Counts issue text frequency.

    mode:
      - "all"
      - "row"
      - "global"
    """
    counts = {}

    for it in issues or []:
        if not isinstance(it, dict):
            continue

        is_global = it.get("row") is None

        if mode == "row" and is_global:
            continue
        if mode == "global" and not is_global:
            continue

        msg = it.get("issue")
        if msg is None:
            continue

        msg = str(msg).strip()
        if not msg:
            continue

        counts[msg] = counts.get(msg, 0) + 1

    return counts


# =========================================================
# NEW: Severity weighting + stable issue IDs
# =========================================================

_SEVERITY_WEIGHTS = {
    "error": 2.5,   # strongest penalty
    "warn": 1.0,    # normal
    "warning": 1.0,
    "info": 0.25,   # very light
}


def _normalize_severity(sev):
    s = str(sev or "warn").strip().lower()
    if s in ("warning",):
        return "warn"
    if s not in ("error", "warn", "info"):
        return "warn"
    return s


def _issue_id(it: dict):
    """
    Creates a stable ID for an issue.
    Used by GUI for drill-down ("clickable issue IDs").
    """
    row = it.get("row")
    col = it.get("column")
    stage = it.get("stage")
    issue = it.get("issue")

    payload = f"{row}|{col}|{stage}|{issue}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _severity_weighted_score(row_issues_for_row: list):
    """
    Row severity score:
    - severity weight (error>warn>info)
    - confidence multiplier
    - count matters too
    """
    if not row_issues_for_row:
        return 0.0

    total = 0.0
    for it in row_issues_for_row:
        sev = _normalize_severity(it.get("severity", "warn"))
        w = float(_SEVERITY_WEIGHTS.get(sev, 1.0))

        conf = it.get("confidence", 0.6)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.6
        conf = max(0.1, min(1.0, conf))

        # each issue contributes:
        # base severity + confidence scaling
        total += w * (0.75 + 0.25 * conf)

    # count matters
    return float(len(row_issues_for_row)) + total


def _top_worst_rows(row_issues: list, k: int = 5):
    """
    Finds top worst rows using severity weighting.
    """
    by_row = {}

    for it in row_issues or []:
        r = it.get("row")
        if r is None:
            continue
        by_row.setdefault(r, []).append(it)

    ranked = []
    for r, its in by_row.items():
        severity = _severity_weighted_score(its)
        ranked.append((severity, r, its))

    ranked.sort(key=lambda x: x[0], reverse=True)

    out = []
    for severity, r, its in ranked[:k]:
        issue_names = {}
        for it in its:
            msg = str(it.get("issue", "")).strip()
            if not msg:
                continue
            issue_names[msg] = issue_names.get(msg, 0) + 1

        out.append({
            "row": r,
            "severity_score": float(round(severity, 3)),
            "issue_count": int(len(its)),
            "top_issue_types": _top_k_dict(issue_names, k=5),
        })

    return out


def _health_grade(overall_score: float):
    """
    Converts 0..100 score into A/B/C/D/F
    """
    try:
        s = float(overall_score)
    except Exception:
        s = 0.0

    if s >= 95:
        return "A"
    if s >= 85:
        return "B"
    if s >= 70:
        return "C"
    if s >= 55:
        return "D"
    return "F"


# =========================================================
# NEW: Drill-down analytics
# =========================================================

def _build_drilldown(issues: list, top_columns: list, k_examples_per_issue=3):
    """
    Creates a drilldown structure:
    - top issues per column
    - example rows
    - stable issue_id for GUI
    """
    # group: col -> issue_text -> list(issues)
    grouped = {}

    for it in issues or []:
        if not isinstance(it, dict):
            continue

        row = it.get("row")
        col = it.get("column")
        msg = it.get("issue")

        if row is None:
            continue  # drilldown is for row-level

        if col is None or msg is None:
            continue

        col = str(col).strip()
        msg = str(msg).strip()

        if not col or not msg:
            continue

        grouped.setdefault(col, {}).setdefault(msg, []).append(it)

    # only keep for top columns
    out = {}

    for item in top_columns or []:
        col = item.get("name")
        if not col:
            continue

        col = str(col)
        if col not in grouped:
            continue

        issue_map = grouped[col]
        issue_items = []

        for issue_text, its in issue_map.items():
            # build examples
            examples = []
            for ex in its[:k_examples_per_issue]:
                ex_id = _issue_id(ex)
                examples.append({
                    "issue_id": ex_id,
                    "row": ex.get("row"),
                    "severity": _normalize_severity(ex.get("severity", "warn")),
                    "confidence": float(ex.get("confidence", 0.6)),
                    "details": ex.get("details", {}),
                })

            # severity mix
            sev_counts = {"error": 0, "warn": 0, "info": 0}
            for ex in its:
                sev = _normalize_severity(ex.get("severity", "warn"))
                sev_counts[sev] = sev_counts.get(sev, 0) + 1

            issue_items.append({
                "issue": str(issue_text),
                "count": int(len(its)),
                "severity_mix": sev_counts,
                "examples": examples,
            })

        issue_items.sort(key=lambda x: x["count"], reverse=True)
        out[col] = issue_items[:10]

    return out


# =========================================================
# NEW: Trend analysis (history)
# =========================================================

def _load_history(path: str):
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _save_history(path: str, history: list):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception:
        pass


def _dataset_fingerprint(df: pd.DataFrame):
    """
    Very lightweight fingerprint:
    - shape
    - column names
    """
    try:
        cols = "|".join([str(c) for c in df.columns])
        payload = f"{len(df)}|{len(df.columns)}|{cols}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]
    except Exception:
        return "unknown"


def _trend_analysis(df: pd.DataFrame, current_score: float, history_path: str):
    """
    Compare to previous run for the same dataset fingerprint.
    """
    fp = _dataset_fingerprint(df)
    now = datetime.datetime.utcnow().isoformat()

    history = _load_history(history_path)

    # filter same dataset
    same = [h for h in history if h.get("fingerprint") == fp and "score" in h]

    last = same[-1] if same else None

    trend = {
        "history_enabled": True,
        "history_path": str(history_path),
        "fingerprint": str(fp),
        "previous_score": float(last.get("score")) if last else None,
        "current_score": float(current_score),
        "delta_score": None,
        "message": None,
        "timestamp_utc": now,
    }

    if last:
        delta = float(current_score) - float(last.get("score", 0.0))
        trend["delta_score"] = float(round(delta, 2))

        if delta > 0:
            trend["message"] = f"Score improved by {abs(delta):.2f} points since last run."
        elif delta < 0:
            trend["message"] = f"Score dropped by {abs(delta):.2f} points since last run."
        else:
            trend["message"] = "Score unchanged since last run."

    # append new record
    history.append({
        "fingerprint": fp,
        "timestamp_utc": now,
        "score": float(current_score),
    })

    # keep last 200 entries
    history = history[-200:]
    _save_history(history_path, history)

    return trend


# -------------------------------------------------
# Recommended actions (unchanged)
# -------------------------------------------------
def _recommended_actions(df: pd.DataFrame, issues: list, scorecard: dict):
    actions = []

    missing_cols = scorecard.get("analytics", {}).get("columns_with_most_missing", [])
    if missing_cols:
        worst_missing = [c for c in missing_cols if c.get("missing", 0) > 0][:3]
        if worst_missing:
            actions.append({
                "priority": "high",
                "action": "Review missing values in key columns",
                "reason": f"Missing values found in: {[x['column'] for x in worst_missing]}",
                "suggestion": "Decide per column: keep NaN, median fill, or ask user."
            })

    top_row_issue_types = scorecard.get("analytics", {}).get("most_frequent_issue_types_row_only", [])
    if top_row_issue_types:
        top3 = top_row_issue_types[:3]
        actions.append({
            "priority": "high",
            "action": "Fix the most common invalid patterns",
            "reason": f"Top issues: {[x['name'] for x in top3]}",
            "suggestion": "Add stronger auto-fixes OR ask user for those columns."
        })

    worst_cols = scorecard.get("analytics", {}).get("top_10_worst_columns_row_only", [])
    if worst_cols:
        top3 = worst_cols[:3]
        actions.append({
            "priority": "medium",
            "action": "Focus validation improvements on worst columns",
            "reason": f"Most problematic columns: {[x['name'] for x in top3]}",
            "suggestion": "Tune semantic/pattern rules for these columns first."
        })

    warnings_count = int(scorecard.get("warnings", {}).get("count", 0))
    if warnings_count > 0:
        actions.append({
            "priority": "medium",
            "action": "Review global warnings (row=None issues)",
            "reason": f"{warnings_count} warnings detected (ambiguous types, rule skips, crashes).",
            "suggestion": "These do not reduce score but indicate weak inference or unstable rules."
        })

    overall = float(scorecard.get("overall_score", 0.0))
    if overall < 70:
        actions.append({
            "priority": "high",
            "action": "Increase strictness or add more validation rules",
            "reason": f"Overall quality score is low ({overall}).",
            "suggestion": "Enable stricter semantic threshold, enforce patterns, and add uniqueness checks."
        })

    if overall >= 95 and warnings_count == 0:
        actions.append({
            "priority": "low",
            "action": "Dataset looks clean",
            "reason": "High score and no warnings.",
            "suggestion": "You can safely export cleaned CSV and proceed to analytics/ML."
        })

    seen = set()
    dedup = []
    for a in actions:
        key = a.get("action", "")
        if key in seen:
            continue
        seen.add(key)
        dedup.append(a)

    return dedup


# -------------------------------------------------
# Stage 6: Data Quality Scorecard
# -------------------------------------------------
def compute_data_quality_scorecard(
    df: pd.DataFrame,
    issues: list,
    history_path: str = "quality_history.json",
):
    """
    Stage 6: Data Quality Scoring (SMART + ENTERPRISE)

    NEW:
    - Severity-weighted scoring (error > warn > info)
    - Trend analysis vs previous runs
    - Drill-down report per column with issue_id + example rows
    """

    n_rows = int(len(df))
    n_cols = int(len(df.columns))

    # -----------------------------
    # 0) Split issues
    # -----------------------------
    row_issues = []
    global_issues = []

    for it in issues or []:
        if not isinstance(it, dict):
            continue

        # attach stable issue_id for GUI
        if "issue_id" not in it:
            it["issue_id"] = _issue_id(it)

        if it.get("row") is None:
            global_issues.append(it)
        else:
            row_issues.append(it)

    # -----------------------------
    # 1) COMPLETENESS
    # -----------------------------
    total_cells = n_rows * n_cols if n_rows > 0 and n_cols > 0 else 0

    if total_cells == 0:
        completeness_ratio = 0.0
        non_null_cells = 0
    else:
        non_null_cells = int(df.notna().sum().sum())
        completeness_ratio = float(non_null_cells / total_cells)

    # -----------------------------
    # 2) ISSUE COUNTS BY STAGE
    # -----------------------------
    stage_counts_all = {}
    stage_counts_row = {}
    stage_counts_global = {}

    for it in issues or []:
        if not isinstance(it, dict):
            continue

        stage = it.get("stage", "unknown")
        stage_counts_all[stage] = stage_counts_all.get(stage, 0) + 1

        if it.get("row") is None:
            stage_counts_global[stage] = stage_counts_global.get(stage, 0) + 1
        else:
            stage_counts_row[stage] = stage_counts_row.get(stage, 0) + 1

    def _count_row(stage_name):
        return int(stage_counts_row.get(stage_name, 0))

    # -----------------------------
    # 3) VALIDITY SCORE (severity weighted)
    # -----------------------------
    # Previously: semantic_issues count
    # Now: severity-weighted "semantic penalty"
    semantic_penalty = 0.0
    for it in row_issues:
        if it.get("stage") != "semantic":
            continue
        sev = _normalize_severity(it.get("severity", "warn"))
        w = float(_SEVERITY_WEIGHTS.get(sev, 1.0))
        conf = it.get("confidence", 0.6)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.6
        conf = max(0.1, min(1.0, conf))
        semantic_penalty += w * conf

    if total_cells == 0:
        validity_ratio = 0.0
    else:
        # normalize penalty by total cells (keeps score stable)
        invalid_rate = min(1.0, semantic_penalty / total_cells)
        validity_ratio = float(1.0 - invalid_rate)

    # -----------------------------
    # 4) PATTERN SCORE (severity weighted)
    # -----------------------------
    pattern_penalty = 0.0
    for it in row_issues:
        if it.get("stage") != "pattern":
            continue
        sev = _normalize_severity(it.get("severity", "warn"))
        w = float(_SEVERITY_WEIGHTS.get(sev, 1.0))
        conf = it.get("confidence", 0.6)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.6
        conf = max(0.1, min(1.0, conf))
        pattern_penalty += w * conf

    if total_cells == 0:
        pattern_ratio = 0.0
    else:
        pat_rate = min(1.0, pattern_penalty / total_cells)
        pattern_ratio = float(1.0 - pat_rate)

    # -----------------------------
    # 5) CONSISTENCY SCORE (severity weighted)
    # -----------------------------
    relationship_penalty = 0.0
    for it in row_issues:
        if it.get("stage") != "relationship":
            continue
        sev = _normalize_severity(it.get("severity", "warn"))
        w = float(_SEVERITY_WEIGHTS.get(sev, 1.0))
        conf = it.get("confidence", 0.6)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.6
        conf = max(0.1, min(1.0, conf))
        relationship_penalty += w * conf

    if total_cells == 0:
        consistency_ratio = 0.0
    else:
        rel_rate = min(1.0, relationship_penalty / max(n_rows, 1))
        consistency_ratio = float(1.0 - rel_rate)

    # -----------------------------
    # 6) UNIQUENESS SCORE
    # -----------------------------
    duplicate_issues = 0
    for it in row_issues:
        msg = str(it.get("issue", "")).lower()
        if "duplicate" in msg:
            duplicate_issues += 1

    if n_rows == 0:
        uniqueness_ratio = 0.0
    else:
        dup_rate = min(1.0, duplicate_issues / n_rows)
        uniqueness_ratio = float(1.0 - dup_rate)

    # -----------------------------
    # 7) OVERALL SCORE (weighted)
    # -----------------------------
    weights = {
        "completeness": 0.25,
        "validity": 0.30,
        "consistency": 0.20,
        "uniqueness": 0.15,
        "pattern": 0.10,
    }

    overall_ratio = (
        weights["completeness"] * completeness_ratio +
        weights["validity"] * validity_ratio +
        weights["consistency"] * consistency_ratio +
        weights["uniqueness"] * uniqueness_ratio +
        weights["pattern"] * pattern_ratio
    )

    overall_score = float(round(overall_ratio * 100, 2))
    grade = _health_grade(overall_score)

    # -----------------------------
    # 8) Analytics
    # -----------------------------
    missing_cols = _count_missing_by_column(df)
    worst_missing_cols = [x for x in missing_cols if x["missing"] > 0][:10]

    issues_by_column_row = _count_issues_by_column(issues, include_global=False)
    issues_by_column_all = _count_issues_by_column(issues, include_global=True)

    worst_columns_row = _top_k_dict(issues_by_column_row, k=10)
    worst_columns_all = _top_k_dict(issues_by_column_all, k=10)

    issue_type_counts_all = _count_issue_types(issues, mode="all")
    issue_type_counts_row = _count_issue_types(issues, mode="row")
    issue_type_counts_global = _count_issue_types(issues, mode="global")

    top_issue_types_all = _top_k_dict(issue_type_counts_all, k=10)
    top_issue_types_row = _top_k_dict(issue_type_counts_row, k=10)
    top_issue_types_global = _top_k_dict(issue_type_counts_global, k=10)

    worst_rows = _top_worst_rows(row_issues, k=5)

    # NEW: drilldown
    drilldown = _build_drilldown(
        row_issues,
        top_columns=worst_columns_row,
        k_examples_per_issue=3
    )

    # NEW: trend
    trend = _trend_analysis(df, overall_score, history_path=history_path)

    # -----------------------------
    # 9) Output scorecard
    # -----------------------------
    scorecard = {
        "overall_score": overall_score,
        "health_grade": grade,
        "dimensions": {
            "completeness": {
                "score": float(round(completeness_ratio * 100, 2)),
                "non_null_cells": int(non_null_cells),
                "total_cells": int(total_cells),
                "missing_cells": int(total_cells - non_null_cells) if total_cells > 0 else 0,
            },
            "validity": {
                "score": float(round(validity_ratio * 100, 2)),
                "semantic_penalty": float(round(semantic_penalty, 4)),
            },
            "pattern": {
                "score": float(round(pattern_ratio * 100, 2)),
                "pattern_penalty": float(round(pattern_penalty, 4)),
            },
            "consistency": {
                "score": float(round(consistency_ratio * 100, 2)),
                "relationship_penalty": float(round(relationship_penalty, 4)),
            },
            "uniqueness": {
                "score": float(round(uniqueness_ratio * 100, 2)),
                "duplicate_issues": int(duplicate_issues),
            },
        },
        "summary": {
            "rows": int(n_rows),
            "columns": int(n_cols),

            "total_issues": int(len(issues or [])),
            "row_issues_used_in_score": int(len(row_issues)),
            "global_issues_not_scored": int(len(global_issues)),

            "issues_by_stage_all": stage_counts_all,
            "issues_by_stage_row": stage_counts_row,
            "issues_by_stage_global": stage_counts_global,
        },
        "warnings": {
            "count": int(len(global_issues)),
            "by_stage": stage_counts_global,
        },
        "analytics": {
            "columns_with_most_missing": worst_missing_cols,

            "top_10_worst_columns_row_only": worst_columns_row,
            "top_10_worst_columns_all": worst_columns_all,

            "most_frequent_issue_types_all": top_issue_types_all,
            "most_frequent_issue_types_row_only": top_issue_types_row,
            "most_frequent_issue_types_global_only": top_issue_types_global,

            "top_5_worst_rows": worst_rows,

            # NEW
            "drilldown_by_column": drilldown,
        },

        # NEW
        "trend": trend,
    }

    # -----------------------------
    # 10) Recommended actions
    # -----------------------------
    scorecard["recommended_actions"] = _recommended_actions(df, issues, scorecard)

    return scorecard


# =========================================================
# Transformer Wrapper (for sklearn Pipeline)
# =========================================================
class DataQualityScorecard(BaseEstimator, TransformerMixin):
    """
    Stage 6 transformer.

    REPORTING ONLY:
    - does not modify df
    - computes scorecard from issues

    SMART MODE:
    - row-level issues affect score
    - global issues reported separately

    ENTERPRISE:
    - worst columns
    - worst rows
    - grade
    - recommended actions

    NEW:
    - trend analysis
    - drilldown
    """

    def __init__(
        self,
        issues: list | None = None,
        history_path: str = "quality_history.json",
    ):
        self.issues_input = issues
        self.history_path = str(history_path)

        self.issues = []
        self.scorecard_ = {}
        self.last_report_ = {}

    def set_issues(self, issues: list):
        self.issues_input = issues

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        original_df = X.copy()
        df = X.copy()

        issues = self.issues_input or []
        self.issues = issues

        self.scorecard_ = compute_data_quality_scorecard(
            df,
            issues,
            history_path=self.history_path,
        )

        summary = self.scorecard_.get("summary", {})
        warnings = self.scorecard_.get("warnings", {})
        trend = self.scorecard_.get("trend", {}) or {}

        self.last_report_ = {
            "overall_score": float(self.scorecard_.get("overall_score", 0.0)),
            "health_grade": str(self.scorecard_.get("health_grade", "F")),
            "total_issues_used": int(len(issues)),
            "row_issues_used_in_score": int(summary.get("row_issues_used_in_score", 0)),
            "global_issues_not_scored": int(summary.get("global_issues_not_scored", 0)),
            "warnings_count": int(warnings.get("count", 0)),

            # NEW
            "trend_message": trend.get("message"),
            "delta_score": trend.get("delta_score"),
            "previous_score": trend.get("previous_score"),
        }

        _check_df_integrity(original_df, df, "DataQualityScorecard")
        return df
