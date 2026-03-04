# =========================================================
# cleaningStage5.py
# Stage 5: Outlier-Aware Imputation
# =========================================================

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from pipeline_utils import _check_df_integrity


# -------------------------------------------------
# Helpers (same style as your other stages)
# -------------------------------------------------
def _make_issue(
    row,
    column,
    issue,
    confidence,
    explanation,
    details=None,
    suggested_fix=None,
    stage="imputation",
):
    # CHANGE #2: force row to int if possible
    safe_row = None
    if row is not None:
        try:
            safe_row = int(row)
        except Exception:
            safe_row = row

    # CHANGE #1: add severity
    # Imputation is not an "error" — it is a repair action.
    severity = "info"

    return {
        "row": safe_row,
        "column": column,
        "stage": stage,
        "severity": severity,
        "issue": issue,
        "confidence": float(confidence),
        "explanation": explanation,
        "details": details or {},
        "suggested_fix": suggested_fix or {},
    }


def _to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce")


def _infer_numeric_range(series: pd.Series):
    """
    Outlier-aware bounds using robust quantiles.
    """
    x = series.dropna()
    if x.empty:
        return None

    low = x.quantile(0.01)
    high = x.quantile(0.99)

    # fallback if degenerate
    if low == high:
        low = x.min()
        high = x.max()

    return float(low), float(high)


def _clip_to_range(v, low, high):
    if pd.isna(v):
        return v
    return float(np.clip(float(v), low, high))


def _choose_group_columns(df: pd.DataFrame, max_group_cols: int = 3):
    """
    Picks good categorical columns for group-based imputation.

    Heuristic:
    - low cardinality
    - not mostly unique
    - not huge text

    Fix:
    - limit to max_group_cols to avoid too-specific grouping
    """
    candidates = []
    n = len(df)

    if n == 0:
        return []

    for col in df.columns:
        s = df[col]

        # skip numeric
        if pd.api.types.is_numeric_dtype(s):
            continue

        # skip datetime
        if pd.api.types.is_datetime64_any_dtype(s.dtype):
            continue

        x = s.dropna().astype(str).str.strip()
        if x.empty:
            continue

        unique_ratio = x.nunique() / len(x)

        if unique_ratio > 0.6:
            continue

        if x.nunique() > 50:
            continue

        if x.str.len().mean() > 25:
            continue

        # scoring: prefer lower unique_ratio and lower cardinality
        score = (unique_ratio, x.nunique())
        candidates.append((score, col))

    # sort best first
    candidates.sort(key=lambda t: t[0])
    group_cols = [col for _, col in candidates[:max_group_cols]]

    return group_cols


def _knn_impute_value(target_index, target_col: str, df_num: pd.DataFrame, k=5):
    """
    Lightweight kNN imputation using numeric columns only.

    Fix:
    - use target_index consistently (df index), not positional row number.
    """
    if target_col not in df_num.columns:
        return None, None

    if target_index not in df_num.index:
        return None, None

    row_vec = df_num.loc[target_index]

    # We need at least 1 feature besides the target column
    feature_cols = [c for c in df_num.columns if c != target_col]
    if not feature_cols:
        return None, None

    # only consider rows where target_col is not missing
    candidates = df_num[df_num[target_col].notna()]
    if candidates.empty:
        return None, None

    A = candidates[feature_cols]
    b = row_vec[feature_cols]

    diff = (A - b).abs()

    valid_mask = A.notna() & b.notna()
    valid_counts = valid_mask.sum(axis=1).replace(0, np.nan)

    dist = (diff.where(valid_mask).sum(axis=1) / valid_counts).dropna()
    if dist.empty:
        return None, None

    nearest_idx = dist.nsmallest(k).index

    values = candidates.loc[nearest_idx, target_col].dropna()
    if values.empty:
        return None, None

    return float(values.median()), list(nearest_idx)


# -------------------------------------------------
# Stage 5: Outlier-Aware Imputation
# -------------------------------------------------
def impute_outlier_aware(
    df: pd.DataFrame,
    max_k=5,
    max_missing_rate: float = 0.50,   # ✅ NEW
    min_group_size: int = 10,         # ✅ NEW
):
    """
    Stage 5:
    - Fills missing numeric values using:
      1) Group-based median (best)
      2) kNN fallback (numeric similarity)
      3) Global median fallback
    - Outlier-aware: clips filled values to inferred range (1%-99%)
    - Returns: (df_imputed, imputation_issues, metrics)

    NEW SAFETY:
    - Skip imputation if column missing rate > max_missing_rate
    - Skip group imputation if group has < min_group_size samples
    - Metrics: imputed counts per method + confidence per method
    """

    df_out = df.copy()
    issues = []

    # ✅ NEW: metrics
    metrics = {
        "max_missing_rate": float(max_missing_rate),
        "min_group_size": int(min_group_size),
        "skipped_sparse_columns": [],
        "too_sparse_reason": {},
        "imputed_counts": {
            "group_median": 0,
            "knn_numeric": 0,
            "global_median": 0,
        },
        "confidence_by_method": {
            "group_median": 0.85,
            "knn_numeric": 0.75,
            "global_median": 0.60,
        }
    }

    if df_out.empty:
        return df_out, issues, metrics

    # Only numeric columns are imputed here
    numeric_cols = [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c])]

    # Also include numeric-like object columns (optional)
    for c in df_out.columns:
        if c in numeric_cols:
            continue
        s = df_out[c]
        if s.dtype == "object":
            num = _to_numeric_safe(s)
            if num.notna().mean() > 0.8:
                df_out[c] = num
                numeric_cols.append(c)

    if not numeric_cols:
        return df_out, issues, metrics

    # group columns for conditional imputation
    group_cols = _choose_group_columns(df_out, max_group_cols=3)

    # precompute numeric-only df for kNN
    df_num = df_out[numeric_cols].copy()

    # -------------------------------------------------
    # NEW: column sparsity check (skip columns >50% missing)
    # -------------------------------------------------
    sparse_cols = set()
    for target_col in numeric_cols:
        miss_rate = float(df_out[target_col].isna().mean())
        if miss_rate > float(max_missing_rate):
            sparse_cols.add(target_col)
            metrics["skipped_sparse_columns"].append(target_col)
            metrics["too_sparse_reason"][target_col] = (
                f"too sparse to safely impute (missing_rate={miss_rate:.1%} > {max_missing_rate:.0%})"
            )

            issues.append(_make_issue(
                row=None,
                column=target_col,
                issue="too sparse to safely impute",
                confidence=0.95,
                explanation=(
                    f"Skipped imputation for '{target_col}' because missing_rate={miss_rate:.1%} "
                    f"is above max_missing_rate={max_missing_rate:.0%}."
                ),
                details={
                    "method": "skip_sparse",
                    "missing_rate": miss_rate,
                    "max_missing_rate": float(max_missing_rate),
                },
                suggested_fix={"action": "skip_imputation"}
            ))

    # -------------------------------------------------
    # 1) GROUP-BASED IMPUTATION (median)
    # -------------------------------------------------
    if group_cols:
        for target_col in numeric_cols:
            if target_col in sparse_cols:
                continue

            s = df_out[target_col]
            if s.isna().sum() == 0:
                continue

            low_high = _infer_numeric_range(s)
            if low_high is None:
                continue
            low, high = low_high

            # NEW: group size check
            group_sizes = df_out.groupby(group_cols, dropna=False).size()
            group_sizes = group_sizes.to_dict()

            group_medians = df_out.groupby(group_cols, dropna=False)[target_col].transform("median")

            m_missing = s.isna() & group_medians.notna()

            for idx in df_out.index[m_missing]:
                # Determine group key for this row
                try:
                    key = tuple(df_out.loc[idx, group_cols].tolist())
                except Exception:
                    key = None

                gsize = None
                if key is not None:
                    gsize = int(group_sizes.get(key, 0))

                # ✅ NEW: skip tiny groups
                if gsize is not None and gsize < int(min_group_size):
                    issues.append(_make_issue(
                        row=idx,
                        column=target_col,
                        issue="skipped group imputation (group too small)",
                        confidence=0.9,
                        explanation=(
                            f"Skipped group median for '{target_col}' because group size={gsize} "
                            f"is below min_group_size={min_group_size}."
                        ),
                        details={
                            "method": "group_median_skip_small_group",
                            "group_columns": group_cols,
                            "group_size": gsize,
                            "min_group_size": int(min_group_size),
                        },
                        suggested_fix={"action": "skip_group_imputation"}
                    ))
                    continue

                filled = float(group_medians.loc[idx])
                filled = _clip_to_range(filled, low, high)

                df_out.loc[idx, target_col] = filled
                df_num.loc[idx, target_col] = filled

                metrics["imputed_counts"]["group_median"] += 1

                issues.append(_make_issue(
                    row=idx,
                    column=target_col,
                    issue="filled missing value using group median",
                    confidence=0.85,
                    explanation=f"Filled missing '{target_col}' using median grouped by {group_cols}.",
                    details={
                        "method": "group_median",
                        "group_columns": group_cols,
                        "filled_value": filled,
                        "clip_range": [low, high],
                        "min_group_size": int(min_group_size),
                        "group_size": gsize,
                    },
                    suggested_fix={"action": "filled"}
                ))

    # -------------------------------------------------
    # 2) kNN FALLBACK IMPUTATION
    # -------------------------------------------------
    for target_col in numeric_cols:
        if target_col in sparse_cols:
            continue

        s = df_out[target_col]

        m_missing = s.isna()
        if m_missing.sum() == 0:
            continue

        low_high = _infer_numeric_range(s)
        if low_high is None:
            continue
        low, high = low_high

        for idx in df_out.index[m_missing]:
            value, neighbors = _knn_impute_value(idx, target_col, df_num, k=max_k)

            if value is None:
                continue

            value = _clip_to_range(value, low, high)

            df_out.loc[idx, target_col] = value
            df_num.loc[idx, target_col] = value

            metrics["imputed_counts"]["knn_numeric"] += 1

            issues.append(_make_issue(
                row=idx,
                column=target_col,
                issue="filled missing value using kNN (numeric similarity)",
                confidence=0.75,
                explanation=f"Filled '{target_col}' using median of {max_k} nearest rows based on numeric similarity.",
                details={
                    "method": "knn_numeric",
                    "k": max_k,
                    "neighbors": neighbors,
                    "filled_value": value,
                    "clip_range": [low, high],
                },
                suggested_fix={"action": "filled"}
            ))

    # -------------------------------------------------
    # 3) FINAL FALLBACK: GLOBAL MEDIAN (last resort)
    # -------------------------------------------------
    for target_col in numeric_cols:
        if target_col in sparse_cols:
            continue

        s = df_out[target_col]
        m_missing = s.isna()

        if m_missing.sum() == 0:
            continue

        global_median = s.median()
        if pd.isna(global_median):
            continue

        low_high = _infer_numeric_range(s)
        if low_high is None:
            continue
        low, high = low_high

        fill_value = _clip_to_range(float(global_median), low, high)

        for idx in df_out.index[m_missing]:
            df_out.loc[idx, target_col] = fill_value
            metrics["imputed_counts"]["global_median"] += 1

            issues.append(_make_issue(
                row=idx,
                column=target_col,
                issue="filled missing value using global median (fallback)",
                confidence=0.6,
                explanation=f"Filled '{target_col}' using global median because no group/kNN neighbors were available.",
                details={
                    "method": "global_median",
                    "filled_value": float(fill_value),
                    "clip_range": [low, high],
                },
                suggested_fix={"action": "filled"}
            ))

    return df_out, issues, metrics


# =========================================================
# Transformer Wrapper (for sklearn Pipeline)
# =========================================================
class OutlierAwareImputer(BaseEstimator, TransformerMixin):
    """
    Stage 5 transformer.

    Purpose:
    - Run impute_outlier_aware(df)
    - Store issues + report
    - Return imputed df

    Controls:
    - enabled=False -> does nothing (keeps NaN, "truthful cleaning")
    - max_k controls kNN neighbors

    NEW:
    - max_missing_rate (default 0.50)
    - min_group_size (default 10)
    - metrics in report
    """

    def __init__(
        self,
        enabled: bool = True,
        max_k: int = 5,
        max_missing_rate: float = 0.50,   # ✅ NEW
        min_group_size: int = 10,         # ✅ NEW
        impute_missing: bool = True,      # ✅ supports your pipeline_mode toggle
    ):
        self.enabled = bool(enabled)
        self.max_k = int(max_k)

        # NEW
        self.max_missing_rate = float(max_missing_rate)
        self.min_group_size = int(min_group_size)

        # NEW: so Stage2 / pipeline can disable Stage5 without breaking init
        self.impute_missing = bool(impute_missing)

        self.issues = []
        self.last_report_ = {}

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        original_df = X.copy()
        df = X.copy()

        self.issues = []
        self.last_report_ = {}

        # If stage is disabled OR pipeline told us not to impute
        if (not self.enabled) or (not self.impute_missing):
            self.last_report_ = {
                "imputation_enabled": False,
                "imputation_issue_count": 0,
                "reason": "disabled" if not self.enabled else "impute_missing=False",
                "max_k": int(self.max_k),
                "max_missing_rate": float(self.max_missing_rate),
                "min_group_size": int(self.min_group_size),
            }
            _check_df_integrity(original_df, df, "OutlierAwareImputer")
            return df

        df_imputed, issues, metrics = impute_outlier_aware(
            df,
            max_k=self.max_k,
            max_missing_rate=self.max_missing_rate,
            min_group_size=self.min_group_size,
        )

        self.issues = issues

        # NEW: quality metrics in report
        self.last_report_ = {
            "imputation_enabled": True,
            "imputation_issue_count": int(len(issues)),
            "max_k": int(self.max_k),
            "max_missing_rate": float(self.max_missing_rate),
            "min_group_size": int(self.min_group_size),

            "imputation_metrics": metrics,
        }

        df_imputed = _check_df_integrity(original_df, df_imputed, "OutlierAwareImputer")
        return df_imputed
