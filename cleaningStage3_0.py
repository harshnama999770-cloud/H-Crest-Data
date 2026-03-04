# =========================================================
# cleaningStage3_0.py
# Stage 3.0: Universal Relationship-Based Validation (SAFE MODE)
#
# ENTERPRISE FIXES:
# - ✅ Fixed catastrophic Memory Explosion (OOM) on large datasets
# - ✅ Added MAX_ISSUES_PER_RULE cap to prevent dict blowout
# =========================================================

import pandas as pd
import numpy as np
import re

from sklearn.base import BaseEstimator, TransformerMixin
from pipeline_utils import _check_df_integrity

# =================================================
# HELPERS
# =================================================

_DATE_HINTS_START = {"start", "from", "begin", "checkin", "signup", "order", "placed", "created"}
_DATE_HINTS_END = {"end", "to", "finish", "checkout", "updated", "last", "delivery", "delivered", "return"}

# pairs that should not be matched together (common false positives)
_DATE_PAIR_BLOCKLIST = {
    ("created", "return"),
    ("created", "delivered"),
    ("created", "delivery"),
    ("signup", "return"),
    ("order", "return"),
}

_MIN_HINTS = {"min", "low", "start", "from"}
_MAX_HINTS = {"max", "high", "end", "to"}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_DIGITS_RE = re.compile(r"\D+")

# 🔴 FIX: Enterprise Memory Cap (Prevents OOM crashes on huge datasets)
MAX_ISSUES_PER_RULE = 5000


def _col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _norm_col(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_")


def _to_datetime_safe(s: pd.Series) -> pd.Series:
    """
    SAFE datetime conversion 
    - coerce invalid
    - utc to naive
    """
    dt = pd.to_datetime(s, errors="coerce", utc=True, dayfirst=True)
    try:
        dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    return dt


def _to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def _looks_like_date_col(colname: str) -> bool:
    n = _norm_col(colname)
    return any(k in n for k in ["date", "dt", "time", "timestamp"])


def _tokens(col: str) -> set:
    """
    tokenize column name into base words.
    """
    n = _norm_col(col)
    parts = [p for p in re.split(r"[_\W]+", n) if p]
    return set(parts)


def _remove_date_noise_tokens(toks: set) -> set:
    """
    remove generic date words so matching is about base entity.
    """
    noise = {
        "date", "dt", "time", "timestamp", "at",
        "created", "updated", "last", "first",
        "start", "end", "from", "to",
        "begin", "finish",
        "checkin", "checkout",
    }
    return set([t for t in toks if t not in noise])


def _pair_strength_score(start_col: str, end_col: str) -> tuple[str, float]:
    """
    Relationship strength scoring.
    Returns (level, confidence_multiplier).

    - high: exact base match
    - medium: partial base overlap
    - low: generic detection
    """
    ta = _remove_date_noise_tokens(_tokens(start_col))
    tb = _remove_date_noise_tokens(_tokens(end_col))

    if not ta or not tb:
        return "low", 0.85

    if ta == tb:
        return "high", 1.00

    overlap = len(ta & tb)
    if overlap >= 2:
        return "high", 1.00
    if overlap == 1:
        return "medium", 0.92

    return "low", 0.85


def _blocked_pair(start_col: str, end_col: str) -> bool:
    """
    ignore pairs like created_at vs return_date.
    """
    ta = _tokens(start_col)
    tb = _tokens(end_col)

    for a, b in _DATE_PAIR_BLOCKLIST:
        if a in ta and b in tb:
            return True
        if a in tb and b in ta:
            return True
    return False


def _make_issue(
    row,
    column,
    issue,
    confidence,
    explanation,
    details=None,
    suggested_fix=None,
    stage="relationship",
    severity=None,  
):
    # force row to int if possible
    safe_row = None
    if row is not None:
        try:
            safe_row = int(row)
        except Exception:
            safe_row = row

    # auto severity
    if severity is None:
        issue_l = str(issue or "").lower()
        if "crashed" in issue_l or "failed" in issue_l:
            severity = "warn"
        elif "invalid email" in issue_l or "before" in issue_l or "greater than" in issue_l:
            severity = "error"
        else:
            severity = "warn"

    return {
        "row": safe_row,
        "column": column,
        "stage": stage,
        "severity": str(severity),
        "issue": issue,
        "confidence": float(confidence),
        "explanation": explanation,
        "details": details or {},
        "suggested_fix": suggested_fix or {},
    }


# =================================================
# UNIVERSAL PAIR DETECTION
# =================================================


def _find_date_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        s = df[c]

        # already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(s):
            cols.append(c)
            continue

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            if _looks_like_date_col(c):
                if _to_datetime_safe(s).notna().mean() > 0.7:
                    cols.append(c)

    return cols


def _find_date_pairs(df: pd.DataFrame, date_cols):
    """
    Universal date pairing:
    - tries to pair start-like and end-like columns by base-word similarity
    - ignores unrelated pairs (created_at vs return_date)
    - provides relationship strength
    """
    norm_map = {_norm_col(c): c for c in date_cols}
    norm_cols = list(norm_map.keys())

    pairs = []

    for a in norm_cols:
        for b in norm_cols:
            if a == b:
                continue

            # must be start -> end
            if any(k in a for k in _DATE_HINTS_START) and any(k in b for k in _DATE_HINTS_END):

                start_col = norm_map[a]
                end_col = norm_map[b]

                if _blocked_pair(start_col, end_col):
                    continue

                strength, mult = _pair_strength_score(start_col, end_col)

                if strength == "low":
                    ta = _remove_date_noise_tokens(_tokens(start_col))
                    tb = _remove_date_noise_tokens(_tokens(end_col))
                    if len(ta & tb) == 0:
                        continue

                pairs.append((start_col, end_col, strength, mult))

    # remove duplicates
    unique = []
    seen = set()
    for x in pairs:
        key = (x[0], x[1])
        if key not in seen:
            unique.append(x)
            seen.add(key)

    return unique


def _find_numeric_minmax_pairs(df: pd.DataFrame):
    """
    Detect numeric pairs:
    - min vs max
    - low vs high
    - start vs end (numeric ranges)
    """
    pairs = []
    cols = list(df.columns)

    for c1 in cols:
        n1 = _norm_col(c1)

        if not any(h in n1 for h in _MIN_HINTS):
            continue

        for c2 in cols:
            if c1 == c2:
                continue

            n2 = _norm_col(c2)
            if not any(h in n2 for h in _MAX_HINTS):
                continue

            t1 = set(n1.split("_")) - _MIN_HINTS
            t2 = set(n2.split("_")) - _MAX_HINTS

            overlap = len(t1 & t2)

            if overlap < 1:
                continue

            if overlap >= 2:
                strength, mult = "high", 1.00
            else:
                strength, mult = "medium", 0.92

            pairs.append((c1, c2, strength, mult))

    out = []
    seen = set()
    for a, b, strength, mult in pairs:
        if (a, b) not in seen:
            out.append((a, b, strength, mult))
            seen.add((a, b))
    return out


def _find_percent_columns(df: pd.DataFrame):
    percent_cols = []
    for c in df.columns:
        n = _norm_col(c)
        if any(k in n for k in ["percent", "pct", "percentage", "%"]):
            percent_cols.append(c)
    return percent_cols


# =================================================
# CORE RELATIONSHIP VALIDATION (ISSUES ONLY)
# =================================================


def validate_relationships(
    df: pd.DataFrame,
    date_order_tolerance: int = 0,
    numeric_order_tolerance: float = 0.0,
):
    """
    Universal relationship validation:
    - never adds columns
    - never drops rows
    - never crashes
    """
    issues = []

    date_tol_days = int(date_order_tolerance)
    num_tol = float(numeric_order_tolerance)

    # =========================================================
    # 1) DATE ORDER RELATIONSHIPS (AUTO)
    # =========================================================
    try:
        date_cols = _find_date_columns(df)
        date_pairs = _find_date_pairs(df, date_cols)

        for start_col, end_col, strength, mult in date_pairs:
            start_dt = _to_datetime_safe(df[start_col])
            end_dt = _to_datetime_safe(df[end_col])

            if date_tol_days <= 0:
                m = start_dt.notna() & end_dt.notna() & (end_dt < start_dt)
            else:
                tol = pd.Timedelta(days=date_tol_days)
                m = start_dt.notna() & end_dt.notna() & (end_dt < (start_dt - tol))

            base_conf = 0.90 * mult

            issue_count = 0
            for idx in df.index[m]:
                # 🔴 FIX: Hard Memory Cap
                if issue_count >= MAX_ISSUES_PER_RULE:
                    break

                issues.append(_make_issue(
                    row=idx,
                    column=end_col,
                    issue=f"{end_col} is before {start_col}",
                    confidence=base_conf,
                    explanation=f"{end_col} must be >= {start_col} (tolerance={date_tol_days} days).",
                    details={
                        "rule": "date_order_auto",
                        "start_column": start_col,
                        "end_column": end_col,
                        "strength": strength,
                        "tolerance_days": date_tol_days,
                        "start_value": str(df.loc[idx, start_col]),
                        "end_value": str(df.loc[idx, end_col]),
                    },
                    suggested_fix={"action": "ask_user", "reason": "date order violation"},
                ))
                issue_count += 1
    except Exception as e:
        issues.append(_make_issue(
            row=None, column=None, issue="relationship rule failed: date order auto block crashed",
            confidence=0.4, explanation=f"Stage 3.0 date-order auto validation crashed: {str(e)}",
            details={"error": str(e)}, suggested_fix={"action": "skip_rule"}
        ))

    # =========================================================
    # 2) NUMERIC MIN/MAX ORDER (AUTO)
    # =========================================================
    try:
        numeric_pairs = _find_numeric_minmax_pairs(df)

        for min_col, max_col, strength, mult in numeric_pairs:
            a = _to_numeric_safe(df[min_col])
            b = _to_numeric_safe(df[max_col])

            m = a.notna() & b.notna() & (a > (b + num_tol))
            base_conf = 0.90 * mult

            issue_count = 0
            for idx in df.index[m]:
                # 🔴 FIX: Hard Memory Cap
                if issue_count >= MAX_ISSUES_PER_RULE:
                    break

                issues.append(_make_issue(
                    row=idx,
                    column=max_col,
                    issue=f"{min_col} is greater than {max_col}",
                    confidence=base_conf,
                    explanation=f"{min_col} must be <= {max_col} (tolerance={num_tol}).",
                    details={
                        "rule": "numeric_order_auto",
                        "min_column": min_col,
                        "max_column": max_col,
                        "strength": strength,
                        "tolerance": float(num_tol),
                        "min_value": str(df.loc[idx, min_col]),
                        "max_value": str(df.loc[idx, max_col]),
                    },
                    suggested_fix={"action": "swap_values", "reason": "min/max reversed"},
                ))
                issue_count += 1
    except Exception as e:
        issues.append(_make_issue(
            row=None, column=None, issue="relationship rule failed: numeric auto block crashed",
            confidence=0.4, explanation=f"Numeric auto validation crashed: {str(e)}",
            details={"error": str(e)}, suggested_fix={"action": "skip_rule"}
        ))

    # =========================================================
    # 3) PERCENT RANGE (AUTO)
    # =========================================================
    try:
        percent_cols = _find_percent_columns(df)

        for pc in percent_cols:
            x = _to_numeric_safe(df[pc])
            m = x.notna() & ((x < 0) | (x > 100))

            issue_count = 0
            for idx in df.index[m]:
                # 🔴 FIX: Hard Memory Cap
                if issue_count >= MAX_ISSUES_PER_RULE:
                    break

                issues.append(_make_issue(
                    row=idx,
                    column=pc,
                    issue="percentage out of range",
                    confidence=0.85,
                    explanation=f"{pc} must be between 0 and 100.",
                    details={
                        "rule": "percent_range_auto",
                        "expected": "0..100",
                        "actual": str(df.loc[idx, pc]),
                    },
                    suggested_fix={"action": "clip", "clip_to": [0, 100]},
                ))
                issue_count += 1
    except Exception as e:
        issues.append(_make_issue(
            row=None, column=None, issue="relationship rule failed: percent auto block crashed",
            confidence=0.4, explanation=f"Percent auto validation crashed: {str(e)}",
            details={"error": str(e)}, suggested_fix={"action": "skip_rule"}
        ))

    # =========================================================
    # 4) EMAIL FORMAT (AUTO)
    # =========================================================
    try:
        email_cols = []
        for c in df.columns:
            if "email" in _norm_col(c):
                email_cols.append(c)

        for ec in email_cols:
            s = df[ec].astype("string").str.strip()
            m = s.notna() & (~s.str.match(_EMAIL_RE, na=False))

            issue_count = 0
            for idx in df.index[m]:
                # 🔴 FIX: Hard Memory Cap
                if issue_count >= MAX_ISSUES_PER_RULE:
                    break

                issues.append(_make_issue(
                    row=idx,
                    column=ec,
                    issue="invalid email format",
                    confidence=0.9,
                    explanation=f"Invalid email format: '{df.loc[idx, ec]}'.",
                    details={"rule": "email_format_auto"},
                    suggested_fix={"action": "ask_user", "reason": "invalid email"},
                ))
                issue_count += 1
    except Exception as e:
        issues.append(_make_issue(
            row=None, column=None, issue="relationship rule failed: email auto block crashed",
            confidence=0.4, explanation=f"Email auto validation crashed: {str(e)}",
            details={"error": str(e)}, suggested_fix={"action": "skip_rule"}
        ))

    # =========================================================
    # 5) PHONE DIGITS (AUTO)
    # =========================================================
    try:
        phone_cols = []
        for c in df.columns:
            n = _norm_col(c)
            if "phone" in n or "mobile" in n or "contact" in n:
                phone_cols.append(c)

        for pc in phone_cols:
            s = df[pc].astype("string")
            digits = s.str.replace(_PHONE_DIGITS_RE, "", regex=True)

            m = digits.notna() & (digits.str.len() > 0)
            if m.mean() < 0.3:
                continue

            bad = m & ((digits.str.len() < 7) | (digits.str.len() > 15))

            issue_count = 0
            for idx in df.index[bad]:
                # 🔴 FIX: Hard Memory Cap
                if issue_count >= MAX_ISSUES_PER_RULE:
                    break

                issues.append(_make_issue(
                    row=idx,
                    column=pc,
                    issue="phone length suspicious",
                    confidence=0.75,
                    explanation=f"Phone number length looks invalid: '{df.loc[idx, pc]}'.",
                    details={"rule": "phone_length_auto", "expected": "7..15 digits"},
                    suggested_fix={"action": "ask_user", "reason": "phone invalid"},
                ))
                issue_count += 1
    except Exception as e:
        issues.append(_make_issue(
            row=None, column=None, issue="relationship rule failed: phone auto block crashed",
            confidence=0.4, explanation=f"Phone auto validation crashed: {str(e)}",
            details={"error": str(e)}, suggested_fix={"action": "skip_rule"}
        ))

    # =========================================================
    # 6) ORDER_ID NULL RATE CHECK (ENTERPRISE RULE)
    # =========================================================
    try:
        order_id_cols = []
        for c in df.columns:
            n = _norm_col(c)
            if "order_id" in n or "order_no" in n:
                order_id_cols.append(c)

        for col in order_id_cols:
            null_rate = df[col].isnull().mean()
            if null_rate > 0.05:
                issues.append(_make_issue(
                    row=None,
                    column=col,
                    issue="high null rate for primary key",
                    confidence=1.0,
                    explanation=(
                        f"Primary key column '{col}' has a high null rate ({null_rate:.2%}). "
                        f"This breaks traceability."
                    ),
                    details={"rule": "order_id_null_check", "null_rate": float(null_rate)},
                    suggested_fix={"action": "investigate_source", "reason": "high nulls in PK"},
                    severity="error", 
                ))
    except Exception as e:
        issues.append(_make_issue(
            row=None, column=None, issue="relationship rule failed: order_id null check crashed",
            confidence=0.4, explanation=f"Order ID null check validation crashed: {str(e)}",
            details={"error": str(e)}, suggested_fix={"action": "skip_rule"}
        ))

    return issues


# =========================================================
# TRANSFORMER (SAFE MODE FIX APPLY)
# =========================================================


class RelationshipValidator(BaseEstimator, TransformerMixin):
    """
    Universal Stage 3.0 transformer.

    SAFE MODE:
    - swap_values for min/max pairs
    - clip percent columns 0..100
    - everything else ask_user
    """

    def __init__(
        self,
        date_order_tolerance: int = 0,
        numeric_order_tolerance: float = 0.0,
    ):
        self.date_order_tolerance = int(date_order_tolerance)
        self.numeric_order_tolerance = float(numeric_order_tolerance)

        self.issues = []
        self.last_report_ = {}

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        original_df = X.copy()
        df = X.copy()

        self.issues = validate_relationships(
            df,
            date_order_tolerance=self.date_order_tolerance,
            numeric_order_tolerance=self.numeric_order_tolerance,
        )

        for it in self.issues:
            row = it.get("row")
            if row is None:
                continue

            col = it.get("column")
            if col is None or col not in df.columns or row not in df.index:
                continue

            fix = it.get("suggested_fix") or {}
            action = fix.get("action")

            # SAFE FIXES ONLY
            if action == "swap_values":
                details = it.get("details") or {}
                min_col = details.get("min_column")
                max_col = details.get("max_column")

                if min_col in df.columns and max_col in df.columns:
                    a = df.loc[row, min_col]
                    b = df.loc[row, max_col]
                    df.loc[row, min_col] = b
                    df.loc[row, max_col] = a

            elif action == "clip":
                clip_to = fix.get("clip_to")
                v = pd.to_numeric(df.loc[row, col], errors="coerce")

                if pd.notna(v) and isinstance(clip_to, list) and len(clip_to) == 2:
                    df.loc[row, col] = float(np.clip(v, clip_to[0], clip_to[1]))

        strength_counts = {"high": 0, "medium": 0, "low": 0}
        for it in self.issues:
            det = it.get("details") or {}
            st = str(det.get("strength", "")).lower()
            if st in strength_counts:
                strength_counts[st] += 1

        self.last_report_ = {
            "relationship_issue_count": int(len(self.issues)),
            "universal_mode": True,
            "date_order_tolerance": int(self.date_order_tolerance),
            "numeric_order_tolerance": float(self.numeric_order_tolerance),
            "strength_counts": strength_counts,
        }

        _check_df_integrity(original_df, df, "RelationshipValidator")
        return df