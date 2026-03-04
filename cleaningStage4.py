# =========================================================
# cleaningStage4.py
# Stage 4: Pattern Generalization (Validation + SAFE FIXES)
# UNIVERSAL SAFE VERSION (NEVER DESTROYS STAGE-2 DTYPES)
#
# ✅ ENTERPRISE ADD:
# - LLM pattern rules (pattern_regex) from llm_schema_infer.py
# - Uses LLM regex ONLY if safe + compiles + applies to enough values
# =========================================================

import pandas as pd
import numpy as np
import re
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from pipeline_utils import _check_df_integrity

# ✅ NEW: LLM schema helper
from llm_client import LocalLLMClient
from llm_schema_infer import infer_schema_with_llm, apply_llm_schema_safety


# -------------------------------------------------
# Pattern helpers
# -------------------------------------------------
def _norm_str(s: pd.Series):
    return s.astype("string").str.strip().str.lower()


def _make_issue(
    row,
    column,
    issue,
    confidence,
    explanation,
    details=None,
    suggested_fix=None,
    stage="pattern",
):
    safe_row = None
    if row is not None:
        try:
            safe_row = int(row)
        except Exception:
            safe_row = row

    issue_l = str(issue or "").lower()
    if "invalid" in issue_l or "too few digits" in issue_l or "missing domain" in issue_l:
        severity = "error"
    else:
        severity = "warn"

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


def _safe_series_str(series: pd.Series) -> pd.Series:
    """
    UNIVERSAL SAFE STRING CONVERSION

    - Keeps NaN as NaN (not "nan")
    - Converts non-null values to string
    - Always returns dtype="string"
    """
    s = series.copy()

    out = pd.Series(pd.NA, index=s.index, dtype="string")
    m = s.notna()
    out.loc[m] = s.loc[m].astype("string")

    return out


def _value_counts_ratio(s: pd.Series):
    vc = s.value_counts(dropna=True)
    if vc.empty:
        return None
    top = vc.index[0]
    top_ratio = vc.iloc[0] / vc.sum()
    return top, float(top_ratio), vc


def df_index(series: pd.Series, mask: pd.Series):
    return series.index[mask.fillna(False)]


# -------------------------------------------------
# Universal safe dtype guards
# -------------------------------------------------
def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _is_bool(series: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(series)


def _is_id_like(col: str) -> bool:
    n = str(col).strip().lower()
    if n == "id" or n.endswith("_id"):
        return True
    if any(k in n for k in ["uuid", "guid", "pan", "gst", "ifsc", "aadhar", "aadhaar"]):
        return True
    return False


# =========================================================
# Common regex patterns + custom support
# =========================================================

COMMON_PATTERNS = {
    "email": {
        "regex": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        "min_hit_ratio": 0.60,
        "severity": "error",
    },
    "phone_generic": {
        "regex": r"^\+?[\d\-\s\(\)]{7,20}$",
        "min_hit_ratio": 0.50,
        "severity": "warn",
    },
    "zip_us": {
        "regex": r"^\d{5}(-\d{4})?$",
        "min_hit_ratio": 0.70,
        "severity": "warn",
    },
    "pin_india": {
        "regex": r"^\d{6}$",
        "min_hit_ratio": 0.70,
        "severity": "warn",
    },
    "ssn_us": {
        "regex": r"^\d{3}-\d{2}-\d{4}$",
        "min_hit_ratio": 0.80,
        "severity": "error",
    },
}


def _pattern_confidence(hit_ratio: float) -> float:
    """
    Convert hit ratio into a confidence score.
    Stable scoring for UI.
    """
    hr = float(hit_ratio)
    if hr >= 0.95:
        return 0.95
    if hr >= 0.85:
        return 0.90
    if hr >= 0.70:
        return 0.80
    if hr >= 0.55:
        return 0.70
    return 0.60


def _regex_is_safe(regex: str) -> bool:
    """
    Enterprise regex safety:
    - avoid extremely long patterns
    - avoid catastrophic backtracking patterns (basic)
    """
    if regex is None:
        return False

    r = str(regex).strip()
    if not r:
        return False

    # too long -> reject
    if len(r) > 250:
        return False

    # very basic backtracking red flags
    bad_tokens = ["(.*.*)", "(.+.+)", "(.*)+", "(.+)+", "(.*){", "(.+){"]

    for t in bad_tokens:
        if t in r:
            return False

    return True


def _pattern_custom_regex(
    series: pd.Series,
    col: str,
    pattern_name: str,
    regex: str,
    min_hit_ratio: float = 0.70,
    severity: str = "warn",
    source: str = "builtin",
):
    """
    Generic validator for any regex pattern.
    """
    issues = []
    s = _safe_series_str(series)
    x = s.dropna().astype("string").str.strip()

    if x.empty:
        return issues

    if not _regex_is_safe(regex):
        return issues

    try:
        compiled = re.compile(regex)
    except Exception:
        issues.append(_make_issue(
            row=None,
            column=col,
            issue=f"{source} pattern '{pattern_name}' regex failed to compile",
            confidence=0.4,
            explanation=f"Regex '{regex}' is invalid.",
            details={"rule": "pattern_compile_failed", "pattern_name": pattern_name, "regex": regex, "source": source},
            suggested_fix={"action": "fix_regex"},
        ))
        return issues

    hits = x.str.match(compiled, na=False)
    hit_ratio = float(hits.mean())

    # If pattern doesn't really apply, do nothing
    if hit_ratio < float(min_hit_ratio):
        return issues

    conf = _pattern_confidence(hit_ratio)

    bad = ~hits
    for idx in x.index[bad]:
        raw = series.loc[idx]
        issues.append(_make_issue(
            row=idx,
            column=col,
            issue=f"{pattern_name} pattern: invalid format",
            confidence=conf,
            explanation=f"Value '{raw}' does not match expected pattern '{pattern_name}'.",
            details={
                "rule": "pattern_regex",
                "pattern_name": pattern_name,
                "regex": regex,
                "hit_ratio": float(hit_ratio),
                "min_hit_ratio": float(min_hit_ratio),
                "severity": str(severity),
                "source": str(source),
            },
            suggested_fix={"action": "ask_user", "reason": f"invalid {pattern_name}"},
        ))

    # override severity if required
    if severity in {"error", "warn"}:
        for it in issues:
            it["severity"] = severity

    return issues


# =========================================================
# ✅ NEW: LLM schema (runs once)
# =========================================================

def _llm_schema_for_dataset(
    df: pd.DataFrame,
    enable_llm: bool,
    llm_base_url: str,
    llm_model: str,
    llm_timeout: int,
    llm_max_cols: int,
):
    """
    Runs LLM ONCE per dataset.
    Returns safe schema dict or None.
    Never throws.
    """
    if not enable_llm:
        return None

    # IMPORTANT: if dataset is huge, only send a small slice
    df_small = df.head(250).copy()

    try:
        client = LocalLLMClient(
            base_url=llm_base_url,
            model=llm_model,
            timeout=int(llm_timeout),
            verbose=False,
        )

        schema = infer_schema_with_llm(df_small, client, max_cols=int(llm_max_cols))
        schema = apply_llm_schema_safety(schema, df_small)

        if not isinstance(schema, dict):
            return None

        if not isinstance(schema.get("columns", {}), dict):
            return None

        return schema

    except Exception as e:
        warnings.warn(f"[PatternValidator LLM] failed: {str(e)}", UserWarning)
        return None


def _llm_patterns_to_custom_patterns(llm_schema: dict, min_confidence: float = 0.70) -> dict:
    """
    Converts LLM schema into Stage4 custom_patterns format.
    Only uses:
      - suggested_rules.pattern_regex
    """
    out = {}

    if not isinstance(llm_schema, dict):
        return out

    cols = llm_schema.get("columns", {})
    if not isinstance(cols, dict):
        return out

    for col, info in cols.items():
        if not isinstance(info, dict):
            continue

        conf = info.get("confidence", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0

        if conf < float(min_confidence):
            continue

        rules = info.get("suggested_rules", {})
        if not isinstance(rules, dict):
            continue

        pattern = rules.get("pattern_regex")
        if not pattern:
            continue

        if not _regex_is_safe(pattern):
            continue

        out[str(col)] = {
            "pattern_name": "llm_pattern",
            "regex": str(pattern),
            "min_hit_ratio": 0.70,
            "severity": "warn",
            "source": "llm",
        }

    return out


# -------------------------------------------------
# Built-in pattern detectors (column-level)
# -------------------------------------------------

def _pattern_email(series: pd.Series, col: str):
    issues = []
    s = _safe_series_str(series)
    x = _norm_str(s.dropna())

    if x.empty:
        return issues

    at_count = x.str.count("@")
    m_bad_at = at_count.ne(1)

    for idx in x.index[m_bad_at]:
        raw = series.loc[idx]
        issues.append(_make_issue(
            row=idx,
            column=col,
            issue="email pattern: invalid @ count",
            confidence=0.9,
            explanation=f"Email should contain exactly one '@'. Found '{raw}'.",
            details={"rule": "email_at_count", "value": str(raw)},
            suggested_fix={"action": "ask_user", "reason": "invalid email"}
        ))

    domain = x.str.extract(r"@(.+)$")[0]
    m_no_domain = domain.isna()

    for idx in x.index[m_no_domain]:
        raw = series.loc[idx]
        issues.append(_make_issue(
            row=idx,
            column=col,
            issue="email pattern: missing domain",
            confidence=0.9,
            explanation=f"Email must contain a domain part after '@'. Found '{raw}'.",
            details={"rule": "email_domain", "value": str(raw)},
            suggested_fix={"action": "ask_user", "reason": "invalid email"}
        ))

    m_domain_no_dot = domain.notna() & (~domain.str.contains(r"\.", regex=True))
    for idx in x.index[m_domain_no_dot]:
        raw = series.loc[idx]
        issues.append(_make_issue(
            row=idx,
            column=col,
            issue="email pattern: domain missing dot",
            confidence=0.9,
            explanation=f"Email domain should contain a dot (.). Found '{raw}'.",
            details={"rule": "email_domain_dot", "value": str(raw)},
            suggested_fix={"action": "ask_user", "reason": "invalid email"}
        ))

    issues.extend(_pattern_custom_regex(
        series=series,
        col=col,
        pattern_name="email",
        regex=COMMON_PATTERNS["email"]["regex"],
        min_hit_ratio=COMMON_PATTERNS["email"]["min_hit_ratio"],
        severity=COMMON_PATTERNS["email"]["severity"],
        source="builtin",
    ))

    return issues


def _pattern_phone(series: pd.Series, col: str, country_series: pd.Series | None = None):
    issues = []
    s = _safe_series_str(series)
    x = s.dropna().astype("string").str.strip()

    if x.empty:
        return issues

    digits = x.str.replace(r"\D", "", regex=True)

    m_too_short = digits.str.len() < 7
    for idx in digits.index[m_too_short]:
        raw = series.loc[idx]
        issues.append(_make_issue(
            row=idx,
            column=col,
            issue="phone pattern: too few digits",
            confidence=0.85,
            explanation=f"Phone number seems too short. Found '{raw}'.",
            details={"rule": "phone_min_digits", "digits": str(digits.loc[idx])},
            suggested_fix={"action": "ask_user", "reason": "invalid phone"}
        ))

    len_stats = _value_counts_ratio(digits.str.len())
    if len_stats is not None:
        top_len, top_ratio, vc = len_stats
        if top_ratio >= 0.7:
            m_rare_len = digits.str.len().map(lambda L: vc.get(L, 0) / vc.sum()) < 0.02
            for idx in digits.index[m_rare_len]:
                raw = series.loc[idx]
                issues.append(_make_issue(
                    row=idx,
                    column=col,
                    issue="phone pattern: rare length",
                    confidence=0.75,
                    explanation=f"Most phone values have length {top_len} digits ({top_ratio:.0%}). Found '{raw}'.",
                    details={
                        "rule": "phone_common_length",
                        "top_length": int(top_len),
                        "top_ratio": float(top_ratio),
                        "actual_length": int(len(str(digits.loc[idx]))),
                    },
                    suggested_fix={"action": "ask_user", "reason": "inconsistent phone format"}
                ))

    m_10 = digits.str.len() == 10
    if m_10.mean() > 0.6:
        leading = digits[m_10].str[0]
        m_bad_lead = leading.notna() & (~leading.isin(list("6789")))
        for idx in leading.index[m_bad_lead]:
            raw = series.loc[idx]
            issues.append(_make_issue(
                row=idx,
                column=col,
                issue="phone pattern: unusual leading digit",
                confidence=0.7,
                explanation=f"Most Indian mobile numbers start with 6–9. Found '{raw}'.",
                details={"rule": "india_leading_digit", "digits": str(digits.loc[idx])},
                suggested_fix={"action": "ask_user", "reason": "phone may be invalid"}
            ))

    # ✅ FIXED: index alignment (prevents FutureWarning)
    if country_series is not None:
        c = _norm_str(country_series)

        m_india = c.eq("india") & series.notna()

        digits_len = digits.reindex(m_india.index).astype("string").str.len()
        m_bad_india = m_india & (digits_len != 10)

        for idx in df_index(series, m_bad_india):
            raw = series.loc[idx]
            issues.append(_make_issue(
                row=idx,
                column=col,
                issue="phone pattern: invalid for India",
                confidence=0.9,
                explanation=f"India phone numbers should have 10 digits. Found '{raw}'.",
                details={"rule": "india_phone_digits", "expected_digits": 10, "actual": str(raw)},
                suggested_fix={"action": "ask_user", "reason": "invalid phone"}
            ))

    issues.extend(_pattern_custom_regex(
        series=series,
        col=col,
        pattern_name="phone",
        regex=COMMON_PATTERNS["phone_generic"]["regex"],
        min_hit_ratio=COMMON_PATTERNS["phone_generic"]["min_hit_ratio"],
        severity=COMMON_PATTERNS["phone_generic"]["severity"],
        source="builtin",
    ))

    return issues


def _pattern_pin_zip(series: pd.Series, col: str, country_series: pd.Series | None = None):
    issues = []
    s = _safe_series_str(series)
    x = s.dropna().astype("string").str.strip()

    if x.empty:
        return issues

    digits = x.str.replace(r"\D", "", regex=True)
    lengths = digits.str.len()

    stats = _value_counts_ratio(lengths)
    if stats is not None:
        top_len, top_ratio, vc = stats
        if top_ratio >= 0.8:
            rare = lengths.map(lambda L: vc.get(L, 0) / vc.sum()) < 0.02
            for idx in lengths.index[rare]:
                raw = series.loc[idx]
                issues.append(_make_issue(
                    row=idx,
                    column=col,
                    issue="postal pattern: rare length",
                    confidence=0.75,
                    explanation=f"Most postal codes have length {top_len}. Found '{raw}'.",
                    details={
                        "rule": "postal_common_length",
                        "top_length": int(top_len),
                        "top_ratio": float(top_ratio),
                        "actual_length": int(lengths.loc[idx]),
                    },
                    suggested_fix={"action": "ask_user", "reason": "postal code format inconsistent"}
                ))

    # ✅ FIXED: index alignment (prevents FutureWarning)
    if country_series is not None:
        c = _norm_str(country_series)

        m_india = c.eq("india") & series.notna()

        digits_len = digits.reindex(m_india.index).astype("string").str.len()
        m_bad_india = m_india & (digits_len != 6)

        for idx in df_index(series, m_bad_india):
            raw = series.loc[idx]
            issues.append(_make_issue(
                row=idx,
                column=col,
                issue="postal pattern: invalid for India",
                confidence=0.9,
                explanation=f"India PIN codes should be 6 digits. Found '{raw}'.",
                details={"rule": "india_pin", "expected_digits": 6, "actual": str(raw)},
                suggested_fix={"action": "ask_user", "reason": "invalid pin"}
            ))

    name = str(col).strip().lower()

    if "pin" in name and (country_series is not None and _norm_str(country_series).eq("india").mean() > 0.2):
        issues.extend(_pattern_custom_regex(
            series=series,
            col=col,
            pattern_name="pin_india",
            regex=COMMON_PATTERNS["pin_india"]["regex"],
            min_hit_ratio=COMMON_PATTERNS["pin_india"]["min_hit_ratio"],
            severity=COMMON_PATTERNS["pin_india"]["severity"],
            source="builtin",
        ))
    else:
        issues.extend(_pattern_custom_regex(
            series=series,
            col=col,
            pattern_name="zip_us",
            regex=COMMON_PATTERNS["zip_us"]["regex"],
            min_hit_ratio=COMMON_PATTERNS["zip_us"]["min_hit_ratio"],
            severity=COMMON_PATTERNS["zip_us"]["severity"],
            source="builtin",
        ))

    return issues


def _pattern_ssn(series: pd.Series, col: str):
    return _pattern_custom_regex(
        series=series,
        col=col,
        pattern_name="ssn_us",
        regex=COMMON_PATTERNS["ssn_us"]["regex"],
        min_hit_ratio=COMMON_PATTERNS["ssn_us"]["min_hit_ratio"],
        severity=COMMON_PATTERNS["ssn_us"]["severity"],
        source="builtin",
    )


def _pattern_url(series: pd.Series, col: str):
    issues = []
    s = _safe_series_str(series)
    x = _norm_str(s.dropna())

    if x.empty:
        return issues

    m_no_scheme = ~x.str.match(r"^https?://")
    for idx in x.index[m_no_scheme]:
        raw = series.loc[idx]
        issues.append(_make_issue(
            row=idx,
            column=col,
            issue="url pattern: missing scheme",
            confidence=0.75,
            explanation=f"URL should start with http:// or https://. Found '{raw}'.",
            details={"rule": "url_scheme", "value": str(raw)},
            suggested_fix={"action": "prepend", "value": "https://"}
        ))

    return issues


def _pattern_username(series: pd.Series, col: str):
    issues = []
    s = _safe_series_str(series)
    x = s.dropna().astype("string").str.strip()

    if x.empty:
        return issues

    m_bad_chars = ~x.str.match(r"^[a-zA-Z0-9_]+$")
    for idx in x.index[m_bad_chars]:
        raw = series.loc[idx]
        issues.append(_make_issue(
            row=idx,
            column=col,
            issue="username pattern: invalid characters",
            confidence=0.85,
            explanation=f"Username should contain only letters, digits, and underscores. Found '{raw}'.",
            details={"rule": "username_chars", "value": str(raw)},
            suggested_fix={"action": "ask_user", "reason": "invalid username"}
        ))

    return issues


def _pattern_name(series: pd.Series, col: str):
    issues = []
    s = _safe_series_str(series)
    x = s.dropna().astype("string").str.strip()

    if x.empty:
        return issues

    m_bad = ~x.str.match(r"^[A-Za-z\s\.\-']+$")
    for idx in x.index[m_bad]:
        raw = series.loc[idx]
        issues.append(_make_issue(
            row=idx,
            column=col,
            issue="name pattern: contains unusual characters",
            confidence=0.7,
            explanation=f"Name values usually contain only letters/spaces. Found '{raw}'.",
            details={"rule": "name_chars", "value": str(raw)},
            suggested_fix={"action": "ask_user", "reason": "name may be corrupted"}
        ))

    return issues


def _pattern_category_typos(series: pd.Series, col: str):
    issues = []
    s = _safe_series_str(series)
    x = _norm_str(s.dropna())

    if x.empty:
        return issues

    unique = x.unique()
    if len(unique) > 25:
        return issues

    vc = x.value_counts(normalize=True)
    rare = vc[vc < 0.01].index.tolist()

    for rv in rare:
        idxs = x.index[x == rv]
        for idx in idxs:
            raw = series.loc[idx]
            issues.append(_make_issue(
                row=idx,
                column=col,
                issue="category pattern: rare value (possible typo)",
                confidence=0.7,
                explanation=f"Value '{raw}' is rare in this categorical column.",
                details={"rule": "category_rare", "rare_value": rv},
                suggested_fix={"action": "ask_user", "reason": "possible category typo"}
            ))

    return issues


# -------------------------------------------------
# Stage 4 (Pattern Generalization)
# -------------------------------------------------
def pattern_validate_column(
    series: pd.Series,
    col: str,
    df: pd.DataFrame | None = None,
    custom_patterns: dict | None = None,
):
    """
    Stage-4 validates TEXT columns only.

    Supports:
    - custom_patterns (from pipeline or LLM)
    """
    name = str(col).lower()

    if _is_id_like(col):
        return []

    if _is_numeric(series) or _is_datetime(series) or _is_bool(series):
        return []

    country_series = None
    if df is not None and "country" in df.columns:
        country_series = df["country"]

    # custom patterns
    if isinstance(custom_patterns, dict) and custom_patterns:
        for key, spec in custom_patterns.items():
            try:
                key_s = str(key)
                if key_s == str(col) or re.search(key_s, str(col), flags=re.IGNORECASE):
                    if isinstance(spec, dict):
                        issues = _pattern_custom_regex(
                            series=series,
                            col=col,
                            pattern_name=str(spec.get("pattern_name", "custom")),
                            regex=str(spec.get("regex", "")),
                            min_hit_ratio=float(spec.get("min_hit_ratio", 0.70)),
                            severity=str(spec.get("severity", "warn")),
                            source=str(spec.get("source", "custom")),
                        )
                        if issues:
                            return issues
            except Exception:
                continue

    # built-ins
    if "email" in name:
        return _pattern_email(series, col)

    if any(k in name for k in ["phone", "mobile", "contact"]):
        return _pattern_phone(series, col, country_series=country_series)

    if any(k in name for k in ["pin", "zip", "postal", "postcode"]):
        return _pattern_pin_zip(series, col, country_series=country_series)

    if "ssn" in name or "social_security" in name:
        return _pattern_ssn(series, col)

    if any(k in name for k in ["url", "website", "link"]):
        return _pattern_url(series, col)

    if any(k in name for k in ["username", "user_name", "handle"]):
        return _pattern_username(series, col)

    if any(k in name for k in ["name", "full_name"]) and "username" not in name:
        return _pattern_name(series, col)

    return _pattern_category_typos(series, col)


def validate_patterns(df: pd.DataFrame, custom_patterns: dict | None = None):
    all_issues = []
    for col in df.columns:
        all_issues.extend(pattern_validate_column(df[col], col, df=df, custom_patterns=custom_patterns))
    return all_issues


# =========================================================
# Transformer
# =========================================================
class PatternValidator(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        apply_fixes: bool = False,
        custom_patterns: dict | None = None,

        # ✅ LLM
        enable_llm: bool = True,
        llm_base_url: str = "http://127.0.0.1:1234",
        llm_model: str = "deepseek-coder-7b-instruct-v1.5",
        llm_timeout: int = 90,  # ✅ FIX
        llm_max_cols: int = 35,
        llm_min_confidence: float = 0.70,
    ):
        self.apply_fixes = bool(apply_fixes)
        self.custom_patterns = custom_patterns

        self.enable_llm = bool(enable_llm)
        self.llm_base_url = str(llm_base_url)
        self.llm_model = str(llm_model)
        self.llm_timeout = int(llm_timeout)
        self.llm_max_cols = int(llm_max_cols)
        self.llm_min_confidence = float(llm_min_confidence)

        self.issues = []
        self.last_report_ = {}
        self.llm_schema_ = None

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        original_df = X.copy()
        df = X.copy()

        # 1) Run LLM once (optional)
        self.llm_schema_ = _llm_schema_for_dataset(
            df=df,
            enable_llm=self.enable_llm,
            llm_base_url=self.llm_base_url,
            llm_model=self.llm_model,
            llm_timeout=self.llm_timeout,
            llm_max_cols=self.llm_max_cols,
        )

        llm_custom_patterns = {}
        if isinstance(self.llm_schema_, dict):
            llm_custom_patterns = _llm_patterns_to_custom_patterns(
                self.llm_schema_,
                min_confidence=self.llm_min_confidence,
            )

        merged_patterns = {}
        if isinstance(llm_custom_patterns, dict):
            merged_patterns.update(llm_custom_patterns)
        if isinstance(self.custom_patterns, dict):
            merged_patterns.update(self.custom_patterns)

        # 2) Validate
        self.issues = validate_patterns(df, custom_patterns=merged_patterns)

        # 3) SAFE FIX APPLY
        if self.apply_fixes:
            for it in self.issues:
                row = it.get("row")
                col = it.get("column")
                fix = (it.get("suggested_fix") or {})
                action = fix.get("action")

                if row is None:
                    continue
                if col not in df.columns or row not in df.index:
                    continue

                if _is_numeric(df[col]) or _is_datetime(df[col]) or _is_bool(df[col]) or _is_id_like(col):
                    continue

                if action == "lowercase":
                    v = df.loc[row, col]
                    if pd.notna(v):
                        df.loc[row, col] = str(v).lower().strip()

                elif action == "replace_prefix":
                    v = df.loc[row, col]
                    if pd.notna(v):
                        s = str(v)
                        frm = fix.get("from", "")
                        to = fix.get("to", "")
                        if frm and s.startswith(frm):
                            df.loc[row, col] = to + s[len(frm):]

                elif action == "prepend":
                    v = df.loc[row, col]
                    if pd.notna(v):
                        s = str(v).strip()
                        prefix = str(fix.get("value", ""))
                        if prefix and not s.startswith(prefix):
                            df.loc[row, col] = prefix + s

        # 4) Report
        by_rule = {}
        for it in self.issues:
            det = it.get("details") or {}
            rule = det.get("rule", "unknown")
            by_rule[rule] = by_rule.get(rule, 0) + 1

        llm_status = "disabled"
        if self.enable_llm:
            llm_status = "ok" if isinstance(self.llm_schema_, dict) else "failed"

        self.last_report_ = {
            "pattern_issue_count": int(len(self.issues)),
            "apply_fixes": bool(self.apply_fixes),
            "custom_patterns_enabled": bool(isinstance(self.custom_patterns, dict) and self.custom_patterns),
            "issues_by_rule": by_rule,

            # LLM
            "llm_enabled": bool(self.enable_llm),
            "llm_model": self.llm_model,
            "llm_status": llm_status,
            "llm_min_confidence": float(self.llm_min_confidence),
            "llm_patterns_applied": int(len(llm_custom_patterns)),
        }

        _check_df_integrity(original_df, df, "PatternValidator")
        return df
