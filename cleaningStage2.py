# =========================================================
# cleaningStage2.py
# Stage 2: Rule-based cleaning (SAFE, non-destructive)
# UNIVERSAL PERMANENT ENTERPRISE VERSION (TYPED + NO CRASH + NULL GUARD + LAYER 3 SEMANTICS)
# =========================================================

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Sequence, Any

import numpy as np
import pandas as pd
import warnings

from difflib import get_close_matches
from sklearn.base import BaseEstimator, TransformerMixin

from pipeline_utils import _check_df_integrity

# Graceful import to maintain NO CRASH guarantee
try:
    from backend.learning_memory import CleaningLearningMemory
except ImportError:
    # Standalone fallback if memory module is absent (with smoothing logic included)
    class CleaningLearningMemory:
        def __init__(self):
            self.memory = {}
            
        def get_penalty(self, col: str, role: str) -> float:
            stats = self.memory.get((col, role), {"seen": 0, "failures": 0})
            seen = stats["seen"]
            failures = stats["failures"]
            
            # ⭐ smoothing: prevent over-punishment early on
            if seen < 5:
                return 0.0
                
            failure_rate = failures / max(seen, 1)
            return min(0.25, failure_rate * 0.4)
            
        def record_outcome(self, col: str, role: str, confidence: float, success: bool) -> None:
            if (col, role) not in self.memory:
                self.memory[(col, role)] = {"seen": 0, "failures": 0}
            self.memory[(col, role)]["seen"] += 1
            if not success:
                self.memory[(col, role)]["failures"] += 1

# =========================================================
# CONSTANTS
# =========================================================

MONEY_SIGNS = ["₹", "$", "€", "£", "¥"]
MONEY_CODES = ["INR", "USD", "EUR", "GBP", "AED", "JPY", "CAD", "AUD", "RS", "RS.", "RUPEES"]

NULL_TOKENS = {
    "nan", "none", "null", "na", "n/a", "nil", "missing", "-", "--", "?", "",
    "unknown", "not_available", "not available"
}

NULL_INFLATION_THRESHOLD = 0.25 
NULL_INCREASE_WARNING_THRESHOLD = 0.10 

# =========================================================
# BASIC HELPERS + NULL GUARD
# =========================================================

def _safe_colname(x: Any) -> str:
    try: return str(x)
    except Exception: return ""

def _norm_colname(x: Any) -> str:
    return _safe_colname(x).strip().lower().replace(" ", "_")

def _is_null_token(s: Any) -> bool:
    if s is None: return True
    x = str(s).strip().lower()
    return (x == "") or (x in NULL_TOKENS)

def conversion_success_rate_non_null(parsed: pd.Series, original: pd.Series) -> float:
    m = original.notna()
    if int(m.sum()) == 0: return 0.0
    return float(parsed[m].notna().mean())

def _conversion_preserves_enough_data(parsed: pd.Series, original: pd.Series, keep_ratio: float = 0.90) -> bool:
    original_valid = int(original.notna().sum())
    if original_valid == 0: return True
    parsed_valid = int(pd.Series(parsed).notna().sum())
    return parsed_valid >= int(original_valid * keep_ratio)

def _check_null_inflation(before_nulls: int, after_nulls: int, col_name: str, threshold: float = NULL_INFLATION_THRESHOLD) -> Tuple[bool, str]:
    before_safe = max(before_nulls, 1)
    increase_ratio = (after_nulls - before_nulls) / before_safe
    
    if increase_ratio > threshold:
        return False, f"NULL_INFLATION_CRITICAL: {col_name} nulls {before_nulls}→{after_nulls} (+{increase_ratio:.1%}) > {threshold*100}% THRESHOLD - REVERTED"
    if increase_ratio > NULL_INCREASE_WARNING_THRESHOLD:
        return True, f"NULL_INFLATION_WARNING: {col_name} nulls {before_nulls}→{after_nulls} (+{increase_ratio:.1%})"
    
    return True, f"OK: {col_name} nulls {before_nulls}→{after_nulls} ({increase_ratio:+.1%})"

# =========================================================
# UNIVERSAL PARSERS
# =========================================================

def parse_money_universal(x: Any) -> float:
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if _is_null_token(s): return np.nan
    s = re.sub(r"[\u00A0\u2000-\u200B]+", " ", s).strip()
    if s.startswith("(") and s.endswith(")"): s = "-" + s[1:-1].strip()
    s = re.sub(r"/-\s*$", "", s)
    s = re.sub(r"\bonly\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bapprox\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bapproximately\b", "", s, flags=re.IGNORECASE)
    for sym in MONEY_SIGNS: s = s.replace(sym, "")
    for code in MONEY_CODES: s = re.sub(rf"\b{re.escape(code)}\b", "", s, flags=re.IGNORECASE)
    s = s.strip().replace(" ", "")
    if re.search(r"[a-zA-Z]", s): return np.nan
    if re.search(r"[^0-9,.\-]", s): return np.nan
    if "," in s and "." in s: s = s.replace(",", "")
    if s.count(",") >= 2 and "." not in s:
        parts = s.split(",")
        s = "".join(parts[:-1]) + "." + parts[-1] if len(parts[-1]) == 2 else "".join(parts)
    if s.count(",") == 1 and "." not in s:
        left, right = s.split(",")
        s = left + "." + right if len(right) == 2 else left + right
    s = s.replace(",", "")
    if s in {"", "-", "--"}: return np.nan
    try: return float(s)
    except Exception: return np.nan

def parse_percent_universal(x: Any) -> float:
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if _is_null_token(s): return np.nan
    if s.endswith("%"):
        try: return float(s[:-1].strip())
        except Exception: return np.nan
    try:
        v = float(s)
        return v * 100.0 if 0 <= v <= 1 else v
    except Exception: return np.nan

def parse_bool_universal(x: Any) -> bool | float:
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if _is_null_token(s): return np.nan
    if s in ["true", "t", "yes", "y", "1", "returned"]: return True
    if s in ["false", "f", "no", "n", "0", "not returned"]: return False
    return np.nan

def parse_date_universal(x: Any) -> pd.Timestamp | str:
    if pd.isna(x): return pd.NaT
    s = str(x).strip()
    if _is_null_token(s): return pd.NaT
    s = re.sub(r"[\u00A0\u2000-\u200B]+", " ", s).strip()
    s = re.sub(r"\b(?:IST|UTC|GMT|CST|EST|PST|AEST|AEDT|JST|MSK|NZST|NZDT)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[+-]\d{2}:\d{2}(?=\s|$)", "", s)
    match = re.search(r"(\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:[AP]M)?)?)|(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:[AP]M)?)|(\d{4}-\d{2}-\d{2})", s)
    if match: s = match.group(0).strip()
    try:
        if re.fullmatch(r"^-?\d+(\.\d+)?$", s):
            num = float(s)
            if 1_000_000_000 <= num <= 2_000_000_000: return pd.to_datetime(int(num), unit="s", errors="coerce")
            if 1_000_000_000_000 <= num <= 2_000_000_000_000: return pd.to_datetime(int(num), unit="ms", errors="coerce")
            if 20000 <= num <= 60000: return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    except Exception: pass
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"]:
        try:
            dt = pd.to_datetime(s, format=fmt, errors="coerce")
            if pd.notna(dt): return dt
        except: continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            dt = pd.to_datetime(s, errors="coerce")
            if pd.notna(dt): return dt
        except: pass
    return pd.NaT

# =========================================================
# ROLE DETECTION
# =========================================================

def detect_role(col_name: str, series: pd.Series) -> str:
    name = _norm_colname(col_name)
    if re.search(r"(^|_)(date|dob|created_at|createdon|timestamp|datetime)(_|$)", name): return "date"
    if re.search(r"(^|_)(amount|price|cost|spent|salary|income|total|revenue)(_|$)", name): return "money"
    if re.search(r"(^|_)(discount|percent|pct|rate|tax|commission)(_|$)", name): return "percent"
    if re.search(r"(^|_)(is|has|flag|active|enabled|valid)(_|$)", name): return "bool"

    sample = series.dropna().astype(str).head(80)
    if sample.empty: return "unknown"
    joined = " ".join(sample.tolist()).lower()

    if any(sym in joined for sym in MONEY_SIGNS) or any(code.lower() in joined for code in MONEY_CODES): return "money"
    if "%" in joined: return "percent"
    
    words = set(joined.split())
    bool_tokens = {"true", "false", "yes", "no", "y", "n"}
    bool_hits = len(words.intersection(bool_tokens))
    
    if bool_hits >= 2 and len(sample) >= 10:
        return "bool"

    if re.search(r"\d{4}-\d{2}-\d{2}", joined) or re.search(r"\d{1,2}/\d{1,2}/\d{4}", joined): return "date"
    return "unknown"

# =========================================================
# STAGE 2 CLEANER
# =========================================================

class QualityRuleCleaner(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        domain_rules: Optional[Dict[str, Tuple[float, float]]] = None,
        cross_rules: Optional[List[Any]] = None,
        category_rules: Optional[Dict[str, Any]] = None,
        business_rules: Optional[List[Any]] = None,
        temporal_rules: Optional[List[Any]] = None,
        constant_policy: str = "warn",
        max_missing_rate: float = 0.6,
        imbalance_threshold: float = 0.95,
        strict: bool = False,
        failure_policies: Optional[Dict[str, str]] = None,
        column_roles: Optional[Dict[str, str]] = None,
        drift_thresholds: Optional[Dict[str, float]] = None,
        id_columns: Optional[List[str]] = None,
        impute_missing: bool = False,
        universal_min_success: float = 0.60,
        allow_partial_universal_conversion: bool = True,
        required_id_columns: Tuple[str, ...] = ("order_id",),
        soft_rule_threshold: float = 0.75,
        null_inflation_threshold: float = NULL_INFLATION_THRESHOLD,
        cleaning_profile: str = "balanced",
        safe_mode_no_risky_coercions: bool = False,
        safe_mode_strict_null_guard: bool = False,
        semantic_types: Optional[Dict[str, str]] = None,  
        semantic_confidences: Optional[Dict[str, float]] = None,
        role_confidence_threshold: float = 0.60,
        high_confidence_threshold: float = 0.85,
        enable_confidence_gating: bool = True,
    ) -> None:

        self.domain_rules = domain_rules or {"age": (0, 100), "income": (0, 1_000_000), "total_spent": (0, 10_000_000)}
        self.cross_rules = cross_rules or []
        self.category_rules = category_rules or {}
        self.business_rules = business_rules or []
        self.temporal_rules = temporal_rules or []
        self.constant_policy = constant_policy
        self.max_missing_rate = float(max_missing_rate)
        self.imbalance_threshold = float(imbalance_threshold)
        self.strict = bool(strict)

        self.failure_policies = failure_policies or {"domain": "warn", "cross": "warn", "category": "warn", "business": "warn", "temporal": "warn", "drift_high": "error"}
        self.column_roles = column_roles or {}
        self.drift_thresholds = drift_thresholds or {"low": 0.05, "medium": 0.15}
        self.id_columns = set([_norm_colname(c) for c in (id_columns or [])])
        self.required_id_columns = tuple([_norm_colname(c) for c in required_id_columns])
        self.impute_missing = bool(impute_missing)
        self.universal_min_success = float(universal_min_success)
        self.allow_partial_universal_conversion = bool(allow_partial_universal_conversion)
        self.soft_rule_threshold = float(soft_rule_threshold)
        self.null_inflation_threshold = float(null_inflation_threshold)
        
        self.cleaning_profile = str(cleaning_profile).strip().lower()
        if self.cleaning_profile not in ["safe", "balanced", "aggressive"]: self.cleaning_profile = "balanced"
        self.safe_mode_no_risky_coercions = bool(safe_mode_no_risky_coercions)
        self.safe_mode_strict_null_guard = bool(safe_mode_strict_null_guard)

        # Confidence gating bounds
        self.role_confidence_threshold = float(role_confidence_threshold)
        self.high_confidence_threshold = float(high_confidence_threshold)
        self.enable_confidence_gating = bool(enable_confidence_gating)

        if self.cleaning_profile == "safe":
            self.safe_mode_no_risky_coercions = True
            self.safe_mode_strict_null_guard = True
            self.null_inflation_threshold = min(self.null_inflation_threshold, 0.10)
        elif self.cleaning_profile == "aggressive":
            self.safe_mode_no_risky_coercions = False
            self.safe_mode_strict_null_guard = False
            self.null_inflation_threshold = max(self.null_inflation_threshold, 0.50)

        self.category_vocab_: Dict[str, set] = {}
        self._domain_rules_lower_: Dict[str, Tuple[float, float]] = {}
        
        self.semantic_types = semantic_types or {}  
        self.semantic_confidences = semantic_confidences or {}
        self.role_memory_ = {} 
        self.role_confidences_ = {}
        
        # Adaptive Learning Memory Injection
        self.learning_memory = CleaningLearningMemory()

    def _resolve_semantic_role(self, col: str, series: pd.Series) -> Tuple[str, float]:
        if hasattr(self, "role_memory_") and col in self.role_memory_:
            return self.role_memory_[col], self.role_confidences_.get(col, 1.0)

        if isinstance(getattr(self, "semantic_types", None), dict):
            semantic_hint = self.semantic_types.get(str(col)) or self.semantic_types.get(_norm_colname(col))
            if semantic_hint and semantic_hint != "unknown":
                confidences = getattr(self, "semantic_confidences", {}) or {}
                conf = confidences.get(str(col)) or confidences.get(_norm_colname(col), 1.0)
                
                if conf >= 0.60:
                    resolved_role = "money" if semantic_hint in ["amount", "money"] else semantic_hint
                    self.role_memory_[col] = resolved_role
                    self.role_confidences_[col] = float(conf)
                    return resolved_role, float(conf)

        resolved_role = detect_role(col, series)
        heuristic_conf = 0.55 if resolved_role != "unknown" else 0.0
        
        dist_conf = 0.0
        if resolved_role == "unknown" and len(series) > 0:
            numeric = pd.to_numeric(series, errors="coerce")
            valid_ratio = float(numeric.notna().mean())
            
            if valid_ratio > 0.80:
                valid_numeric = numeric.dropna()
                
                if not valid_numeric.empty:
                    positive_ratio = float((valid_numeric >= 0).mean())
                    median_abs = float(valid_numeric.abs().median())
                    between_ratio = float(valid_numeric.between(0, 100).mean())
                    
                    if between_ratio > 0.95:
                        resolved_role = "percent"
                        dist_conf = min(0.95, 0.70 + (between_ratio - 0.95))
                    elif positive_ratio > 0.90 and median_abs > 100:
                        resolved_role = "money"
                        dist_conf = min(0.92, 0.65 + (positive_ratio - 0.90))

        final_conf = max(heuristic_conf if resolved_role != "unknown" else 0.0, dist_conf)
        
        if resolved_role == "unknown":
            final_conf = 0.30
            
        self.role_memory_[col] = resolved_role
        self.role_confidences_[col] = float(final_conf)
        
        return resolved_role, float(final_conf)

    def _compute_role_confidence(self, col: str, series: pd.Series, resolved_role: str) -> float:
        score = 0.0

        # --- signal 1: semantic hint ---
        sem_conf = (getattr(self, "semantic_confidences", {}) or {}).get(col)
        if sem_conf is not None:
            score += 0.45 * float(sem_conf)
        else:
            score += 0.20 # baseline trust for intrinsic heuristics if explicit absent

        # --- signal 2 & 3: type dominance & distribution ---
        if resolved_role in ("money", "percent"):
            numeric = pd.to_numeric(series, errors="coerce")
            numeric_ratio = float(numeric.notna().mean()) if len(series) > 0 else 0.0
            score += 0.25 * numeric_ratio

            valid = numeric.dropna()
            if not valid.empty:
                positive_ratio = float((valid >= 0).mean())
                score += 0.20 * positive_ratio
                
        elif resolved_role == "date":
            str_series = series.astype(str)
            date_like = str_series.str.contains(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,4}', regex=True).mean()
            score += 0.45 * float(date_like)
            
        elif resolved_role == "bool":
            str_series = series.astype(str).str.lower()
            bool_like = str_series.isin({"true", "false", "yes", "no", "y", "n", "1", "0"}).mean()
            score += 0.45 * float(bool_like)

        # --- signal 4: name heuristic strength ---
        name = _norm_colname(col)
        if any(k in name for k in ["amount", "price", "rate", "percent", "income", "cost", "fee", "tax", "discount", "date", "is_", "has_", "flag"]):
            score += 0.10

        # -----------------------------
        # learning penalty adjustment
        # -----------------------------
        penalty = self.learning_memory.get_penalty(col, resolved_role)
        score = max(0.0, score - penalty)

        return float(min(score, 1.0))

    def _violate(self, rule: str, message: str, severity: str = "warn") -> None:
        self.violations_.append({"rule": rule, "message": message, "severity": severity})
        if self.strict and severity == "error": raise ValueError(message)

    def _is_id_col(self, col: str) -> bool:
        c = _norm_colname(col)
        if c in self.id_columns: return True
        if c == "id" or c.endswith("_id"): return True
        if "uuid" in c: return True
        if any(k in c for k in ["aadhar", "aadhaar", "pan", "gst", "ifsc"]): return True
        return False

    def _normalize_text_safe(self, series: pd.Series) -> pd.Series:
        s = series.astype("string").str.strip()
        s = s.replace(r"^\s*$", pd.NA, regex=True)
        lower = s.str.lower()
        s = s.mask(lower.isin(list(NULL_TOKENS)), pd.NA)
        return s

    def _is_category_like(self, series: pd.Series) -> bool:
        x = series.dropna().astype(str).str.strip()
        if x.empty: return False
        n = len(x)
        nunique = x.nunique()
        if nunique > 60 or (nunique / max(n, 1)) > 0.6 or float(x.str.len().mean()) > 25: return False
        return True

    def fit(self, X: pd.DataFrame, y: Optional[Sequence[Any]] = None) -> "QualityRuleCleaner":
        df = X.copy()
        self.schema_ = list(df.columns)
        self.train_profile_, self.constant_cols_, self.high_missing_cols_, self.violations_, self.category_vocab_ = {}, [], [], [], {}
        if not hasattr(self, "role_memory_"): self.role_memory_ = {}
        if not hasattr(self, "role_confidences_"): self.role_confidences_ = {}
        self._domain_rules_lower_ = {_norm_colname(k): v for k, v in (self.domain_rules or {}).items()}

        df_cols_norm = {_norm_colname(c) for c in df.columns}
        for rid in self.required_id_columns:
            if rid not in df_cols_norm: self._violate("missing_id", f"Required ID column '{rid}' is missing", severity="error" if self.strict else "warn")

        for col in df.columns:
            s = df[col]
            self.train_profile_[col] = {
                "missing_rate": float(s.isnull().mean()),
                "unique_rate": float(s.nunique(dropna=True) / max(len(s), 1)),
                "top_freq": (float(s.value_counts(normalize=True, dropna=True).iloc[0]) if not s.value_counts(normalize=True, dropna=True).empty else 0.0),
            }
            if float(s.isnull().mean()) > self.max_missing_rate: self.high_missing_cols_.append(col)
            if int(s.nunique(dropna=True)) <= 1: self.constant_cols_.append(col)

            if self._is_id_col(col): continue

            if (s.dtype == "object") or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
                normalized = self._normalize_text_safe(s)
                if self._is_category_like(normalized):
                    self.category_vocab_[col] = set(normalized.dropna().astype(str).str.strip().str.lower().unique())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        original_input_df = X.copy()
        df = X.copy()
        original_order = list(df.columns)
        lineage: Dict[str, List[str]] = {c: [] for c in df.columns}
        self.violations_ = []
        baseline_nulls: Dict[str, int] = {col: int(df[col].isna().sum()) for col in df.columns}

        if hasattr(self, "schema_"):
            missing_train_cols = [c for c in self.schema_ if c not in df.columns]
            if missing_train_cols: self._violate("schema_missing", f"Missing columns vs training: {missing_train_cols}")

        drift_report: Dict[str, Dict[str, float]] = {}
        low_thr, med_thr = float(self.drift_thresholds.get("low", 0.05)), float(self.drift_thresholds.get("medium", 0.15))

        for col in df.columns:
            if col not in getattr(self, "train_profile_", {}): continue
            s = df[col]
            cur_missing, cur_unique = float(s.isnull().mean()), float(s.nunique(dropna=True) / max(len(s), 1))
            train_missing, train_unique = float(self.train_profile_[col]["missing_rate"]), float(self.train_profile_[col]["unique_rate"])
            d_missing, d_unique = abs(cur_missing - train_missing), abs(cur_unique - train_unique)
            level = "high" if (d_missing >= med_thr or d_unique >= med_thr) else "medium" if (d_missing >= low_thr or d_unique >= low_thr) else "none"

            drift_report[col] = {"train_missing": train_missing, "cur_missing": cur_missing, "delta_missing": d_missing, "train_unique": train_unique, "cur_unique": cur_unique, "delta_unique": d_unique, "level": level}
            if level == "high": self._violate("drift_high", f"{col} drift HIGH", severity=self.failure_policies.get("drift_high", "error"))
            elif level == "medium": self._violate("drift_medium", f"{col} drift MEDIUM", severity="warn")

        cols_norm_map = {_norm_colname(c): c for c in df.columns}
        for rid in self.required_id_columns:
            if rid in cols_norm_map:
                real_col = cols_norm_map[rid]
                if df[real_col].isnull().any(): self._violate("id_nulls", f"{real_col} contains nulls")
                if df[real_col].duplicated().any(): self._violate("id_duplicates", f"{real_col} contains duplicates")

        protected_text_cols = set()
        for col in df.columns:
            if self._is_id_col(col):
                protected_text_cols.add(col)
                continue
            
            is_text = False
            s_drop = df[col].dropna()
            
            if not s_drop.empty:
                numeric_like_ratio = pd.to_numeric(s_drop, errors="coerce").notna().mean()
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    if numeric_like_ratio < 0.7: is_text = True
                elif df[col].dtype == "object":
                    str_ratio = s_drop.apply(lambda x: isinstance(x, str)).mean()
                    if str_ratio > 0.8 and numeric_like_ratio < 0.7: is_text = True
            
            if is_text:
                protected_text_cols.add(col)
                lineage[col].append("stage2:protected_text_categorical")

        for col in df.columns:
            if self._is_id_col(col): continue
            if col in protected_text_cols:
                try:
                    s = df[col].astype("string").str.strip()
                    df[col] = s.replace(r"^\s*$", pd.NA, regex=True)
                    lineage[col].append("stage2:text_trim_only_protected")
                except Exception: pass
                continue
            if (df[col].dtype == "object") or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                df[col] = self._normalize_text_safe(df[col])
                lineage[col].append("stage2:text_strip_safe")

        for col, vocab in (self.category_vocab_ or {}).items():
            if col not in df.columns or self._is_id_col(col) or pd.api.types.is_numeric_dtype(df[col]): continue
            if self.cleaning_profile == "safe":
                self._violate("safe_mode_skip", f"Skipped category repair for {col} (SAFE MODE)", severity="info")
                continue
            norm = self._normalize_text_safe(df[col]).str.lower()
            if not self._is_category_like(norm): continue

            mask = ~norm.isin(vocab) & norm.notna()
            unseen = norm.loc[mask].unique()
            if len(unseen) > 0:
                for val in unseen:
                    match = get_close_matches(str(val), list(vocab), n=1)
                    if match: df.loc[norm == val, col] = match[0]
                    else: df.loc[norm == val, col] = pd.NA
                lineage[col].append("stage2:category_repair")
                self._violate("category", f"{col} unseen categories repaired")

        for col in df.columns:
            if self._is_id_col(col): continue
            col_lower = _norm_colname(col)
            if col_lower not in (self._domain_rules_lower_ or {}): continue
            if col in protected_text_cols:
                self._violate("protected_col", f"Skipped domain rules for {col} (Protected TEXT)", severity="info")
                continue

            low, high = self._domain_rules_lower_[col_lower]
            numeric = pd.to_numeric(df[col], errors="coerce")
            bad_mask = numeric.notna() & ((numeric < low) | (numeric > high))
            if bad_mask.any():
                df.loc[bad_mask, col] = np.nan
                lineage[col].append("stage2:domain_to_nan")
                self._violate("domain", f"{col} violated domain [{low}, {high}]")

            if not (self.cleaning_profile == "safe" and self.safe_mode_no_risky_coercions):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                lineage[col].append("stage2:numeric_coerce_safe")

        role_report, confidence_report, conversion_report = {}, {}, {}

        for col in df.columns:
            if self._is_id_col(col): continue
            if col in protected_text_cols and self.cleaning_profile == "safe":
                self._violate("protected_col_bypass", f"Skipped universal parsing for {col} (SAFE mode)", severity="info")
                continue

            role, _ = self._resolve_semantic_role(col, df[col])
            confidence = self._compute_role_confidence(col, df[col], role)

            role_report[col] = role
            confidence_report[col] = confidence
            original_col_backup = df[col].copy()

            if role == "money":
                if self.enable_confidence_gating and confidence < self.role_confidence_threshold:
                    self._violate("low_confidence_skip", f"{col} role '{role}' confidence {confidence:.2f} below threshold", severity="info")
                    self.learning_memory.record_outcome(col=col, role=role, confidence=confidence, success=False)
                    continue

                # Adaptive profile
                effective_profile = self.cleaning_profile
                if confidence < 0.50: effective_profile = "safe"
                elif confidence < 0.70: effective_profile = "balanced"

                is_safe_mode = (effective_profile == "safe" and self.safe_mode_no_risky_coercions)
                is_high_conf = confidence >= self.high_confidence_threshold
                local_min_success = self.universal_min_success if is_high_conf else max(0.75, self.universal_min_success)
                keep_ratio = 0.85 if is_high_conf else 0.95

                if is_safe_mode: pass
                else:
                    before_nulls = int(original_col_backup.isna().sum())
                    parsed = df[col].apply(parse_money_universal)
                    rate = conversion_success_rate_non_null(parsed, original_col_backup)
                    conversion_report[col] = {"rate": rate}
                    preserves = _conversion_preserves_enough_data(parsed, original_col_backup, keep_ratio=keep_ratio)
                    after_nulls = int(parsed.isna().sum())
                    allow_change, null_msg = _check_null_inflation(before_nulls, after_nulls, col, self.null_inflation_threshold)

                    success_flag = False

                    if rate >= local_min_success and preserves and allow_change:
                        df[col] = parsed
                        lineage[col].append(f"stage2:money_parse_rate_{rate:.3f}")
                        self._violate("null_check", null_msg, severity="info")
                        success_flag = True
                    elif (self.allow_partial_universal_conversion and int(parsed.notna().sum()) > 0 and preserves and allow_change):
                        df[col] = parsed
                        lineage[col].append(f"stage2:money_parse_partial_rate_{rate:.3f}")
                        self._violate("money_partial", f"{col} partially parsed money (rate={rate:.3f})")
                        success_flag = True
                    else:
                        df[col] = original_col_backup
                        if not allow_change: self._violate("null_inflation_revert", null_msg, severity="error")

                    if success_flag and not pd.api.types.is_numeric_dtype(df[col]) and not is_safe_mode:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        lineage[col].append("stage2:numeric_coerce_safe")

                    if pd.api.types.is_numeric_dtype(df[col]):
                        neg_mask = df[col].notna() & (df[col] < 0)
                        if neg_mask.any():
                            df.loc[neg_mask, col] = np.nan
                            lineage[col].append("stage2:money_negative_to_nan")

                    self.learning_memory.record_outcome(col=col, role=role, confidence=confidence, success=success_flag)

            elif role == "percent":
                if self.enable_confidence_gating and confidence < self.role_confidence_threshold:
                    self._violate("low_confidence_skip", f"{col} role '{role}' confidence {confidence:.2f} below threshold", severity="info")
                    self.learning_memory.record_outcome(col=col, role=role, confidence=confidence, success=False)
                    continue

                effective_profile = self.cleaning_profile
                if confidence < 0.50: effective_profile = "safe"
                elif confidence < 0.70: effective_profile = "balanced"

                is_safe_mode = (effective_profile == "safe" and self.safe_mode_no_risky_coercions)
                is_high_conf = confidence >= self.high_confidence_threshold
                local_min_success = self.universal_min_success if is_high_conf else max(0.75, self.universal_min_success)
                keep_ratio = 0.85 if is_high_conf else 0.95

                if is_safe_mode: pass
                else:
                    before_nulls = int(original_col_backup.isna().sum())
                    parsed = df[col].apply(parse_percent_universal)
                    rate = conversion_success_rate_non_null(parsed, original_col_backup)
                    conversion_report[col] = {"rate": rate}
                    preserves = _conversion_preserves_enough_data(parsed, original_col_backup, keep_ratio=keep_ratio)
                    after_nulls = int(parsed.isna().sum())
                    allow_change, null_msg = _check_null_inflation(before_nulls, after_nulls, col, self.null_inflation_threshold)

                    success_flag = False

                    if rate >= local_min_success and preserves and allow_change:
                        df[col] = parsed
                        lineage[col].append(f"stage2:percent_parse_rate_{rate:.3f}")
                        success_flag = True
                    elif (self.allow_partial_universal_conversion and int(parsed.notna().sum()) > 0 and preserves and allow_change):
                        df[col] = parsed
                        lineage[col].append(f"stage2:percent_parse_partial_rate_{rate:.3f}")
                        success_flag = True
                    else:
                        df[col] = original_col_backup
                        if not allow_change: self._violate("null_inflation_revert", null_msg, severity="error")

                    if success_flag and not pd.api.types.is_numeric_dtype(df[col]) and not is_safe_mode:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    if pd.api.types.is_numeric_dtype(df[col]):
                        bad = df[col].notna() & ((df[col] < 0) | (df[col] > 100))
                        if bad.any():
                            df.loc[bad, col] = np.nan
                            lineage[col].append("stage2:percent_out_of_range_to_nan")

                    self.learning_memory.record_outcome(col=col, role=role, confidence=confidence, success=success_flag)

            elif role == "bool":
                if self.enable_confidence_gating and confidence < self.role_confidence_threshold:
                    self._violate("low_confidence_skip", f"{col} role '{role}' confidence {confidence:.2f} below threshold", severity="info")
                    self.learning_memory.record_outcome(col=col, role=role, confidence=confidence, success=False)
                    continue

                effective_profile = self.cleaning_profile
                if confidence < 0.50: effective_profile = "safe"
                elif confidence < 0.70: effective_profile = "balanced"

                is_safe_mode = (effective_profile == "safe" and self.safe_mode_no_risky_coercions)
                is_high_conf = confidence >= self.high_confidence_threshold
                local_min_success = self.universal_min_success if is_high_conf else max(0.75, self.universal_min_success)
                keep_ratio = 0.85 if is_high_conf else 0.95

                if is_safe_mode: pass
                else:
                    before_nulls = int(original_col_backup.isna().sum())
                    parsed = df[col].apply(parse_bool_universal)
                    parsed_series = pd.Series(parsed)
                    rate = conversion_success_rate_non_null(parsed_series, original_col_backup)
                    conversion_report[col] = {"rate": rate}
                    preserves = _conversion_preserves_enough_data(parsed_series, original_col_backup, keep_ratio=keep_ratio)
                    after_nulls = int(parsed_series.isna().sum())
                    allow_change, null_msg = _check_null_inflation(before_nulls, after_nulls, col, self.null_inflation_threshold)

                    success_flag = False

                    if ((rate >= local_min_success or (self.allow_partial_universal_conversion and int(parsed_series.notna().sum()) > 0)) and preserves and allow_change):
                        mapped = parsed_series.map({True: 1.0, False: 0.0, np.nan: np.nan})
                        mapped = pd.to_numeric(mapped, errors="coerce")
                        mapped = mapped.where(mapped.isin([0.0, 1.0]), np.nan)
                        df[col] = mapped.astype("float64")
                        lineage[col].append(f"stage2:bool_parse_01_float_rate_{rate:.3f}")
                        success_flag = True
                    else:
                        df[col] = original_col_backup
                        if not allow_change: self._violate("null_inflation_revert", null_msg, severity="error")

                    self.learning_memory.record_outcome(col=col, role=role, confidence=confidence, success=success_flag)

            elif role == "date":
                if self.enable_confidence_gating and confidence < self.role_confidence_threshold:
                    self._violate("low_confidence_skip", f"{col} role '{role}' confidence {confidence:.2f} below threshold", severity="info")
                    self.learning_memory.record_outcome(col=col, role=role, confidence=confidence, success=False)
                    continue

                effective_profile = self.cleaning_profile
                if confidence < 0.50: effective_profile = "safe"
                elif confidence < 0.70: effective_profile = "balanced"

                is_safe_mode = (effective_profile == "safe" and self.safe_mode_no_risky_coercions)
                is_high_conf = confidence >= self.high_confidence_threshold
                local_min_success = 0.65 if is_high_conf else 0.80
                keep_ratio = 0.85 if is_high_conf else 0.95

                if is_safe_mode: pass
                else:
                    before_nulls = int(original_col_backup.isna().sum())
                    parsed = df[col].apply(parse_date_universal)
                    parsed_series = pd.Series(parsed)
                    rate = conversion_success_rate_non_null(parsed_series, original_col_backup)
                    conversion_report[col] = {"rate": rate}
                    preserves = _conversion_preserves_enough_data(parsed_series, original_col_backup, keep_ratio=keep_ratio)
                    after_nulls = int(parsed_series.isna().sum())
                    allow_change, null_msg = _check_null_inflation(before_nulls, after_nulls, col, self.null_inflation_threshold)
                    
                    success_flag = False

                    if rate >= local_min_success and preserves and allow_change:
                        df[col] = pd.to_datetime(parsed_series, errors="coerce")
                        lineage[col].append(f"stage2:date_parse_rate_{rate:.3f}")
                        success_flag = True
                    else:
                        df[col] = original_col_backup
                        if not allow_change: self._violate("null_inflation_revert", null_msg, severity="error")

                    self.learning_memory.record_outcome(col=col, role=role, confidence=confidence, success=success_flag)

        for col in df.columns:
            if "city" not in _norm_colname(col) or self._is_id_col(col) or pd.api.types.is_numeric_dtype(df[col]): continue

            before_nulls_city = int(df[col].isna().sum())
            s = self._normalize_text_safe(df[col])
            bad_mask = s.str.lower().isin(["unknown", "n/a", "na", "null"])
            valid_vals = s.loc[~bad_mask & s.notna()]

            if bad_mask.any() and not valid_vals.empty:
                if self.cleaning_profile != "safe":
                    mode_result = valid_vals.mode()
                    if not mode_result.empty:
                        df.loc[bad_mask, col] = mode_result.iloc[0]
                        lineage[col].append("stage2:city_mode_fill")

            after_nulls_city = int(df[col].isna().sum())
            allow_city_change, city_msg = _check_null_inflation(before_nulls_city, after_nulls_city, col, self.null_inflation_threshold)
            
            if not allow_city_change:
                df[col] = original_input_df[col]
                lineage[col].append("stage2:city_revert_null_guard")

        if self.impute_missing and self.cleaning_profile != "safe":
            for col in df.columns:
                if self._is_id_col(col): continue
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                        lineage[col].append("stage2:median_fill")
                    else:
                        mode_series = df[col].dropna().mode()
                        df[col] = df[col].fillna(mode_series.iloc[0] if not mode_series.empty else "unknown")
                        lineage[col].append("stage2:mode_fill")

        for col in df.columns:
            if "age" not in _norm_colname(col) or self._is_id_col(col): continue
            if col in protected_text_cols or (self.cleaning_profile == "safe" and self.safe_mode_no_risky_coercions): continue

            before_age_nulls = int(df[col].isna().sum())
            numeric = pd.to_numeric(df[col], errors="coerce")
            
            if pd.api.types.is_numeric_dtype(numeric):
                numeric = numeric.mask((numeric < 0) | (numeric > 100), np.nan)
                if self.impute_missing and self.cleaning_profile != "safe": 
                    numeric = numeric.fillna(numeric.median()).round()
                df[col] = numeric
                lineage[col].append("stage2:age_range_fix")

            baseline_nulls[col] = int(df[col].isna().sum())
            original_input_df[col] = df[col].copy()

        df = df[original_order]

        for col in df.columns:
            try:
                before_nulls = int(baseline_nulls.get(col, 0))
                after_nulls = int(df[col].isna().sum())
                before_safe = max(before_nulls, 1)
                increase_ratio = (after_nulls - before_nulls) / before_safe

                if increase_ratio > self.null_inflation_threshold:
                    df[col] = original_input_df[col]
                    lineage[col].append("stage2:null_guard_revert")
                    self._violate("null_inflation_revert", f"CRITICAL: {col} nulls {before_nulls}->{after_nulls} (+{increase_ratio:.1%}) > {self.null_inflation_threshold*100}% limit -> REVERTED", severity="error")
            except Exception as e:
                self._violate("null_guard_error", f"Null guard failed for {col}: {str(e)}", severity="warn")

        final_null_summary = []
        for col in df.columns:
            before = baseline_nulls[col]
            after = int(df[col].isna().sum())
            final_null_summary.append({"column": col, "before_nulls": before, "after_nulls": after, "change_ratio": float((after - before) / max(before, 1))})

        self.last_report_ = {
            "cleaning_profile": self.cleaning_profile,
            "violations": self.violations_,
            "lineage": lineage,
            "roles": role_report,
            "role_confidence": confidence_report,
            "low_confidence_columns": [c for c, v in confidence_report.items() if v < self.role_confidence_threshold],
            "protected_text_columns": list(protected_text_cols),
            "conversion_rates": conversion_report,
            "drift": drift_report,
            "null_inflation_summary": final_null_summary,
            "high_missing_cols": list(self.high_missing_cols_) if hasattr(self, "high_missing_cols_") else [],
            "constant_cols": list(self.constant_cols_) if hasattr(self, "constant_cols_") else [],
            "schema": list(getattr(self, "schema_", [])),
            "learning_feedback": {
                "memory_size": len(getattr(self.learning_memory, "memory", {})),
            },
        }

        _check_df_integrity(original_input_df, df, "QualityRuleCleaner")
        return df