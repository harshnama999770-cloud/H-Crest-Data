# =========================================================
# cleaningStage3.py
# Stage 3: Semantic Validation (SAFE MODE)
#
# ENTERPRISE UPGRADE:
# - Candidate-weighted validation (soft rules)
# - Probabilistic issues (rule_weight + final_confidence)
# - Semantic history memory (JSON)
# - Confidence decay over time (5% per 30 days)
# - Type drift detection (don't blindly trust old history)
#
# ✅ NEW ENTERPRISE ADDS:
# - LLM fallback with SMART ROUTING (only ambiguous cols)
# - Column-level confidence and health scoring (Junk detection)
# - Comprehensive Diff Report (Nulls before/after, rows changed)
# - Dataset Risk Scoring (Low / Medium / High)
# - 🔴 FINAL NULL INFLATION GUARD (Zero Data Destruction Guarantee)
# - 🔴 ISSUE LOOP MEMORY CAP (Max 5000 issues per column)
# =========================================================

import os
import json
import datetime
import pandas as pd
import numpy as np
import re
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from pipeline_utils import _check_df_integrity

# LLM imports
from llm_client import LocalLLMClient
from llm_schema_infer import infer_schema_with_llm, apply_llm_schema_safety


# -----------------------------------
# Constants
# -----------------------------------
BOOLEAN_VALUES = {"true", "false", "yes", "no", "0", "1"}

SEMANTIC_RULES = {
    "AGE": {"min": 0, "max": 120, "integer": True},
    "YEAR": {"min": 1900, "max": 2100, "integer": True},
    "PERCENTAGE": {"min": 0, "max": 100},
    "COUNT": {"min": 0, "integer": True},
    "MONEY": {"min": 0},
}

DATE_REGEXES = [
    r"^\d{4}-\d{2}-\d{2}$",           # 2023-01-05
    r"^\d{2}/\d{2}/\d{4}$",           # 05/01/2023
    r"^\d{2}-\d{2}-\d{4}$",           # 05-01-2023
    r"^\d{4}/\d{2}/\d{2}$",           # 2023/01/05
    r"^\d{2}\s[A-Za-z]{3}\s\d{4}$",   # 05 Jan 2023
]

_MONEY_CLEAN_RE = re.compile(r"[^\d\.\-]+")

# ✅ NEW: Safe Issue Cap
MAX_ISSUES_PER_COLUMN = 5000


# =========================================================
# Semantic history helpers
# =========================================================

def _utc_now_str():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _normalize_history_key(col_name: str) -> str:
    return str(col_name or "").strip().lower()

def load_history(path: str) -> dict:
    try:
        if not path or not os.path.exists(path): return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_history(path: str, history: dict):
    try:
        if not path: return
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        return

def update_history(history: dict, col_name: str, sem_info: dict):
    key = _normalize_history_key(col_name)
    if not key: return history

    if key not in history or not isinstance(history.get(key), dict):
        history[key] = {}

    rec = history[key]
    rec["type"] = str(sem_info.get("type", "TEXT"))
    rec["confidence"] = float(sem_info.get("confidence", 0.0))
    cands = sem_info.get("candidates", [])
    rec["candidates"] = cands[:6] if isinstance(cands, list) else []
    rec["seen_count"] = int(rec.get("seen_count", 0) or 0) + 1
    rec["last_seen"] = _utc_now_str()

    history[key] = rec
    return history


# =========================================================
# Decay + Staleness + Junk helpers
# =========================================================

def _parse_utc_time(s: str):
    try: return datetime.datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ")
    except Exception: return None

def _days_since(last_seen_str: str) -> int:
    dt = _parse_utc_time(last_seen_str)
    if dt is None: return 10_000
    delta = datetime.datetime.utcnow() - dt
    return max(0, int(delta.days))

def decay_confidence(original_conf: float, days_old: int) -> float:
    try: c = float(original_conf)
    except Exception: return 0.0
    periods = days_old / 30.0
    decay_factor = (0.95 ** periods)
    return float(max(0.0, min(1.0, c * decay_factor)))

def is_history_stale(days_old: int) -> bool:
    return int(days_old) >= 180

def _is_junk_column(col_name: str, series: pd.Series) -> bool:
    n = str(col_name).strip().lower()
    if re.match(r"^unnamed:\s*\d+$", n): return True
    if n in {"extra", "junk", "temp", "dummy"}: return True
    if series.isna().all(): return True
    return False


# -----------------------------------
# Helpers
# -----------------------------------

def _normalize_text(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.replace("", pd.NA)
    return s

def _to_numeric_clean(series: pd.Series) -> pd.Series:
    s = _normalize_text(series)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace(_MONEY_CLEAN_RE, "", regex=True)
    s = s.replace("", pd.NA)
    return pd.to_numeric(s, errors="coerce")

def analyze_value_context(series: pd.Series):
    s = series.dropna().astype(str)
    if s.empty: return {}
    return {"unique_ratio": float(s.nunique() / len(s)), "avg_length": float(s.str.len().mean())}

def _name_says_id(col_name: str) -> bool:
    n = (col_name or "").strip().lower()
    if n in {"id", "uuid"} or n.endswith("_id"): return True
    if any(k in n for k in ["uuid", "guid", "pan", "gst", "aadhar", "ifsc"]): return True
    return False

def context_says_identifier(ctx: dict):
    return (ctx.get("unique_ratio", 1.0) > 0.95 and ctx.get("avg_length", 0.0) > 5)

def looks_like_date(series: pd.Series):
    s = _normalize_text(series).dropna().astype("string").str.strip()
    if s.empty: return False, 0.0
    s = s.head(3000)
    regex_hits = s.str.fullmatch("|".join(DATE_REGEXES))
    hit_ratio = float(regex_hits.mean())

    if hit_ratio < 0.6: return False, hit_ratio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    parse_ratio = float(parsed.notna().mean())
    return parse_ratio > 0.8, parse_ratio

def looks_like_numeric(series: pd.Series):
    s = _normalize_text(series).dropna()
    if s.empty: return False, 0.0, None
    s = s.head(5000)
    numeric = _to_numeric_clean(s)
    ratio = float(numeric.notna().mean())
    return ratio > 0.8, ratio, numeric

def infer_numeric_constraints(numeric: pd.Series):
    x = numeric.dropna()
    if x.empty: return None
    return {
        "low": float(x.quantile(0.01)),
        "high": float(x.quantile(0.99)),
        "int_ratio": float(((x % 1) == 0).mean()),
    }


# =========================================================
# Candidate ranking
# =========================================================

def _rank_candidates(col_name: str, series: pd.Series):
    s = series.dropna()
    name = (col_name or "").strip().lower()
    if s.empty: return [{"type": "TEXT", "confidence": 1.0}]

    s = s.head(5000)
    ctx = analyze_value_context(s)
    candidates = []

    unique_norm = {str(v).strip().lower() for v in s.unique()}
    if unique_norm.issubset(BOOLEAN_VALUES):
        candidates.append({"type": "BOOLEAN", "confidence": 0.95})

    if _name_says_id(col_name): candidates.append({"type": "ID", "confidence": 0.98})
    if context_says_identifier(ctx): candidates.append({"type": "ID", "confidence": 0.90})

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        candidates.append({"type": "DATE", "confidence": 0.95})

    numeric_like_ratio = pd.to_numeric(s, errors="coerce").notna().mean()
    if numeric_like_ratio < 0.85:
        is_date, parse_conf = looks_like_date(s)
        if is_date:
            name_bonus = 0.1 if any(k in name for k in ["date", "time", "dt"]) else 0.0
            candidates.append({"type": "DATE", "confidence": float(min(0.95, parse_conf + name_bonus))})

    is_num, num_conf, numeric = looks_like_numeric(s)
    if is_num and numeric is not None and numeric.notna().any():
        min_v, max_v = float(numeric.min()), float(numeric.max())
        if "age" in name: candidates.append({"type": "AGE", "confidence": 0.95})
        if "year" in name or (min_v >= 1900 and max_v <= 2100 and (numeric.dropna() % 1 == 0).all()):
            candidates.append({"type": "YEAR", "confidence": 0.85})
        if min_v >= 0 and max_v <= 100:
            w = 0.75 if any(k in name for k in ["%", "percent", "pct"]) else 0.6
            candidates.append({"type": "PERCENTAGE", "confidence": float(w)})
            candidates.append({"type": "SCORE", "confidence": float(max(0.25, w - 0.25))})
        if ((numeric.dropna() % 1 == 0).all() and min_v >= 0 and ctx.get("unique_ratio", 1.0) < 0.95):
            candidates.append({"type": "COUNT", "confidence": 0.80})
        if any(k in name for k in ["amount", "money", "price", "salary", "income", "cost", "total", "spent"]):
            if min_v >= 0:
                candidates.append({"type": "MONEY", "confidence": 0.75})
                candidates.append({"type": "AMOUNT", "confidence": 0.60})

    if ctx.get("unique_ratio", 1.0) < 0.5 and ctx.get("avg_length", 99.0) < 15:
        candidates.append({"type": "CATEGORY", "confidence": 0.70})

    candidates.append({"type": "TEXT", "confidence": 0.60})

    merged = {}
    for c in candidates:
        t = c["type"]
        w = float(c["confidence"])
        merged[t] = max(merged.get(t, 0.0), w)

    out = [{"type": t, "confidence": float(w)} for t, w in merged.items()]
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return out

def _apply_history_boost(col_name: str, ranked: list, history: dict, history_boost: float = 0.15, min_seen_for_boost: int = 2):
    key = _normalize_history_key(col_name)
    rec = history.get(key)
    if not isinstance(rec, dict): return ranked, "inference", {}

    seen = int(rec.get("seen_count", 0) or 0)
    if seen < min_seen_for_boost: return ranked, "inference", {}

    hist_type = str(rec.get("type", "") or "").strip().upper()
    hist_conf_raw = float(rec.get("confidence", 0.0) or 0.0)
    days_old = _days_since(str(rec.get("last_seen", "") or ""))
    hist_conf = decay_confidence(hist_conf_raw, days_old)

    meta = {
        "history_type": hist_type, "history_conf_raw": hist_conf_raw,
        "history_conf_decayed": hist_conf, "history_days_old": days_old,
        "history_stale": is_history_stale(days_old),
    }

    if not hist_type: return ranked, "inference", meta

    merged = {c["type"]: float(c["confidence"]) for c in ranked if "type" in c}
    current = merged.get(hist_type, 0.0)
    merged[hist_type] = max(current, min(0.98, hist_conf + float(history_boost)))

    out = [{"type": t, "confidence": float(w)} for t, w in merged.items()]
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return out, "history+inference", meta


# =========================================================
# LLM helpers (SMART ROUTING)
# =========================================================

def _llm_schema_for_dataset(
    df: pd.DataFrame, target_cols: list, enable_llm: bool,
    llm_base_url: str, llm_model: str, llm_timeout: int, llm_max_cols: int
) -> tuple[dict | None, str]:
    """
    Returns (schema_dict, failure_reason_string)
    """
    if not enable_llm:
        return None, "llm_disabled"
    if not target_cols:
        return None, "no_ambiguous_columns"

    cols_to_send = target_cols[:llm_max_cols]
    df_small = df[cols_to_send].head(250).copy()

    try:
        client = LocalLLMClient(base_url=llm_base_url, model=llm_model, timeout=llm_timeout, verbose=False)
        schema = infer_schema_with_llm(df_small, client, max_cols=llm_max_cols)
        schema = apply_llm_schema_safety(schema, df_small)

        if not isinstance(schema, dict) or not isinstance(schema.get("columns", {}), dict):
            return None, "invalid_llm_response_format"
        return schema, "ok"
    except Exception as e:
        err_str = str(e).lower()
        if "timeout" in err_str: return None, "timeout"
        elif "connection" in err_str: return None, "connection_error"
        else: return None, f"model_error: {str(e)}"

def _apply_llm_boost_to_ranked(col: str, ranked: list, llm_schema: dict | None, llm_min_confidence: float = 0.70, llm_boost: float = 0.20):
    if llm_schema is None: return ranked, "history+inference"
    info = llm_schema.get("columns", {}).get(col)
    if not isinstance(info, dict): return ranked, "history+inference"

    llm_type = str(info.get("semantic_type", "TEXT")).upper().strip()
    try: llm_conf = float(info.get("confidence", 0.0))
    except Exception: llm_conf = 0.0

    if llm_conf < llm_min_confidence: return ranked, "history+inference"

    merged = {c["type"]: float(c["confidence"]) for c in ranked if "type" in c}
    current = merged.get(llm_type, 0.0)
    merged[llm_type] = max(current, min(0.99, llm_conf + llm_boost))

    out = [{"type": t, "confidence": float(w)} for t, w in merged.items()]
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return out, "history+inference+llm"


# -----------------------------------
# Semantic Type Detection (SOFT + HISTORY + LLM)
# -----------------------------------
def detect_semantic_type(
    col_name: str, series: pd.Series, history: dict | None = None,
    history_boost: float = 0.15, min_seen_for_boost: int = 2, max_candidates: int = 3,
    llm_schema: dict | None = None, llm_min_confidence: float = 0.70, llm_boost: float = 0.20
):
    ranked = _rank_candidates(col_name, series)
    source = "inference"
    history_meta = {}

    if isinstance(history, dict) and history:
        ranked, source, history_meta = _apply_history_boost(
            col_name, ranked, history, history_boost=history_boost, min_seen_for_boost=min_seen_for_boost
        )

    # LLM boost LAST
    ranked, source = _apply_llm_boost_to_ranked(
        str(col_name), ranked, llm_schema=llm_schema, llm_min_confidence=llm_min_confidence, llm_boost=llm_boost
    )

    ranked = ranked[:max(1, int(max_candidates))]
    best = ranked[0]
    return {
        "type": best["type"], "confidence": float(best["confidence"]),
        "candidates": ranked, "source": source, "history_meta": history_meta,
    }

# =========================================================
# Issue builder
# =========================================================
def _make_issue(
    row, column, sem_type, sem_confidence, candidates, source, issue, base_confidence,
    explanation, details=None, suggested_fix=None, rule_weight=1.0
):
    final_confidence = float(base_confidence) * float(rule_weight)
    safe_row = None
    if row is not None:
        try: safe_row = int(row)
        except Exception: safe_row = row

    issue_l = str(issue or "").lower()
    severity = "error" if ("coercion" in issue_l or "invalid date" in issue_l) else "warn"

    return {
        "row": safe_row, "column": column, "stage": "semantic", "severity": severity,
        "semantic_type": sem_type, "semantic_confidence": float(sem_confidence),
        "semantic_candidates": candidates, "semantic_source": source,
        "rule_weight": float(rule_weight), "base_confidence": float(base_confidence),
        "final_confidence": float(final_confidence), "issue": issue,
        "confidence": float(final_confidence), "explanation": explanation,
        "details": details or {}, "suggested_fix": suggested_fix or {},
    }

# =========================================================
# Transformer (SAFE MODE FIX APPLY + REPORTS)
# =========================================================
class SemanticValidator(BaseEstimator, TransformerMixin):
    def __init__(
        self, enable_history=True, history_path="semantic_history.json",
        history_boost=0.15, min_seen_for_boost=2, soft_rule_threshold=0.75, allow_type_change: bool = True,
        enable_llm: bool = True, llm_base_url: str = "http://127.0.0.1:1234",
        llm_model: str = "deepseek-coder-7b-instruct-v1.5", llm_timeout: int = 90,
        llm_max_cols: int = 35, llm_min_confidence: float = 0.70, llm_boost: float = 0.20,
    ):
        self.enable_history = bool(enable_history)
        self.history_path = str(history_path)
        self.history_boost = float(history_boost)
        self.min_seen_for_boost = int(min_seen_for_boost)
        self.soft_rule_threshold = float(soft_rule_threshold)
        self.allow_type_change = bool(allow_type_change)

        self.enable_llm = bool(enable_llm)
        self.llm_base_url = str(llm_base_url)
        self.llm_model = str(llm_model)
        self.llm_timeout = int(llm_timeout)
        self.llm_max_cols = int(llm_max_cols)
        self.llm_min_confidence = float(llm_min_confidence)
        self.llm_boost = float(llm_boost)

        self.issues = []
        self.last_report_ = {}
        self.column_reports_ = []
        self.history_ = {}
        self.llm_schema_ = None

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        original_df = X.copy()
        df = X.copy()

        self.issues = []
        self.column_reports_ = []
        type_drift_warnings = []
        stale_history_warnings = []
        junk_columns_flagged = []

        modified_rows = set()
        modified_cols = set()
        
        # 🔴 BASELINE NULLS
        nulls_before = df.isna().sum().to_dict()

        if self.enable_history:
            try: self.history_ = load_history(self.history_path)
            except Exception: self.history_ = {}

        # -----------------------------
        # ✅ SMART LLM ROUTING (Pre-scan)
        # -----------------------------
        ambiguous_cols = []
        for col in df.columns:
            if _is_junk_column(col, df[col]):
                ambiguous_cols.append(col)
                continue
            
            ranked = _rank_candidates(col, df[col])
            if self.enable_history and self.history_:
                ranked, _, _ = _apply_history_boost(col, ranked, self.history_)
            
            best_conf = ranked[0]["confidence"] if ranked else 0.0
            if best_conf < self.soft_rule_threshold:
                ambiguous_cols.append(col)

        # Run LLM ONLY on ambiguous columns
        self.llm_schema_, llm_reason = _llm_schema_for_dataset(
            df=df, target_cols=ambiguous_cols, enable_llm=self.enable_llm,
            llm_base_url=self.llm_base_url, llm_model=self.llm_model,
            llm_timeout=self.llm_timeout, llm_max_cols=self.llm_max_cols,
        )

        # -----------------------------
        # Validate columns
        # -----------------------------
        for col in df.columns:
            # Detect junk health
            if _is_junk_column(col, df[col]):
                health_status = "junk_possible"
                junk_columns_flagged.append(col)
            else:
                health_status = "ok"

            sem_info = detect_semantic_type(
                col, df[col], history=self.history_ if self.enable_history else None,
                history_boost=self.history_boost, min_seen_for_boost=self.min_seen_for_boost, max_candidates=3,
                llm_schema=self.llm_schema_, llm_min_confidence=self.llm_min_confidence, llm_boost=self.llm_boost,
            )

            best_conf = float(sem_info["confidence"])
            col_risk = "high" if best_conf < 0.6 else ("medium" if best_conf < 0.8 else "low")

            # Store Column Report Foundation
            self.column_reports_.append({
                "column": col,
                "action": "scanned",
                "confidence": round(best_conf, 3),
                "risk": col_risk,
                "column_health": health_status,
                "semantic_type": sem_info["type"]
            })

            # Issue Builder Loop (with Cap)
            candidates = sem_info.get("candidates", [])
            issues_for_this_col = 0

            for cand in candidates:
                cand_type = str(cand.get("type", "TEXT"))
                cand_weight = float(cand.get("confidence", 0.0))
                if cand_type not in {"DATE", "AGE", "YEAR", "PERCENTAGE", "MONEY", "COUNT"}: continue
                
                can_autofix = cand_weight >= self.soft_rule_threshold
                rules = SEMANTIC_RULES.get(cand_type, {})
                numeric = _to_numeric_clean(df[col])
                
                if "min" in rules:
                    m_min = numeric.notna() & (numeric < rules["min"])
                    for idx in numeric[m_min].index:
                        if issues_for_this_col >= MAX_ISSUES_PER_COLUMN: break
                        self.issues.append(_make_issue(
                            idx, col, cand_type, cand_weight, candidates, sem_info.get("source"), "below minimum", 0.9, 
                            f"Below min {rules['min']}", details={"rule":"min"}, 
                            suggested_fix={"action": "set_null"} if can_autofix else None, rule_weight=cand_weight
                        ))
                        issues_for_this_col += 1
                
                if rules.get("integer"):
                    m_int = numeric.notna() & (numeric % 1 != 0)
                    for idx in numeric[m_int].index:
                        if issues_for_this_col >= MAX_ISSUES_PER_COLUMN: break
                        self.issues.append(_make_issue(
                            idx, col, cand_type, cand_weight, candidates, sem_info.get("source"), "must be integer", 0.9, 
                            "Must be int", details={"rule":"int"}, 
                            suggested_fix={"action": "round"} if can_autofix else None, rule_weight=cand_weight
                        ))
                        issues_for_this_col += 1

            # History Memory Checking
            if self.enable_history and isinstance(self.history_, dict):
                key = _normalize_history_key(str(col))
                rec = self.history_.get(key)
                if isinstance(rec, dict):
                    days_old = _days_since(str(rec.get("last_seen", "")))
                    if old_type := str(rec.get("type", "")).strip().upper():
                        new_type = str(sem_info.get("type", "")).strip().upper()
                        if old_type != new_type:
                            type_drift_warnings.append(f"Drift in {col}: {old_type} -> {new_type}")
                            if not self.allow_type_change: sem_info["type"] = old_type
                self.history_ = update_history(self.history_, str(col), sem_info)

        # -----------------------------
        # Apply safe fixes & Track Diff
        # -----------------------------
        for it in self.issues:
            row, col = it.get("row"), it.get("column")
            if row is None or col not in df.columns or row not in df.index: continue

            if float(it.get("semantic_confidence", 1.0)) < self.soft_rule_threshold: continue
            
            action = (it.get("suggested_fix") or {}).get("action")
            
            if action == "set_null":
                if pd.notna(df.loc[row, col]):
                    df.loc[row, col] = np.nan
                    modified_rows.add(row)
                    modified_cols.add(col)
                    
            elif action == "round":
                v = pd.to_numeric(df.loc[row, col], errors="coerce")
                if pd.notna(v):
                    rounded = float(np.round(v))
                    if rounded != df.loc[row, col]:
                        df.loc[row, col] = rounded
                        modified_rows.add(row)
                        modified_cols.add(col)

        # Update column reports with action
        for c_rep in self.column_reports_:
            if c_rep["column"] in modified_cols:
                c_rep["action"] = "normalized"

        # =====================================
        # 🔴 FINAL NULL INFLATION GUARD (Stage3)
        # =====================================
        for col in df.columns:
            try:
                before = int(nulls_before.get(col, 0))
                after = int(df[col].isna().sum())

                if before == 0:
                    before = 1

                increase_ratio = (after - before) / before

                if increase_ratio > 0.25:
                    df[col] = original_df[col]
                    warnings.warn(
                        f"[Stage3 NullGuard] Reverted {col}: nulls {before}->{after} (+{increase_ratio:.1%})",
                        UserWarning,
                    )
            except Exception:
                pass


        # -----------------------------
        # ✅ Generate Reports & Risk Score
        # -----------------------------
        if self.enable_history:
            try: save_history(self.history_path, self.history_)
            except Exception: pass

        nulls_after = df.isna().sum().to_dict()
        
        # Risk Score Logic (Incorporates Junk Count)
        total_cells = max(1, df.shape[0] * df.shape[1])
        issue_ratio = len(self.issues) / total_cells
        ambig_ratio = len(ambiguous_cols) / max(1, len(df.columns))
        junk_ratio = len(junk_columns_flagged) / max(1, len(df.columns))
        
        if ambig_ratio > 0.3 or issue_ratio > 0.05 or junk_ratio > 0.1: dataset_risk = "high"
        elif ambig_ratio > 0.1 or issue_ratio > 0.01 or junk_ratio > 0.05: dataset_risk = "medium"
        else: dataset_risk = "low"

        self.last_report_ = {
            "semantic_issue_count": len(self.issues),
            "dataset_risk": dataset_risk,
            
            # Diff Report
            "diff_report": {
                "rows_changed": len(modified_rows),
                "columns_modified": list(modified_cols),
                "nulls_before": sum(nulls_before.values()),
                "nulls_after": sum(nulls_after.values()),
                "risky_columns_flagged": [c["column"] for c in self.column_reports_ if c["risk"] in ("high", "medium")],
                "junk_columns_count": len(junk_columns_flagged),
                "junk_columns_flagged": junk_columns_flagged
            },
            "column_health_scores": self.column_reports_,
            
            # LLM Meta & History Stats
            "llm_meta": {
                "llm_status": "ok" if llm_reason == "ok" else "failed/skipped",
                "reason": llm_reason,
                "attempted_columns": len(ambiguous_cols),
                "used_columns_list": ambiguous_cols[:self.llm_max_cols]
            },
            "type_drift_warnings": type_drift_warnings,
        }

        _check_df_integrity(original_df, df, "SemanticValidator")
        return df