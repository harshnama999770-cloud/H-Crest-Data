# =========================================================
# llm_schema_infer.py
# SAFE Schema + Column Meaning Inference using Local LLM
#
# Works with LM Studio (DeepSeekCoder 7B)
#
# Purpose:
# - Infer semantic meaning of columns safely
# - Suggest column types (TEXT, MONEY, DATE, AGE, ID, etc.)
# - Suggest rules (pattern regex, min/max, percent range)
#
# IMPORTANT SAFETY RULE:
# - LLM never modifies your dataframe.
# - It only returns suggestions.
# =========================================================

import json
import re
import warnings
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from functools import wraps

from llm_client import LocalLLMClient

# =========================================================
# RETRY + BACKOFF DECORATOR
# =========================================================

def with_retry_and_backoff(
    max_retries: int = 1,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    timeout_factor: float = 1.5,
    allowed_exceptions: tuple = (Exception,),
    verbose: bool = False
):
    """
    Retry decorator with exponential backoff and timeout awareness.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timeout = kwargs.pop('timeout', None)
            if timeout:
                kwargs['timeout'] = int(timeout * timeout_factor)
            
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    if verbose and attempt > 0:
                        print(f"Retry attempt {attempt}/{max_retries}")
                    return func(*args, **kwargs)
                
                except allowed_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise
                    
                    # Exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if verbose:
                        print(f"Retrying in {delay:.1f}s... ({str(e)[:100]})")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator

# =========================================================
# Helpers
# =========================================================

def _safe_str(x):
    try:
        return str(x)
    except Exception:
        return "<unprintable>"

def _is_mostly_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
    s = series.dropna()
    if s.empty:
        return False

    # Speed guard (never scan millions of rows)
    s = s.head(5000)

    numeric = pd.to_numeric(s, errors="coerce")
    return float(numeric.notna().mean()) >= float(threshold)

def _is_mostly_datetime(series: pd.Series, threshold: float = 0.7) -> bool:
    """
    Warning-free datetime detection.

    Fixes:
    - Removes pandas "Could not infer format" warning spam
    - Faster than raw dateutil fallback
    - Handles mixed formats safely
    """
    s = series.dropna()
    if s.empty:
        return False

    # Speed guard
    s = s.head(3000)

    # Convert to string safely (prevents weird dtype behavior)
    s = s.astype("string").str.strip()

    # If values are too short, skip (prevents false positives)
    if len(s) < 3:
        return False

    # Try common formats first (no warning, fast)
    common_formats = [
        "%Y-%m-%d",                # 2024-01-31
        "%d-%m-%Y",                # 31-01-2024
        "%m-%d-%Y",                # 01-31-2024
        "%d/%m/%Y",                # 31/01/2024
        "%m/%d/%Y",                # 01/31/2024
        "%Y/%m/%d",                # 2024/01/31
        "%d.%m.%Y",                # 31.01.2024
        "%Y-%m-%d %H:%M:%S",       # 2024-01-31 10:30:00
        "%Y-%m-%d %H:%M",          # 2024-01-31 10:30
        "%Y-%m-%dT%H:%M:%S",       # 2024-01-31T10:30:00
        "%Y-%m-%dT%H:%M",          # 2024-01-31T10:30
    ]

    best_ratio = 0.0

    for fmt in common_formats:
        try:
            parsed = pd.to_datetime(s, errors="coerce", format=fmt)
            ratio = float(parsed.notna().mean())
            if ratio > best_ratio:
                best_ratio = ratio
        except Exception:
            continue

        # early exit if already good enough
        if best_ratio >= float(threshold):
            return True

    # Fallback: dateutil (slow) but silence the warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(s, errors="coerce")

    ratio = float(parsed.notna().mean())
    return ratio >= float(threshold)

def _sample_values(series: pd.Series, k: int = 12) -> List[str]:
    """
    Returns small safe sample values.
    """
    s = series.dropna()
    if s.empty:
        return []

    # unique sample for variety
    s = s.astype("string").dropna().unique().tolist()

    out = []
    for v in s[:k]:
        out.append(_safe_str(v)[:120])
    return out

def _dataset_summary(df: pd.DataFrame, max_cols: int = 35) -> Dict[str, Any]:
    """
    Creates a safe summary of dataset for LLM.
    We do NOT send the full data (privacy + speed).
    """

    summary = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": [],
    }

    cols = list(df.columns)[:max_cols]

    for c in cols:
        # speed guard: do not scan huge columns
        s = df[c].head(5000)

        col_info = {
            "name": str(c),
            "dtype": str(s.dtype),
            "missing_percent": float(round(s.isna().mean() * 100, 2)),
            "unique_count": int(s.nunique(dropna=True)),
            "sample_values": _sample_values(s, k=10),
        }

        if _is_mostly_numeric(s):
            col_info["local_hint"] = "numeric-like"
        elif _is_mostly_datetime(s):
            col_info["local_hint"] = "datetime-like"
        else:
            col_info["local_hint"] = "text-like"

        summary["columns"].append(col_info)

    return summary

def _extract_json_block(raw: str) -> Optional[dict]:
    """
    Extracts JSON object from raw LLM output safely.
    """
    if raw is None:
        return None

    raw = str(raw).strip()
    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end != -1 and end > start:
        block = raw[start: end + 1]
        try:
            return json.loads(block)
        except Exception:
            return None

    return None

# =========================================================
# MAIN LLM Schema Inference (FULL) - WITH RETRY
# =========================================================

DEFAULT_SYSTEM_PROMPT = """
You are an enterprise data cleaning assistant.

Your job:
- Infer what each column means from name + sample values.
- Suggest semantic_type for each column.
- Suggest safe validation rules.

CRITICAL RULES:
1) NEVER modify or rewrite the dataset.
2) NEVER hallucinate columns not present.
3) If unsure, mark type as TEXT with low confidence.
4) IDs must be protected (ID columns should never be converted to numeric).
5) Return ONLY valid JSON. No extra explanation.
"""

DEFAULT_USER_PROMPT_TEMPLATE = """
Infer schema for this dataset.

Return JSON with this format:

{
  "dataset_level": {
    "recommended_cleaning_mode": "conservative|balanced|aggressive",
    "warnings": ["..."]
  },
  "columns": {
    "<col_name>": {
      "semantic_type": "ID|TEXT|CATEGORY|DATE|DATETIME|EMAIL|PHONE|ZIP|MONEY|PERCENTAGE|AGE|YEAR|COUNT|BOOLEAN|URL",
      "confidence": 0.0,
      "reason": "...",
      "protect_from_conversion": true|false,
      "suggested_rules": {
        "pattern_regex": "... or null",
        "min": number or null,
        "max": number or null,
        "integer": true|false|null
      }
    }
  }
}

Dataset summary:
{DATASET_SUMMARY_JSON}
"""

@with_retry_and_backoff(max_retries=1, base_delay=2.0, max_delay=8.0, timeout_factor=1.2, verbose=True)
def _safe_llm_chat(client, **kwargs):
    """Wrapped LLM call with retry."""
    return client.chat(**kwargs)

def infer_schema_with_llm(
    df: pd.DataFrame,
    llm_client: LocalLLMClient,
    max_cols: int = 35,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    timeout: int = 90,
) -> Dict[str, Any]:
    """
    Uses LLM to infer schema safely WITH RETRY + BACKOFF.
    """

    summary = _dataset_summary(df, max_cols=max_cols)

    user_prompt = DEFAULT_USER_PROMPT_TEMPLATE.replace(
        "{DATASET_SUMMARY_JSON}",
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    try:
        raw = _safe_llm_chat(
            llm_client,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )

        parsed = _extract_json_block(raw)

        if not isinstance(parsed, dict):
            return {
                "error": "LLM did not return valid JSON",
                "raw": str(raw)[:2000],
                "fallback": {
                    "dataset_level": {
                        "recommended_cleaning_mode": "balanced",
                        "warnings": ["LLM output invalid JSON. Using safe fallback."]
                    },
                    "columns": {}
                }
            }

        return parsed

    except Exception as e:
        return {
            "error": f"LLM call failed: {str(e)}",
            "fallback": {
                "dataset_level": {
                    "recommended_cleaning_mode": "balanced",
                    "warnings": [f"LLM failed: {str(e)}"]
                },
                "columns": {}
            }
        }

# =========================================================
# NEW: STAGE0 HELPER (ALIAS ONLY) - WITH RETRY + BACKOFF
# =========================================================

# ✅ Enterprise whitelist of canonical names
CANONICAL_ALIAS_VOCAB = {
    "email", "phone", "user_id", "customer_id",
    "order_id", "invoice_id", "transaction_id",
    "order_date", "delivery_date", "return_date",
    "created_at", "updated_at",
    "total_amount", "amount_paid", "refund_amount",
    "qty", "price", "country", "state", "city", "zip",
    "dob", "age", "year"
}

ALIAS_SYSTEM_PROMPT = """
You are a schema mapping assistant.

Task:
- Suggest safe column alias renames.
- Use ONLY the given columns.
- Do NOT modify data.
- Return ONLY JSON.

IMPORTANT:
- If unsure, do not suggest alias.
- Only suggest high-confidence mappings.
"""

ALIAS_USER_PROMPT_TEMPLATE = """
Given this dataset summary, suggest safe alias renames.

Return ONLY JSON in this exact format:

{
  "status": "success",
  "aliases": {
    "<raw_col>": {
      "to": "<canonical_col>",
      "confidence": 0.0,
      "reason": "..."
    }
  }
}

Rules:
- DO NOT suggest more than 12 aliases.
- Only suggest canonical column names from this list:
  email, phone, user_id, customer_id, order_id, invoice_id, transaction_id,
  order_date, delivery_date, return_date, created_at, updated_at,
  total_amount, amount_paid, refund_amount, qty, price,
  country, state, city, zip, dob, age, year

Dataset summary:
{DATASET_SUMMARY_JSON}
"""

@with_retry_and_backoff(max_retries=1, base_delay=1.5, max_delay=6.0, timeout_factor=1.3, verbose=True)
def _safe_alias_llm_chat(client, **kwargs):
    """Wrapped alias LLM call with retry."""
    return client.chat(**kwargs)

def infer_column_aliases_with_llm(
    df: pd.DataFrame,
    model: str = "deepseek-coder-7b-instruct-v1.5",
    base_url: str = "http://127.0.0.1:1234",
    max_rows: int = 25,
    timeout: int = 90,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> Dict[str, Any]:
    """
    Stage0 helper WITH RETRY + BACKOFF:
    - Returns alias suggestions only.
    - Designed to be fast and safe.
    """

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"status": "skipped", "reason": "empty dataframe", "aliases": {}}

    df_small = df.head(int(max_rows)).copy()
    summary = _dataset_summary(df_small, max_cols=35)

    user_prompt = ALIAS_USER_PROMPT_TEMPLATE.replace(
        "{DATASET_SUMMARY_JSON}",
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    try:
        client = LocalLLMClient(
            base_url=base_url,
            model=model,
            timeout=int(timeout),
            verbose=False,
        )

        raw = _safe_alias_llm_chat(
            client,
            messages=[
                {"role": "system", "content": ALIAS_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            timeout=timeout
        )

        parsed = _extract_json_block(raw)

        if not isinstance(parsed, dict):
            return {
                "status": "failed",
                "error": "LLM did not return valid JSON",
                "raw": str(raw)[:1500],
                "aliases": {}
            }

        aliases = parsed.get("aliases", {})
        if not isinstance(aliases, dict):
            aliases = {}

        # ---------------------------------------------------
        # ENTERPRISE SAFETY FILTERS
        # ---------------------------------------------------

        safe_aliases = {}
        used_targets = set()

        for i, (raw_col, info) in enumerate(aliases.items()):
            if i >= 12:
                break

            if not isinstance(raw_col, str):
                continue

            if raw_col not in df.columns:
                continue

            if not isinstance(info, dict):
                continue

            to = info.get("to")
            conf = info.get("confidence", 0.0)
            reason = info.get("reason", "")

            if not isinstance(to, str):
                continue

            to = to.strip().lower()

            # whitelist canonical names only
            if to not in CANONICAL_ALIAS_VOCAB:
                continue

            # prevent collisions: two columns mapping to same target
            if to in used_targets:
                continue

            try:
                conf = float(conf)
            except Exception:
                conf = 0.0

            conf = float(max(0.0, min(1.0, conf)))

            safe_aliases[str(raw_col)] = {
                "to": str(to),
                "confidence": conf,
                "reason": _safe_str(reason)[:200],
            }
            used_targets.add(to)

        return {
            "status": str(parsed.get("status", "success")),
            "aliases": safe_aliases,
        }

    except Exception as e:
        return {
            "status": "failed", 
            "error": f"LLM call failed with retry: {str(e)}", 
            "aliases": {}
        }

# =========================================================
# SAFETY POST-PROCESSING (IMPORTANT)
# =========================================================

def apply_llm_schema_safety(schema: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Makes sure LLM schema cannot damage your pipeline.
    """

    if not isinstance(schema, dict):
        return schema

    cols = schema.get("columns", {})
    if not isinstance(cols, dict):
        schema["columns"] = {}
        return schema

    safe_cols = {}

    for col, info in cols.items():
        if col not in df.columns:
            continue

        if not isinstance(info, dict):
            continue

        sem_type = str(info.get("semantic_type", "TEXT")).upper().strip()
        conf = info.get("confidence", 0.5)

        try:
            conf = float(conf)
        except Exception:
            conf = 0.5

        conf = max(0.0, min(1.0, conf))

        protect = bool(info.get("protect_from_conversion", False))

        # hard ID rule
        if sem_type in {"ID"}:
            protect = True

        rules = info.get("suggested_rules", {})
        if not isinstance(rules, dict):
            rules = {}

        pattern = rules.get("pattern_regex")
        if pattern is not None:
            pattern = str(pattern).strip()

            if len(pattern) > 250:
                pattern = None

            if pattern:
                try:
                    re.compile(pattern)
                except Exception:
                    pattern = None

        def _safe_num(x):
            if x is None:
                return None
            try:
                return float(x)
            except Exception:
                return None

        safe_rules = {
            "pattern_regex": pattern if pattern else None,
            "min": _safe_num(rules.get("min")),
            "max": _safe_num(rules.get("max")),
            "integer": rules.get("integer") if rules.get("integer") in [True, False, None] else None
        }

        safe_cols[col] = {
            "semantic_type": sem_type,
            "confidence": conf,
            "reason": _safe_str(info.get("reason", ""))[:250],
            "protect_from_conversion": protect,
            "suggested_rules": safe_rules
        }

    schema["columns"] = safe_cols

    if "dataset_level" not in schema or not isinstance(schema.get("dataset_level"), dict):
        schema["dataset_level"] = {}

    mode = str(schema["dataset_level"].get("recommended_cleaning_mode", "balanced")).lower()
    if mode not in {"conservative", "balanced", "aggressive"}:
        mode = "balanced"

    schema["dataset_level"]["recommended_cleaning_mode"] = mode

    warnings_list = schema["dataset_level"].get("warnings", [])
    if not isinstance(warnings_list, list):
        warnings_list = []

    schema["dataset_level"]["warnings"] = [str(w)[:200] for w in warnings_list[:10]]

    return schema

# =========================================================
# OPTIONAL: Convert schema into pipeline config
# =========================================================

def schema_to_pipeline_overrides(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts schema output into config overrides your pipeline can use.
    """

    out = {
        "force_id_protection": [],
        "semantic_hints": {},
        "pattern_rules": {},
    }

    cols = schema.get("columns", {})
    if not isinstance(cols, dict):
        return out

    for col, info in cols.items():
        sem_type = str(info.get("semantic_type", "TEXT")).upper().strip()
        protect = bool(info.get("protect_from_conversion", False))

        if protect:
            out["force_id_protection"].append(col)

        out["semantic_hints"][col] = sem_type

        rules = info.get("suggested_rules", {})
        if isinstance(rules, dict):
            pattern = rules.get("pattern_regex")
            if pattern:
                out["pattern_rules"][col] = pattern

    return out

# =========================================================
# SIMPLE USAGE EXAMPLE
# =========================================================
if __name__ == "__main__":
    df = pd.DataFrame({
        "invoice_id": ["INV-1001", "INV-1002", "INV-1003"],
        "amount_paid": ["₹1,200", "₹2,000", None],
        "customer_email": ["a@gmail.com", "wrong-email", "b@yahoo.com"],
        "created_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
    })

    alias = infer_column_aliases_with_llm(
        df,
        model="deepseek-coder-7b-instruct-v1.5",
        base_url="http://127.0.0.1:1234",
        timeout=30,
    )

    print("\n=== ALIAS OUTPUT ===")
    print(json.dumps(alias, indent=2, ensure_ascii=False))
