# ============================================================
# cleaningStage0.py
# Stage 0: Universal Column Mapping (Production Grade + LLM)
#
# ENTERPRISE FIXES:
# - ✅ Fixed llm_timeout tuple crash (TypeError)
# - ✅ Fixed catastrophic Pandas df.drop() duplicate column data destruction
# - ✅ Replaced deprecated .where() with safe .fillna()
# ============================================================

import re
import pandas as pd
import numpy as np
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from pipeline_utils import _check_df_integrity

# ✅ NEW: LLM alias inference helper
from llm_schema_infer import infer_column_aliases_with_llm


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _normalize_col_name(name: str) -> str:
    """
    Normalize a column name into a canonical safe token.
    """
    if name is None or pd.isna(name) or str(name).strip() == "":
        return ""

    s = str(name).strip().lower()

    # Replace common separators and spaces/tabs with underscore
    s = re.sub(r"\W+", "_", s)

    # Remove any remaining weird chars
    s = re.sub(r"[^a-z0-9_]+", "", s)

    # Remove repeated underscores
    s = re.sub(r"_+", "_", s)

    # Strip leading/trailing underscores
    s = s.strip("_")

    return s


def _build_default_alias_map():
    """
    Alias map:
    raw_column_name -> canonical_column_name

    Only include high-confidence mappings.
    This is UNIVERSAL and safe.
    """
    return {
        # Email
        "mail": "email",
        "emailaddress": "email",
        "email_address": "email",
        "e_mail": "email",
        "e-mail": "email",
        "user_email": "email",

        # Phone
        "mobile": "phone",
        "mobile_no": "phone",
        "mobile_number": "phone",
        "phone_no": "phone",
        "phone_number": "phone",
        "contact_no": "phone",
        "contact_number": "phone",
        "tel": "phone",
        "telephone": "phone",

        # User ID
        "userid": "user_id",
        "user_id": "user_id",
        "customerid": "user_id",
        "customer_id": "user_id",

        # Dates
        "orderdate": "order_date",
        "order_date": "order_date",
        "purchase_date": "order_date",
        "purchased_on": "order_date",

        "deliverydate": "delivery_date",
        "delivery_date": "delivery_date",
        "delivered_on": "delivery_date",

        "returndate": "return_date",
        "return_date": "return_date",

        "createdat": "created_at",
        "created_at": "created_at",
        "createdon": "created_at",

        "updatedat": "updated_at",
        "updated_at": "updated_at",
        "updatedon": "updated_at",

        # DOB / Age
        "dateofbirth": "dob",
        "date_of_birth": "dob",
        "dob": "dob",

        # Money
        "amount": "total_amount",
        "totalamount": "total_amount",
        "total_amount": "total_amount",
        "amount_paid": "amount_paid",
        "paid_amount": "amount_paid",

        "refund": "refund_amount",
        "refundamount": "refund_amount",
        "refund_amount": "refund_amount",

        # Quantity
        "quantity": "qty",
        "qty": "qty",
        "qnty": "qty",

        # Country
        "nation": "country",
        "countryname": "country",
        "country_name": "country",

        # Price/Cost
        "unit_price": "price",
        "unitprice": "price",
        "cost": "price",
        "rate": "price",
    }


# ============================================================
# STAGE 0 TRANSFORMER (LLM ALWAYS ON)
# ============================================================

class ColumnNormalizerStage0(BaseEstimator, TransformerMixin):
    """
    Stage 0: Universal Column Mapping

    Responsibilities:
    - Normalize column names
    - Apply alias mapping (static + LLM)
    - Merge duplicates created by normalization
    - Ensure NO new columns are added (except rename semantics)

    ENTERPRISE RULES:
    - LLM is ALWAYS ON
    - LLM suggestions are used only if confidence >= llm_min_confidence
    - If LLM fails, pipeline continues normally
    - LLM is cached per fit() call (so no repeated LLM calls)
    """

    def __init__(
        self,
        alias_map=None,
        enable_aliasing=True,

        # ✅ LLM ALWAYS ON
        enable_llm: bool = True,
        llm_base_url: str = "http://127.0.0.1:1234",
        llm_model: str = "deepseek-coder-7b-instruct-v1.5",
        llm_timeout: int = 90,  # 🔴 FIX 1: Removed tuple crash (10, 60) -> 90
        llm_min_confidence: float = 0.80,
        llm_max_rows: int = 25,
    ):
        self.enable_aliasing = bool(enable_aliasing)
        self.alias_map = alias_map or _build_default_alias_map()

        # LLM config
        self.enable_llm = bool(enable_llm)
        self.llm_base_url = str(llm_base_url)
        self.llm_model = str(llm_model)
        
        # Safely parse timeout
        try:
            self.llm_timeout = int(llm_timeout)
        except Exception:
            self.llm_timeout = 90
            
        self.llm_min_confidence = float(llm_min_confidence)
        self.llm_max_rows = int(llm_max_rows)

        self.last_report_ = {}

        # cache
        self._cached_llm_aliases_ = None

    def fit(self, X, y=None):
        """
        Fit caches LLM suggestions.
        Stage0 itself does not learn.
        """
        self.is_fitted_ = True

        # Reset cache
        self._cached_llm_aliases_ = {}

        if not self.enable_llm:
            return self

        # LLM call should NEVER crash fit
        try:
            llm_out = infer_column_aliases_with_llm(
                X,
                model=self.llm_model,
                base_url=self.llm_base_url,
                max_rows=self.llm_max_rows,
                timeout=self.llm_timeout,
            )

            aliases = llm_out.get("aliases", {})
            if not isinstance(aliases, dict):
                aliases = {}

            self._cached_llm_aliases_ = aliases

        except Exception as e:
            self._cached_llm_aliases_ = {}
            warnings.warn(f"[Stage0 LLM] failed: {str(e)}", UserWarning)

        return self

    def _get_llm_alias_rename_map(self, df: pd.DataFrame) -> dict:
        """
        Converts cached LLM output into rename map:
        raw_col -> canonical_col
        """
        if not self.enable_llm:
            return {}

        if not isinstance(self._cached_llm_aliases_, dict):
            return {}

        rename_map = {}

        for raw_col, info in self._cached_llm_aliases_.items():
            if raw_col not in df.columns:
                continue

            if not isinstance(info, dict):
                continue

            to = info.get("to")
            conf = info.get("confidence", 0.0)

            if not isinstance(to, str):
                continue

            try:
                conf = float(conf)
            except Exception:
                conf = 0.0

            # enterprise: only apply high confidence
            if conf < self.llm_min_confidence:
                continue

            to_norm = _normalize_col_name(to)
            if not to_norm:
                continue

            # Prevent renaming to same name
            if raw_col == to_norm:
                continue

            rename_map[raw_col] = to_norm

        return rename_map

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        original_df = X.copy()
        df = X.copy()

        report = {
            "stage": "Stage0_ColumnNormalizer",
            "renamed_columns": {},
            "merged_columns": [],
            "final_columns": [],
            "llm": {
                "enabled": bool(self.enable_llm),
                "model": self.llm_model,
                "min_confidence": float(self.llm_min_confidence),
                "applied_aliases": {},
                "suggested_aliases": int(len(self._cached_llm_aliases_ or {})),
            }
        }

        # --------------------------------------------
        # 1) Normalize all column names
        # --------------------------------------------
        normalized_map = {c: _normalize_col_name(c) for c in df.columns}

        df = df.rename(columns=normalized_map)
        report["renamed_columns"].update(normalized_map)

        # --------------------------------------------
        # 2) Apply static alias mapping
        # --------------------------------------------
        if self.enable_aliasing:
            alias_rename_map = {}
            for c in list(df.columns):
                if c in self.alias_map:
                    alias_rename_map[c] = self.alias_map[c]

            if alias_rename_map:
                df = df.rename(columns=alias_rename_map)
                report["renamed_columns"].update(alias_rename_map)

        # --------------------------------------------
        # 3) Apply LLM alias mapping (ALWAYS ON)
        # --------------------------------------------
        llm_rename_map = self._get_llm_alias_rename_map(df)

        if llm_rename_map:
            df = df.rename(columns=llm_rename_map)
            report["renamed_columns"].update(llm_rename_map)
            report["llm"]["applied_aliases"] = llm_rename_map

        # --------------------------------------------
        # 4) Merge duplicate columns (after renaming)
        # --------------------------------------------
        cols = list(df.columns)

        # Unique list of duplicate column names (preserving order)
        dup_names = list(dict.fromkeys([c for c in cols if cols.count(c) > 1]))

        # 🔴 FIX 2 & 3: Safe Positional Dropping & Safe Data Merging
        drop_indices = []
        for name in dup_names:
            idxs = [i for i, c in enumerate(df.columns) if c == name]

            base_i = idxs[0]
            for merge_i in idxs[1:]:
                # Safely fill missing values in the base column using the duplicate column
                df.iloc[:, base_i] = df.iloc[:, base_i].fillna(df.iloc[:, merge_i])
                drop_indices.append(merge_i)

            report["merged_columns"].append(name)

        # Drop the merged duplicate columns securely by positional index, NOT by name
        if drop_indices:
            keep_indices = [i for i in range(df.shape[1]) if i not in drop_indices]
            df = df.iloc[:, keep_indices]

        # --------------------------------------------
        # 5) Final integrity check
        # --------------------------------------------
        report["final_columns"] = list(df.columns)

        _check_df_integrity(
            original_df,
            df,
            "UniversalColumnMapper",
            allow_new_columns=True,
            allow_drop_columns=True,
            enforce_column_order=False
        )

        self.last_report_ = report
        return df