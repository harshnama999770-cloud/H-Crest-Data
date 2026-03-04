# ============================================================
# STAGE 1 — Structural Auto Cleaner (Universal + Production Grade)
# FIXED: Universal ID Safe Mode (Never destroy IDs)
# ENTERPRISE FIXES:
# - ✅ Removed Pandas 2.2+ deprecated infer_datetime_format crash
# - ✅ Fixed silent NaN clipping data-destruction bug
# - ✅ Fixed StringDtype coercion crash (np.nan -> pd.NA)
# ============================================================
import warnings
import pandas as pd
import numpy as np
import datetime
import uuid
import json
import hashlib
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from pipeline_utils import DataIntegrityError, _check_df_integrity


class StructuralAutoCleaner(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        datetime_threshold=0.7,
        id_uniqueness_threshold=0.9,   # kept for backward compatibility
        numeric_threshold=0.6,
        outlier_method="iqr",
        outlier_factor=1.5,
        allow_unseen_columns=True,

        # ---- production controls ----
        entity_cols=None,
        numeric_rounding=None,
        duplicate_subset=None,

        # ---- important control ----
        impute_missing=True,

        # ============================================================
        # ✅ Universal Mode Switch
        # ============================================================

        force_training_schema=False,

        # ============================================================
        # ✅ NEW: Universal ID Safe Mode
        # ============================================================

        id_safe_mode=True,
        id_name_keywords=None,
        max_id_sample=5000,

        # ============================================================
        # ✅ NEW: Temporal leakage protection (optional)
        # ============================================================
        date_column: str | None = None,
        future_cutoff_date: str | datetime.date | datetime.datetime | None = None,
    ):

        if not (0 <= datetime_threshold <= 1):
            raise ValueError("datetime_threshold must be between 0 and 1")

        if not (0 <= id_uniqueness_threshold <= 1):
            raise ValueError("id_uniqueness_threshold must be between 0 and 1")

        if not (0 <= numeric_threshold <= 1):
            raise ValueError("numeric_threshold must be between 0 and 1")

        # ✅ FIXED: <= 0
        if outlier_factor <= 0:
            raise ValueError("outlier_factor must be greater than 0")

        self.datetime_threshold = float(datetime_threshold)
        self.id_uniqueness_threshold = float(id_uniqueness_threshold)
        self.numeric_threshold = float(numeric_threshold)
        self.outlier_method = str(outlier_method)
        self.outlier_factor = float(outlier_factor)
        self.allow_unseen_columns = bool(allow_unseen_columns)

        self.entity_cols = entity_cols
        self.numeric_rounding = numeric_rounding
        self.duplicate_subset = duplicate_subset

        self.impute_missing = bool(impute_missing)

        # universal
        self.force_training_schema = bool(force_training_schema)

        # ID safe mode
        self.id_safe_mode = bool(id_safe_mode)
        self.max_id_sample = int(max_id_sample)

        # NEW: leakage protection
        self.date_column = str(date_column).strip() if date_column else None
        self.future_cutoff_date = future_cutoff_date

        if id_name_keywords is None:
            id_name_keywords = [
                "id", "uuid",
                "order",

                # NEW/expanded:
                "invoice", "inv",
                "txn", "transaction",
                "ref", "reference",
                "receipt",
                "payment",
                "tracking",
                "shipment",
                "delivery",
                "dispatch",

                "customer", "user",
                "session", "token",

                # NEW:
                "bill", "billing",
                "purchase",
                "sale",
                "booking",
            ]
        self.id_name_keywords = [str(x).strip().lower() for x in id_name_keywords]

    # ============================================================
    # INTERNAL: ID-LIKE DETECTION (UNIVERSAL)
    # ============================================================

    def _is_id_like(self, col_name, s: pd.Series) -> bool:
        """
        Universal heuristic:
        Detect ID columns even when:
        - numeric looking
        - duplicated
        - mixed types
        - messy
        """

        if not self.id_safe_mode:
            return False

        col_l = str(col_name).strip().lower()

        # ---- name-based detection ----
        name_is_id = any(k in col_l for k in self.id_name_keywords)

        # If name strongly indicates ID, protect immediately
        if name_is_id:
            return True

        # ---- value-based detection ----
        s_str = s.astype("string")
        s_norm = s_str.str.strip()

        non_null = s_norm.dropna()
        if non_null.empty:
            return False

        sample = non_null.head(self.max_id_sample)

        # key patterns
        has_alpha = sample.str.contains(r"[A-Za-z]", regex=True, na=False).mean()
        has_digit = sample.str.contains(r"\d", regex=True, na=False).mean()
        has_mixed = sample.str.contains(r"(?=.*[A-Za-z])(?=.*\d)", regex=True, na=False).mean()
        has_sep = sample.str.contains(r"[-_/]", regex=True, na=False).mean()

        # uniqueness
        uniqueness = sample.nunique(dropna=True) / max(len(sample), 1)

        # length stats
        lens = sample.str.len()
        avg_len = float(lens.mean()) if len(lens) else 0.0
        std_len = float(lens.std()) if len(lens) else 0.0

        # very long numeric strings: ids, phones, card-like tokens
        long_numeric_ratio = sample.str.fullmatch(r"\d{12,}", na=False).mean()

        # strict numeric-only ratio
        numeric_only_ratio = sample.str.fullmatch(r"\d+", na=False).mean()

        # NEW: 10+ digit IDs (invoice/transaction IDs etc.)
        ten_plus_digits_ratio = sample.str.fullmatch(r"\d{10,}", na=False).mean()

        # ------------------------------------------------------------
        # UNIVERSAL DECISION
        # ------------------------------------------------------------

        if has_mixed > 0.03:
            return True

        if has_sep > 0.08:
            return True

        if (has_digit > 0.80 and avg_len >= 5 and uniqueness > 0.35 and std_len <= 6):
            return True

        if long_numeric_ratio > 0.10:
            return True

        if ten_plus_digits_ratio > 0.10:
            return True

        if numeric_only_ratio > 0.85 and uniqueness > 0.50 and avg_len >= 6:
            return True

        phone_pattern = r"^\d{9,15}$"
        if sample.str.fullmatch(phone_pattern, na=False).mean() > 0.5:
            return True

        credit_card_pattern = r"^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11})$"
        if sample.str.fullmatch(credit_card_pattern, na=False).mean() > 0.1:
            return True

        return False

    # ============================================================
    # INTERNAL: leakage filter
    # ============================================================

    def _filter_future_rows_for_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prevents training on future data.
        Only applied in fit().
        """
        if not self.date_column or not self.future_cutoff_date:
            return df

        if self.date_column not in df.columns:
            # do not crash training
            self.warnings_.append(
                f"[TEMPORAL] date_column '{self.date_column}' not found. No leakage filtering applied."
            )
            return df

        try:
            cutoff = pd.to_datetime(self.future_cutoff_date, errors="coerce")
        except Exception:
            cutoff = pd.NaT

        if pd.isna(cutoff):
            self.warnings_.append(
                f"[TEMPORAL] future_cutoff_date '{self.future_cutoff_date}' invalid. No leakage filtering applied."
            )
            return df

        # 🔴 FIX 1: Removed deprecated infer_datetime_format
        dt = pd.to_datetime(df[self.date_column], errors="coerce")
        m_keep = dt.notna() & (dt <= cutoff)

        before = int(len(df))
        after = int(m_keep.sum())

        if after == 0:
            self.warnings_.append(
                f"[TEMPORAL] All rows removed by cutoff filter. Training will proceed on full data to avoid crash."
            )
            return df

        if after < before:
            self.warnings_.append(
                f"[TEMPORAL] Leakage protection: fit() filtered {before - after} future rows "
                f"using {self.date_column} <= {str(cutoff.date())}."
            )

        return df.loc[m_keep].copy()

    # ============================================================
    # FIT
    # ============================================================

    def fit(self, X, y=None):

        df = X.copy()

        # init warnings before filtering
        self.warnings_ = []

        # ============================================================
        # NEW: Filter training data by cutoff date (prevents leakage)
        # ============================================================
        df = self._filter_future_rows_for_fit(df)

        self.schema_ = list(df.columns)

        self.numeric_cols_ = []
        self.categorical_cols_ = []
        self.datetime_cols_ = []
        self.id_like_cols_ = []

        self.medians_ = {}
        self.modes_ = {}
        self.outlier_models_ = {}

        for col in df.columns:
            s = df[col]

            # 1) UNIVERSAL ID SAFE MODE
            if self._is_id_like(col, s):
                self.id_like_cols_.append(col)
                self.warnings_.append(f"[ID-LIKE] {col}")
                continue

            # ---------------------------
            # Normalize for stable checks
            # ---------------------------
            s_as_string = s.astype("string")
            s_norm = s_as_string.str.strip().str.lower()

            # -------- datetime detection --------
            numeric_ratio = pd.to_numeric(s, errors="coerce").notna().mean()

            # ✅ safer dtype check
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):

                # ✅ speed guard (VERY IMPORTANT for big data)
                s_sample = s_norm.dropna().head(3000)

                if not s_sample.empty:
                    # ✅ warning-free parsing
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        parsed = pd.to_datetime(s_sample, errors="coerce")

                        dt_ratio = parsed.notna().mean()

                        if dt_ratio > self.datetime_threshold and numeric_ratio < 0.9:
                            self.datetime_cols_.append(col)
                            continue

            # -------- numeric detection --------
            coerced = pd.to_numeric(s, errors="coerce")
            ratio = coerced.notna().mean()

            if ratio > self.numeric_threshold:
                self.numeric_cols_.append(col)
            else:
                self.categorical_cols_.append(col)

        # ---------- statistics ----------
        for col in self.numeric_cols_:
            s = pd.to_numeric(df[col], errors="coerce")
            self.medians_[col] = s.median()

            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low = q1 - self.outlier_factor * iqr
            high = q3 + self.outlier_factor * iqr

            # 🔴 FIX 2: Guard against NaN outlier limits (Prevents silent data destruction in transform)
            if pd.notna(low) and pd.notna(high):
                self.outlier_models_[col] = (low, high)

        # categorical modes
        for col in self.categorical_cols_:
            mode_series = df[col].dropna().mode()
            self.modes_[col] = (
                mode_series.iloc[0] if not mode_series.empty else "unknown"
            )

        profile = {
            "schema": self.schema_,
            "numeric": self.numeric_cols_,
            "categorical": self.categorical_cols_,
            "datetime": self.datetime_cols_,
            "id_like": self.id_like_cols_,
        }

        self.schema_version_ = f"v1_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        raw = json.dumps(profile, sort_keys=True, default=str).encode()
        self.schema_hash_ = hashlib.sha256(raw).hexdigest()

        self.profile_ = profile
        return self

    # ============================================================
    # TRANSFORM
    # ============================================================

    def transform(self, X):

        original_input_df = X.copy()
        df = X.copy()
        report = {}

        # ---- backward compatibility ----
        if not hasattr(self, "duplicate_subset"):
            self.duplicate_subset = None
        if not hasattr(self, "numeric_rounding"):
            self.numeric_rounding = None
        if not hasattr(self, "entity_cols"):
            self.entity_cols = None
        if not hasattr(self, "impute_missing"):
            self.impute_missing = True
        if not hasattr(self, "force_training_schema"):
            self.force_training_schema = False
        if not hasattr(self, "id_safe_mode"):
            self.id_safe_mode = True

        # ============================================================
        # 1) Universal schema logic
        # ============================================================

        unseen_cols_in_X = [c for c in df.columns if c not in self.schema_]

        if unseen_cols_in_X and not self.allow_unseen_columns:
            raise DataIntegrityError(
                f"[{self.__class__.__name__}] Input DataFrame contains new columns "
                f"not present during fitting and allow_unseen_columns=False: "
                f"{unseen_cols_in_X}"
            )

        # If strict mode, add missing training columns
        if self.force_training_schema:
            for c in self.schema_:
                if c not in df.columns:
                    df[c] = np.nan

        # Preserve original upload order ONLY
        final_columns_order = list(original_input_df.columns)

        if self.force_training_schema:
            for c in self.schema_:
                if c not in final_columns_order and c in df.columns:
                    final_columns_order.append(c)

        df = df[final_columns_order]

        # ============================================================
        # 2) Type enforcement (SAFE)
        # ============================================================

        # never touch ID-like columns
        id_protected = set(self.id_like_cols_)

        for col in self.numeric_cols_:
            if col in df.columns:
                if col in id_protected:
                    continue
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in self.datetime_cols_:
            if col in df.columns:
                if col in id_protected:
                    continue
                # 🔴 FIX 1: Removed deprecated infer_datetime_format
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # ============================================================
        # 3) Categorical normalization (SAFE)
        # ============================================================

        for col in self.categorical_cols_:
            if col not in df.columns:
                continue

            # NEVER normalize IDs
            if col in id_protected:
                continue

            nunique = df[col].nunique(dropna=True)
            s = df[col].astype("string").str.strip()

            if nunique <= 200:
                s = s.str.lower()

            # 🔴 FIX 3: Replaced np.nan with pd.NA to prevent StringDtype crashes
            df[col] = s.replace(r"^\s*$", pd.NA, regex=True)

        # ============================================================
        # 4) Missing handling (SAFE)
        # ============================================================

        missing_report = {}

        for col in df.columns:

            # never impute IDs
            if col in id_protected:
                continue

            miss = int(df[col].isnull().sum())
            if miss == 0:
                continue

            if not self.impute_missing:
                missing_report[col] = {
                    "count": miss,
                    "percent": round(miss / len(df) * 100, 2),
                    "strategy": "kept_as_nan"
                }
                continue

            if col in self.numeric_cols_ and col in self.medians_:
                df[col] = df[col].fillna(self.medians_[col])
                if self.numeric_rounding is not None:
                    df[col] = df[col].round(self.numeric_rounding)
                strat = "median"

            elif col in self.categorical_cols_ and col in self.modes_:
                df[col] = df[col].fillna(self.modes_[col])
                strat = "mode"

            elif col in self.datetime_cols_:
                strat = "datetime_skipped"

            else:
                strat = "skipped_unseen_or_unprofiled"

            missing_report[col] = {
                "count": miss,
                "percent": round(miss / len(df) * 100, 2),
                "strategy": strat
            }

        report["missing"] = missing_report

        # ============================================================
        # 5) Outlier clipping (only for numeric cols that exist)
        # ============================================================

        outlier_report = {}

        for col, (low, high) in self.outlier_models_.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # 🔴 FIX 2: Final safeguard to ensure invalid models never clip data
                if pd.isna(low) or pd.isna(high):
                    continue
                    
                mask = (df[col] < low) | (df[col] > high)
                count = int(mask.sum())

                if count > 0:
                    # ✅ NEW: store original values before clipping
                    original_vals = df.loc[mask, col].copy()

                    # clip
                    df[col] = df[col].clip(low, high)

                    # ✅ NEW: first 5 examples
                    examples = []
                    try:
                        for idx in original_vals.head(5).index:
                            examples.append({
                                "row": int(idx) if str(idx).isdigit() else idx,
                                "original": float(original_vals.loc[idx]),
                                "clipped_to": float(df.loc[idx, col]),
                            })
                    except Exception:
                        examples = []

                    pct = (count / max(len(df), 1)) * 100.0
                    severity = "high" if (count / max(len(df), 1)) > 0.10 else "medium"

                    outlier_report[col] = {
                        "count": count,
                        "percent": round(pct, 3),
                        "severity": severity,
                        "clip_range": [float(low), float(high)],
                        "examples_first_5": examples,
                    }

        report["outliers"] = outlier_report

        report["schema"] = self.profile_
        report["schema_version"] = self.schema_version_
        report["schema_hash"] = self.schema_hash_
        report["id_like_detected"] = list(self.id_like_cols_)

        # NEW: include warnings (temporal filtering etc.)
        report["warnings"] = list(getattr(self, "warnings_", []))

        # NEW: include leakage settings
        report["temporal"] = {
            "date_column": self.date_column,
            "future_cutoff_date": str(self.future_cutoff_date) if self.future_cutoff_date else None,
        }

        self.last_report_ = report

        # ============================================================
        # 6) Integrity check
        # ============================================================

        _check_df_integrity(
            original_input_df,
            df,
            "StructuralAutoCleaner",
            allow_new_columns=False,
            allow_drop_columns=False
        )

        return df


# ============================================================
# PERSISTENCE HELPERS
# ============================================================

def fit_cleaner(df, path="stage1_cleaner.pkl"):
    cleaner = StructuralAutoCleaner()
    cleaner.fit(df)
    joblib.dump(cleaner, path)
    return cleaner


def transform_cleaner(df, path="stage1_cleaner.pkl"):
    cleaner = joblib.load(path)
    clean_df = cleaner.transform(df)
    return clean_df, cleaner.last_report_