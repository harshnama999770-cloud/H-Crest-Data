# =========================================================
# export_normalizer.py
# Final export-safe normalizer (SAFE MODE)
# =========================================================

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from pipeline_utils import _check_df_integrity


class ExportNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizes final DataFrame types before exporting.

    SAFE DESIGN:
    - never rejects because of missing columns
    - never adds columns
    - never drops columns
    - only converts datetime -> string
    - keeps numeric columns numeric

    Purpose:
    - prevent datetime nanosecond corruption in CSV/XLSX export
    """

    def __init__(
        self,
        datetime_format="%Y-%m-%d %H:%M:%S",
        date_columns=None,

        # NEW: safety thresholds
        min_parse_ratio=0.80,
        max_sample=400,
    ):
        self.datetime_format = datetime_format
        self.date_columns = date_columns

        self.min_parse_ratio = float(min_parse_ratio)
        self.max_sample = int(max_sample)

        self.original_columns = None

        # runtime
        self.original_dtypes_ = None
        self.last_report_ = {}

    def fit(self, X, y=None):
        self.original_columns = list(X.columns)
        self.original_dtypes_ = {c: str(X[c].dtype) for c in X.columns}
        self.last_report_ = {}
        return self

    # =========================================================
    # Better date detection
    # =========================================================

    def _looks_like_date_string(self, s: pd.Series) -> bool:
        """
        More robust guard to prevent numeric columns being parsed as datetime.

        Rules:
        - Must contain separators (- / : .) OR month words
        - Reject if mostly digits-only (ex: "20240101" or "12345")
        - Reject if average length is too short
        """
        x = s.dropna().astype("string").str.strip().str.lower()
        if x.empty:
            return False

        x = x.head(self.max_sample)

        # if mostly numeric-only -> don't guess
        digits_only = x.str.fullmatch(r"\d+", na=False).mean()
        if digits_only >= 0.85:
            return False

        # separators / month names
        has_sep = x.str.contains(r"[-/:\.]", regex=True).mean()

        has_month = x.str.contains(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b",
            regex=True
        ).mean()

        # too short strings rarely dates
        avg_len = float(x.str.len().mean()) if len(x) else 0.0
        if avg_len < 6:
            return False

        return (has_sep >= 0.35) or (has_month >= 0.15)

    def _datetime_parse_ratio(self, s: pd.Series):
        """
        Returns:
          (parsed_series, ratio, tz_detected)
        """
        x = s.copy()

        # sample for speed
        if len(x) > self.max_sample:
            x = x.head(self.max_sample)

        tz_detected = False

        # detect timezone-like patterns (strings)
        try:
            x_str = x.dropna().astype("string").str.strip()
            if not x_str.empty:
                tz_detected = (
                    x_str.str.contains(r"Z$", regex=True).mean() > 0.05
                    or x_str.str.contains(r"[+-]\d{2}:?\d{2}$", regex=True).mean() > 0.05
                    or x_str.str.contains(r"\bUTC\b", regex=True).mean() > 0.05
                )
        except Exception:
            tz_detected = False

        parsed = pd.to_datetime(s, errors="coerce", cache=True)

        non_null = int(s.notna().sum())
        if non_null == 0:
            return parsed, 0.0, tz_detected

        ratio = float(parsed.notna().sum() / non_null)
        return parsed, ratio, tz_detected

    # =========================================================
    # Transform
    # =========================================================

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.original_columns is None:
            raise RuntimeError("ExportNormalizer has not been fitted. Call fit() first.")

        original_df = X.copy()
        df = X.copy()

        warnings = []
        conversion_report = {}

        # ----------------------------------------
        # 1) Detect datetime columns safely
        # ----------------------------------------
        cols_to_process_as_datetime = []

        if self.date_columns:
            # explicit user config
            cols_to_process_as_datetime = [c for c in self.date_columns if c in df.columns]
        else:
            for col in df.columns:

                # already datetime dtype
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    cols_to_process_as_datetime.append(col)
                    continue

                # only attempt on object/string
                if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):

                    if not self._looks_like_date_string(df[col]):
                        continue

                    _, ratio, _ = self._datetime_parse_ratio(df[col])

                    # Only accept if parse success is strong
                    if ratio >= self.min_parse_ratio:
                        cols_to_process_as_datetime.append(col)

        # ----------------------------------------
        # 2) Convert datetime columns to safe string
        # + VALIDATE conversion success
        # ----------------------------------------
        for col in list(dict.fromkeys(cols_to_process_as_datetime)):
            if col not in df.columns:
                continue

            before_non_null = int(df[col].notna().sum())

            # detect tz info BEFORE conversion
            tz_loss_warning = False
            tz_detected = False

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # pandas tz-aware dtype?
                try:
                    tz_detected = getattr(df[col].dt, "tz", None) is not None
                except Exception:
                    tz_detected = False
            else:
                _, _, tz_detected = self._datetime_parse_ratio(df[col])

            parsed = pd.to_datetime(df[col], errors="coerce", cache=True)

            # timezone conversion check
            # (pandas often drops tz silently when formatting)
            try:
                if getattr(parsed.dt, "tz", None) is not None:
                    tz_detected = True
            except Exception:
                pass

            out = pd.Series(pd.NA, index=df.index, dtype="string")
            m = parsed.notna()

            # Convert safely
            try:
                # If tz-aware, convert to naive (but warn)
                try:
                    if getattr(parsed.dt, "tz", None) is not None:
                        tz_loss_warning = True
                        parsed = parsed.dt.tz_convert(None)
                except Exception:
                    pass

                out.loc[m] = parsed.loc[m].dt.strftime(self.datetime_format)
            except Exception as e:
                warnings.append(f"[EXPORT] datetime conversion failed for column '{col}': {str(e)}")
                continue

            after_non_null = int(out.notna().sum())

            # Data loss check
            loss = before_non_null - after_non_null
            loss_ratio = float(loss / max(before_non_null, 1)) if before_non_null > 0 else 0.0

            conversion_report[col] = {
                "converted": True,
                "before_non_null": before_non_null,
                "after_non_null": after_non_null,
                "loss_count": int(loss),
                "loss_ratio": float(round(loss_ratio, 6)),
                "timezone_detected": bool(tz_detected),
                "timezone_dropped": bool(tz_loss_warning),
                "format": str(self.datetime_format),
            }

            if loss > 0:
                warnings.append(
                    f"[EXPORT] Column '{col}' lost {loss} values during datetime parsing "
                    f"({loss_ratio:.1%}). Values became NA."
                )

            if tz_detected:
                warnings.append(
                    f"[EXPORT] Column '{col}' contains timezone information. "
                    f"Export formatting may drop timezone."
                )

            if tz_loss_warning:
                warnings.append(
                    f"[EXPORT] Column '{col}' was timezone-aware and was converted to timezone-naive "
                    f"for export formatting."
                )

            # finally set
            df[col] = out

        # ----------------------------------------
        # 3) Ensure numeric columns stay numeric
        # ----------------------------------------
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ----------------------------------------
        # 4) Export column order stability
        # ----------------------------------------
        final_cols = [c for c in self.original_columns if c in df.columns]
        for c in df.columns:
            if c not in final_cols:
                final_cols.append(c)

        df = df[final_cols]

        # ----------------------------------------
        # 5) Metadata preservation (report only)
        # ----------------------------------------
        # NOTE: We cannot attach comments to DataFrame directly.
        # But we store original dtypes in last_report_ so your exporter can write:
        #   # dtype: col=...
        preserved_dtypes = dict(self.original_dtypes_ or {})

        # ----------------------------------------
        # 6) Integrity check (NO schema restrictions)
        # ----------------------------------------
        _check_df_integrity(
            original_df,
            df,
            "ExportNormalizer",
            allow_new_columns=True,
            allow_drop_columns=True,
            enforce_column_order=False
        )

        # ----------------------------------------
        # 7) Final report
        # ----------------------------------------
        self.last_report_ = {
            "export_normalizer": "success",
            "datetime_format": str(self.datetime_format),
            "min_parse_ratio": float(self.min_parse_ratio),
            "date_columns_override": list(self.date_columns) if self.date_columns else None,
            "converted_datetime_columns": list(conversion_report.keys()),
            "datetime_conversion_report": conversion_report,
            "warnings": warnings,
            "original_dtypes": preserved_dtypes,
        }

        return df
