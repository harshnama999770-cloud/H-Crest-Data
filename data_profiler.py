import pandas as pd
import numpy as np
import math
from typing import Dict, Any

def detect_outliers_iqr(series: pd.Series) -> int:
    """
    Step 6 — Outlier detection (IQR)
    Vectorized computation of outliers based on the interquartile range.
    Requires a minimum sample size to avoid statistically insignificant results.
    """
    # Require minimum sample size
    if len(series) < 5:
        return 0
        
    # Calculate percentiles natively
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    
    # Must not crash or over-flag on constant columns (where IQR = 0)
    if iqr == 0.0:
        return 0
        
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Vectorized check
    outliers = (series < lower_bound) | (series > upper_bound)
    return int(outliers.sum())


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    A pure, read-only statistical profiler that computes structural health 
    metrics for a pandas DataFrame safely and deterministically.
    """
    rows, columns = df.shape
    
    result = {
        "rows": int(rows),
        "columns": int(columns),
        "column_profiles": {}
    }
    
    # Tolerate empty dataframe entirely
    if df.empty and columns == 0:
        return result

    # Helper function to sanitize floats for strict JSON serialization
    # (Replaces NaN/Inf with None to conform to native JSON standards)
    def _json_safe_float(val) -> Any:
        if pd.isna(val) or math.isinf(val):
            return None
        return float(val)

    # Process each column in a single O(n) pass
    for col in df.columns:
        # Step 1 — safe series extraction
        s = df[col]
        non_null = s.dropna()
        
        # Step 2 — missing metrics
        missing_count = int(s.isna().sum())
        
        # Safely compute missing_percent, handling 0-row DataFrames to avoid NaN
        raw_missing_percent = float(s.isna().mean() * 100)
        missing_percent = 0.0 if pd.isna(raw_missing_percent) else raw_missing_percent
        
        # Step 3 — cardinality
        unique_count = int(non_null.nunique())
        
        # Guard divide-by-zero and prevent 100% duplicate rate on entirely empty/null columns
        if len(non_null) == 0:
            duplicate_percent = 0.0
        else:
            duplicate_percent = float(1.0 - (unique_count / len(non_null)))
        
        # Step 7 — top values snapshot
        # Extract to dict and enforce native JSON serializable types (str keys, int values)
        raw_top_values = non_null.value_counts().head(5).to_dict()
        top_values = {
            str(k): int(v) for k, v in raw_top_values.items()
        }
        
        # Base dictionary initialization
        col_profile = {
            "dtype_raw": str(s.dtype),
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "unique_count": unique_count,
            "duplicate_percent": duplicate_percent,
            "top_values": top_values
        }
        
        # Step 4 — numeric detection
        # Strictly using the required pandas API to determine numeric processing
        if pd.api.types.is_numeric_dtype(s):
            
            # Step 5 — numeric stats (conditional)
            if len(non_null) > 0:
                col_profile["min"] = _json_safe_float(non_null.min())
                col_profile["max"] = _json_safe_float(non_null.max())
                col_profile["mean"] = _json_safe_float(non_null.mean())
                col_profile["std"] = _json_safe_float(non_null.std())
            else:
                col_profile["min"] = None
                col_profile["max"] = None
                col_profile["mean"] = None
                col_profile["std"] = None
                
            # Step 6 — outlier counting via helper function
            col_profile["outlier_count"] = detect_outliers_iqr(non_null)
            
        # Store profile safely using stringified column names 
        # (Handling cases where dataframe columns might be integers/tuples)
        result["column_profiles"][str(col)] = col_profile
        
    return result