import pandas as pd


class DataIntegrityError(Exception):
    """Custom exception for data integrity violations in the pipeline."""
    pass


def _check_df_integrity(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    stage_name: str,
    allow_new_columns: bool = False,
    allow_drop_columns: bool = False,
    enforce_column_order: bool = True,
):
    """
    Production-grade integrity check.

    Enforces:
      - same row count (always)
      - by default: no dropped columns
      - by default: no new columns
      - optionally: enforce column order

    Returns:
      - transformed_df with columns aligned to original order if enforce_column_order=True
    """

    # -------------------------
    # 1) Row count check
    # -------------------------
    if len(original_df) != len(transformed_df):
        raise DataIntegrityError(
            f"[{stage_name}] Row count mismatch: Original had {len(original_df)} rows, "
            f"transformed has {len(transformed_df)} rows. Rows must not be dropped."
        )

    original_cols = list(original_df.columns)
    transformed_cols = list(transformed_df.columns)

    original_set = set(original_cols)
    transformed_set = set(transformed_cols)

    dropped_cols = list(original_set - transformed_set)
    new_cols = list(transformed_set - original_set)

    # -------------------------
    # 2) Dropped columns check
    # -------------------------
    if dropped_cols and not allow_drop_columns:
        raise DataIntegrityError(
            f"[{stage_name}] Dropped columns detected: {dropped_cols}. "
            f"Original columns must be preserved."
        )

    # -------------------------
    # 3) New columns check
    # -------------------------
    if new_cols and not allow_new_columns:
        raise DataIntegrityError(
            f"[{stage_name}] New unexpected columns detected: {new_cols}. "
            f"Pipeline must not add new columns unless explicitly allowed."
        )

    # -------------------------
    # 4) Column order enforcement
    # -------------------------
    if enforce_column_order:
        # Keep original columns first (in original order)
        cols = [c for c in original_cols if c in transformed_df.columns]

        # If new columns exist and are allowed, append them at end
        if allow_new_columns:
            extra = [c for c in transformed_cols if c not in original_set]
            cols.extend(extra)

        transformed_df = transformed_df[cols]

    return transformed_df
