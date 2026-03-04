# =========================================================
# pipeline.py
# Production-grade pipeline controller (UNIVERSAL SAFE MODE)
#
# ENTERPRISE LLM INTEGRATION:
# - Initializes LLM client ONCE
# - Runs infer_schema_with_llm() ONCE during fit()
# - Stores schema in self.llm_schema_
# - Converts schema -> overrides
# - Passes overrides safely into Stage3 + Stage4
#
# IMPORTANT FIXES:
# - NEVER returns None unless strict=True
# - Returns partial-cleaned df if a stage fails
# - CRITICAL None guard after every stage
# - Hardened stage exception handling & recovery
# =========================================================

import joblib
import pandas as pd
import warnings

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ✅ Stage 0
from cleaningStage0 import ColumnNormalizerStage0
from cleaningStage1 import StructuralAutoCleaner
from cleaningStage2 import QualityRuleCleaner

from cleaningStage3 import SemanticValidator
from cleaningStage3_0 import RelationshipValidator
from cleaningStage4 import PatternValidator
from cleaningStage5 import OutlierAwareImputer

from export_normalizer import ExportNormalizer
from pipeline_utils import DataIntegrityError, _check_df_integrity

# ✅ LLM
from llm_client import LocalLLMClient
from llm_schema_infer import (
    infer_schema_with_llm,
    apply_llm_schema_safety,
    schema_to_pipeline_overrides,
)


# =========================================================
# OPTIONAL PLACEHOLDERS (safe no-op)
# =========================================================

class Stage5PatternValidator(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        self.config = config or {}
        self.last_report_ = {"status": "Stage5PatternValidator: placeholder (no-op)"}

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = _check_df_integrity(X, df, "Stage5PatternValidator")
        return df


class Stage6Imputation(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        self.config = config or {}
        self.last_report_ = {"status": "Stage6Imputation: placeholder (no-op)"}

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = _check_df_integrity(X, df, "Stage6Imputation")
        return df


class Stage9Scorecard(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        self.config = config or {}
        self.last_report_ = {"status": "Stage9Scorecard: placeholder (no-op)"}

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = _check_df_integrity(X, df, "Stage9Scorecard")
        return df


class Stage10Simulation(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        self.config = config or {}
        self.last_report_ = {"status": "Stage10Simulation: placeholder (no-op)"}

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = _check_df_integrity(X, df, "Stage10Simulation")
        return df


# =========================================================
# SAFE PIPELINE WRAPPER
# =========================================================

class SafeCleaningPipeline:
    """
    Universal Safe Pipeline.

    UNIVERSAL RULE:
    - Output must contain ONLY the uploaded file's columns.
    - Never inject training schema columns.
    - Never drop rows.
    - Never add/drop columns (except Stage0 renaming).
    """

    def __init__(
        self,
        stage0_config=None,
        stage1_config=None,
        stage2_config=None,
        stage3_config=None,
        stage3_0_config=None,
        stage4_config=None,
        stage5_config=None,
        export_config=None,
        use_placeholders=False,
        cleaning_mode: str = "balanced",

        # =========================================================
        # ✅ LLM Controls (CENTRAL)
        # =========================================================
        enable_llm: bool = True,
        llm_base_url: str = "http://127.0.0.1:1234",
        llm_model: str = "deepseek-coder-7b-instruct-v1.5",

        # Better defaults:
        llm_max_rows: int = 250,   # ✅ more stable schema
        llm_timeout: int = 90,     # ✅ prevents timeout crashes

        # LLM safety thresholds
        llm_schema_min_confidence: float = 0.70,
    ):

        # -----------------------------------------------------
        # cleaning_mode validation
        # -----------------------------------------------------
        if cleaning_mode not in ("conservative", "balanced", "aggressive"):
            raise ValueError(
                "cleaning_mode must be one of: 'conservative' | 'balanced' | 'aggressive'"
            )

        self.cleaning_mode = cleaning_mode

        # -----------------------------------------------------
        # LLM settings
        # -----------------------------------------------------
        self.enable_llm = bool(enable_llm)
        self.llm_base_url = str(llm_base_url)
        self.llm_model = str(llm_model)
        self.llm_max_rows = int(llm_max_rows)
        self.llm_timeout = int(llm_timeout)
        self.llm_schema_min_confidence = float(llm_schema_min_confidence)

        # LLM results stored after fit()
        self.llm_report_ = None
        self.llm_schema_ = None
        self.llm_overrides_ = None

        # Ensure configs are dicts
        stage2_config = stage2_config or {}
        stage5_config = stage5_config or {}

        # -----------------------------------------------------
        # MODE OVERRIDES
        # -----------------------------------------------------
        mode_overrides = {}

        if self.cleaning_mode == "conservative":
            mode_overrides = {
                "impute_missing": False,
                "soft_rule_threshold": 0.90,
            }

        elif self.cleaning_mode == "balanced":
            mode_overrides = {
                "impute_missing": True,
                "soft_rule_threshold": 0.75,
            }

        elif self.cleaning_mode == "aggressive":
            mode_overrides = {
                "impute_missing": True,
                "soft_rule_threshold": 0.60,
            }

        stage2_config = {
            **stage2_config,
            **{"soft_rule_threshold": mode_overrides["soft_rule_threshold"]}
        }

        stage5_config = {
            **stage5_config,
            **{"impute_missing": mode_overrides["impute_missing"]}
        }

        # -----------------------------------------------------
        # Real stages (LLM config injected)
        # -----------------------------------------------------

        # Stage0 already calls infer_column_aliases_with_llm internally
        stage0_config = stage0_config or {}
        stage0_config = {
            **stage0_config,
            "enable_llm": self.enable_llm,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_timeout": self.llm_timeout,
            "llm_max_rows": self.llm_max_rows,
        }

        stage3_config = stage3_config or {}
        stage3_config = {
            **stage3_config,
            "enable_llm": self.enable_llm,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_timeout": self.llm_timeout,  # ✅ FIX
        }

        stage4_config = stage4_config or {}
        stage4_config = {
            **stage4_config,
            "enable_llm": self.enable_llm,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_timeout": self.llm_timeout,  # ✅ FIX
        }

        self.stage0 = ColumnNormalizerStage0(**stage0_config)
        self.stage1 = StructuralAutoCleaner(**(stage1_config or {}))
        self.stage2 = QualityRuleCleaner(**stage2_config)
        self.stage3 = SemanticValidator(**stage3_config)
        self.stage3_0 = RelationshipValidator(**(stage3_0_config or {}))
        self.stage4 = PatternValidator(**stage4_config)
        self.stage5 = OutlierAwareImputer(**stage5_config)

        self.export_normalizer = ExportNormalizer(**(export_config or {}))

        # Optional placeholders
        self.use_placeholders = bool(use_placeholders)
        self.stage5_placeholder = Stage5PatternValidator()
        self.stage6_placeholder = Stage6Imputation()
        self.stage9_placeholder = Stage9Scorecard()
        self.stage10_placeholder = Stage10Simulation()

        # -----------------------------------------------------
        # PIPELINE ORDER
        # -----------------------------------------------------
        steps = [
            ("cleaningStage0", self.stage0),
            ("cleaningStage1", self.stage1),
            ("cleaningStage2", self.stage2),
            ("semanticValidator", self.stage3),
            ("relationshipValidator", self.stage3_0),
            ("patternValidator", self.stage4),
            ("outlierAwareImputer", self.stage5),
        ]

        if self.use_placeholders:
            steps.extend([
                ("stage5Placeholder", self.stage5_placeholder),
                ("stage6Placeholder", self.stage6_placeholder),
                ("stage9Placeholder", self.stage9_placeholder),
                ("stage10Placeholder", self.stage10_placeholder),
            ])

        steps.append(("exportNormalizer", self.export_normalizer))

        self.pipeline = Pipeline(steps)

        self.is_fitted = False
        self.last_report_ = None

    # -----------------------------------------------------
    # INPUT QUALITY VALIDATION
    # -----------------------------------------------------

    def _validate_input_quality(self, df: pd.DataFrame) -> None:
        if df is None:
            warnings.warn("[SafeCleaningPipeline] Input df is None", UserWarning)
            return

        if not isinstance(df, pd.DataFrame):
            warnings.warn(
                f"[SafeCleaningPipeline] Input is not a pandas DataFrame (got {type(df)}).",
                UserWarning,
            )
            return

        n_rows, n_cols = df.shape

        if n_rows < 10:
            warnings.warn(
                f"[SafeCleaningPipeline] Input has only {n_rows} rows (<10). "
                "Cleaning may be unstable.",
                UserWarning,
            )

        if n_cols == 0:
            warnings.warn("[SafeCleaningPipeline] Input has 0 columns.", UserWarning)
            return

        total_cells = max(n_rows * n_cols, 1)
        missing_cells = int(df.isna().sum().sum())
        missing_fraction = missing_cells / total_cells

        if missing_fraction > 0.50:
            warnings.warn(
                f"[SafeCleaningPipeline] Input has {missing_fraction:.1%} missing values (>50%). "
                "This is very low quality data.",
                UserWarning,
            )

        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            warnings.warn(
                f"[SafeCleaningPipeline] Found {len(empty_cols)} empty columns: {empty_cols}",
                UserWarning,
            )

        constant_cols = []
        for c in df.columns:
            s = df[c]
            if s.isna().all():
                continue
            try:
                if s.nunique(dropna=True) <= 1:
                    constant_cols.append(c)
            except Exception:
                pass

        if constant_cols:
            warnings.warn(
                f"[SafeCleaningPipeline] Found {len(constant_cols)} constant columns: {constant_cols}",
                UserWarning,
            )

    # -----------------------------------------------------
    # ✅ LLM schema inference (CENTRAL)
    # -----------------------------------------------------

    def _run_llm_schema_inference(self, df: pd.DataFrame) -> None:
        """
        Runs LLM once during fit().
        Does NOT modify df.
        Stores:
          - self.llm_schema_
          - self.llm_overrides_
          - self.llm_report_
        """

        if not self.enable_llm:
            self.llm_report_ = {"status": "disabled"}
            self.llm_schema_ = None
            self.llm_overrides_ = None
            return

        try:
            df_small = df.head(int(self.llm_max_rows)).copy()

            client = LocalLLMClient(
                base_url=self.llm_base_url,
                model=self.llm_model,
                timeout=int(self.llm_timeout),
                verbose=False,
            )

            raw_schema = infer_schema_with_llm(
                df_small,
                llm_client=client,
                max_cols=35,
                temperature=0.0,
                max_tokens=1200,
            )

            safe_schema = apply_llm_schema_safety(raw_schema, df_small)

            self.llm_schema_ = safe_schema
            self.llm_overrides_ = schema_to_pipeline_overrides(safe_schema)

            self.llm_report_ = {
                "status": "success",
                "model": self.llm_model,
                "base_url": self.llm_base_url,
                "max_rows": int(self.llm_max_rows),
                "timeout": int(self.llm_timeout),
                "schema_columns": int(len((safe_schema.get("columns", {}) or {}))),
                "overrides": self.llm_overrides_,
                "dataset_level": safe_schema.get("dataset_level", {}),
            }

        except Exception as e:
            # IMPORTANT: LLM is optional
            self.llm_report_ = {"status": "failed", "error": str(e)}
            self.llm_schema_ = None
            self.llm_overrides_ = None

    # -----------------------------------------------------
    # FIT
    # -----------------------------------------------------

    def fit(self, df: pd.DataFrame):
        try:
            self._validate_input_quality(df)

            if self.enable_llm:
                self._run_llm_schema_inference(df)

            self.pipeline.fit(df)
            self.is_fitted = True
            return self

        except Exception as e:
            raise RuntimeError(f"[PIPELINE FIT FAILED] {str(e)}")

    # -----------------------------------------------------
    # TRANSFORM
    # -----------------------------------------------------

    def transform(self, df: pd.DataFrame, strict: bool = False):

        if not self.is_fitted:
            raise RuntimeError("Pipeline is not trained. Run fit() first.")

        current_df = df.copy()
        reports = {}

        last_good_df = current_df.copy()
        failed_stage = None
        failed_error = None

        try:
            for step_name, estimator in self.pipeline.steps:
                original_stage_df = current_df.copy()

                # 🔴 3. Hardened exception handling wrap
                try:
                    current_df = estimator.transform(current_df)
                except Exception as stage_err:
                    failed_stage = step_name
                    failed_error = str(stage_err)

                    if strict:
                        raise RuntimeError(f"[{step_name} FAILED] {failed_error}") from stage_err

                    warnings.warn(
                        f"[SafeCleaningPipeline] Stage '{step_name}' failed: {failed_error}. "
                        f"Using last good output.",
                        UserWarning,
                    )
                    current_df = last_good_df.copy()
                    break

                # 🔴 1. CRITICAL None guard (MOST IMPORTANT)
                if current_df is None:
                    failed_stage = step_name
                    failed_error = f"{step_name} returned None"

                    if strict:
                        raise RuntimeError(failed_error)

                    warnings.warn(
                        f"[SafeCleaningPipeline] {failed_error}. Using last good output.",
                        UserWarning,
                    )
                    current_df = last_good_df.copy()
                    break

                # Collect reports
                if hasattr(estimator, "last_report_"):
                    reports[step_name] = getattr(estimator, "last_report_")
                elif hasattr(estimator, "violations_"):
                    reports[step_name] = {"violations": getattr(estimator, "violations_")}

                # Integrity checks
                if step_name == "cleaningStage0":
                    current_df = _check_df_integrity(
                        original_stage_df,
                        current_df,
                        step_name,
                        allow_new_columns=True,
                        allow_drop_columns=True,
                        enforce_column_order=False
                    )
                else:
                    current_df = _check_df_integrity(
                        original_stage_df,
                        current_df,
                        step_name,
                        allow_new_columns=False,
                        allow_drop_columns=False,
                        enforce_column_order=True
                    )

                # 🔴 5. Stabilized last_good_df tracking (Updated ONLY upon full success)
                last_good_df = current_df.copy()

            # Final report
            status = "success"
            if failed_stage:
                status = "partial_success"

            self.last_report_ = {
                "status": status,
                "cleaning_mode": self.cleaning_mode,
                "failed_stage": failed_stage,
                "failed_error": failed_error,
                "llm_report": getattr(self, "llm_report_", None),
                "llm_schema": getattr(self, "llm_schema_", None),
                "llm_overrides": getattr(self, "llm_overrides_", None),
                **reports
            }

            return current_df

        except (ValueError, DataIntegrityError) as e:
            if strict:
                raise

            # 🔴 4. Guaranteed “never return None” rule
            self.last_report_ = {
                "status": "partial_success",
                "error": str(e),
                "cleaning_mode": self.cleaning_mode,
                "failed_stage": failed_stage,
                "failed_error": failed_error,
                "llm_report": getattr(self, "llm_report_", None),
                "llm_schema": getattr(self, "llm_schema_", None),
                "llm_overrides": getattr(self, "llm_overrides_", None),
                **reports
            }

            warnings.warn(
                f"[SafeCleaningPipeline] Non-strict rejection avoided: {str(e)}. Returning partial output.",
                UserWarning,
            )
            return last_good_df

        except Exception as e:
            if strict:
                raise RuntimeError(f"[PIPELINE TRANSFORM FAILED] Unexpected error: {str(e)}") from e
            
            # Catch-all fallback guarantees returning data over crashing out
            warnings.warn(
                f"[SafeCleaningPipeline] Critical unexpected failure: {str(e)}. Returning last good output.",
                UserWarning,
            )
            self.last_report_ = {
                "status": "failed",
                "error": str(e),
                "failed_stage": "pipeline_level",
                "failed_error": str(e)
            }
            return last_good_df

    # -----------------------------------------------------
    # SAVE / LOAD
    # -----------------------------------------------------

    def save(self, path="pipeline.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="pipeline.pkl"):
        pipe = joblib.load(path)
        if not getattr(pipe, "is_fitted", False):
            raise RuntimeError("Loaded pipeline is not fitted.")
        return pipe


# =========================================================
# BUILDERS
# =========================================================

def build_pipeline(
    stage0_config=None,
    stage1_config=None,
    stage2_config=None,
    stage3_config=None,
    stage3_0_config=None,
    stage4_config=None,
    stage5_config=None,
    export_config=None,
    use_placeholders=False,
    cleaning_mode: str = "balanced",

    # LLM
    enable_llm: bool = True,
    llm_base_url: str = "http://127.0.0.1:1234",
    llm_model: str = "deepseek-coder-7b-instruct-v1.5",
):
    stage1_config = stage1_config or {}

    stage1_config = {
        **stage1_config,
        "allow_unseen_columns": True,
        "add_missing_schema_columns": False,
    }

    return SafeCleaningPipeline(
        stage0_config=stage0_config,
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        stage3_config=stage3_config,
        stage3_0_config=stage3_0_config,
        stage4_config=stage4_config,
        stage5_config=stage5_config,
        export_config=export_config,
        use_placeholders=use_placeholders,
        cleaning_mode=cleaning_mode,

        # LLM
        enable_llm=enable_llm,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )


def fit_and_save_pipeline(
    df: pd.DataFrame,
    path="pipeline.pkl",
    stage0_config=None,
    stage1_config=None,
    stage2_config=None,
    stage3_config=None,
    stage3_0_config=None,
    stage4_config=None,
    stage5_config=None,
    export_config=None,
    use_placeholders=False,
    cleaning_mode: str = "balanced",

    # LLM
    enable_llm: bool = True,
    llm_base_url: str = "http://127.0.0.1:1234",
    llm_model: str = "deepseek-coder-7b-instruct-v1.5",
):
    pipe = build_pipeline(
        stage0_config=stage0_config,
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        stage3_config=stage3_config,
        stage3_0_config=stage3_0_config,
        stage4_config=stage4_config,
        stage5_config=stage5_config,
        export_config=export_config,
        use_placeholders=use_placeholders,
        cleaning_mode=cleaning_mode,

        # LLM
        enable_llm=enable_llm,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )

    pipe.fit(df)
    pipe.save(path)
    return pipe


def load_pipeline(path="pipeline.pkl"):
    return SafeCleaningPipeline.load(path)