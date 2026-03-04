
import joblib
import pandas as pd
import warnings

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from cleaningStage0 import ColumnNormalizerStage0
from cleaningStage1 import StructuralAutoCleaner
from cleaningStage2 import QualityRuleCleaner
from cleaningStage2_5 import SemanticProfiler
from cleaningStage3 import SemanticValidator
from cleaningStage3_0 import RelationshipValidator
from cleaningStage4 import PatternValidator
from cleaningStage5 import OutlierAwareImputer

from export_normalizer import ExportNormalizer
from pipeline_utils import DataIntegrityError, _check_df_integrity


from backend.data_intelligence import build_data_intelligence



class Stage5PatternValidator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return _check_df_integrity(X, X.copy(), "Stage5PatternValidator")

class Stage6Imputation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return _check_df_integrity(X, X.copy(), "Stage6Imputation")

class Stage9Scorecard(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return _check_df_integrity(X, X.copy(), "Stage9Scorecard")

class Stage10Simulation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return _check_df_integrity(X, X.copy(), "Stage10Simulation")


class SafeCleaningPipeline:
    """
    Universal Safe Pipeline integrating Stage 2.5 (SemanticProfiler).

    GUARANTEES:
    - Never silently returns None
    - Stage-level failure isolation
    - Partial recovery supported
    - Strict mode enforcement
    """

    def __init__(
        self,
        stage0_config=None,
        stage1_config=None,
        stage2_config=None,
        stage2_5_config=None,
        stage3_config=None,
        stage3_0_config=None,
        stage4_config=None,
        stage5_config=None,
        export_config=None,
        use_placeholders=False,
        cleaning_mode: str = "balanced",  
    ):
        
        self.cleaning_mode = cleaning_mode

        stage2_config = stage2_config or {}
        stage2_config["cleaning_profile"] = self.cleaning_mode

 
        self.stage0 = ColumnNormalizerStage0(**(stage0_config or {}))
        self.stage1 = StructuralAutoCleaner(**(stage1_config or {}))
        self.stage2 = QualityRuleCleaner(**stage2_config)
        self.stage2_5 = SemanticProfiler(**(stage2_5_config or {}))
        self.stage3 = SemanticValidator(**(stage3_config or {}))
        self.stage3_0 = RelationshipValidator(**(stage3_0_config or {}))
        self.stage4 = PatternValidator(**(stage4_config or {}))
        self.stage5 = OutlierAwareImputer(**(stage5_config or {}))
        self.export_normalizer = ExportNormalizer(**(export_config or {}))

        # Optional placeholders
        self.use_placeholders = bool(use_placeholders)
        self.stage5_placeholder = Stage5PatternValidator()
        self.stage6_placeholder = Stage6Imputation()
        self.stage9_placeholder = Stage9Scorecard()
        self.stage10_placeholder = Stage10Simulation()

        # Pipeline order
        steps = [
            ("cleaningStage0", self.stage0),
            ("cleaningStage1", self.stage1),
            ("cleaningStage2", self.stage2),
            ("semanticProfiler", self.stage2_5),
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


    def fit(self, df: pd.DataFrame):
        try:
            self.pipeline.fit(df)
            self.is_fitted = True
            return self
        except Exception as e:
            raise RuntimeError(f"[PIPELINE FIT FAILED] {str(e)}")

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

                # -----------------------------
                # RUN STAGE
                # -----------------------------
                try:
                    current_df = estimator.transform(current_df)
                except Exception as stage_err:
                    failed_stage = step_name
                    failed_error = str(stage_err)

                    if strict:
                        raise

                    warnings.warn(
                        f"[SafeCleaningPipeline] Stage '{step_name}' failed: {failed_error}. "
                        f"Returning partial cleaned output.",
                        UserWarning,
                    )
                    current_df = last_good_df.copy()
                    break

                # -----------------------------
                # 🚨 NONE GUARD
                # -----------------------------
                if current_df is None:
                    failed_stage = step_name
                    failed_error = f"{step_name} returned None"

                    if strict:
                        raise RuntimeError(
                            f"[SafeCleaningPipeline] {failed_error}"
                        )

                    warnings.warn(
                        f"[SafeCleaningPipeline] Stage '{step_name}' returned None. "
                        f"Reverting to last good output.",
                        UserWarning,
                    )
                    current_df = last_good_df.copy()
                    break

                if step_name == "cleaningStage1":
                    intel = build_data_intelligence(current_df)
                    self.semantic_types_ = intel.get("semantic_types", {})
                    self.profile_before_cleaning_ = intel.get("profile", {})
                    
                    if hasattr(self.stage2, "semantic_types"):
                        self.stage2.semantic_types = self.semantic_types_

                # -----------------------------
                # COLLECT REPORTS
                # -----------------------------
                if hasattr(estimator, "last_report_"):
                    reports[step_name] = getattr(estimator, "last_report_")
                elif hasattr(estimator, "violations_"):
                    reports[step_name] = {
                        "violations": getattr(estimator, "violations_")
                    }

                # -----------------------------
                # INTEGRITY CHECK
                # -----------------------------
                if step_name == "cleaningStage0":
                    current_df = _check_df_integrity(
                        original_stage_df,
                        current_df,
                        step_name,
                        allow_new_columns=True,
                        allow_drop_columns=True,
                        enforce_column_order=False,
                    )
                else:
                    current_df = _check_df_integrity(
                        original_stage_df,
                        current_df,
                        step_name,
                        allow_new_columns=False,
                        allow_drop_columns=False,
                        enforce_column_order=True,
                    )

                # update last good snapshot
                last_good_df = current_df.copy()

            # -----------------------------
            # FINAL REPORT (SUCCESS)
            # -----------------------------
            status = "success" if failed_stage is None else "partial_success"

            self.last_report_ = {
                "status": status,
                "cleaning_mode": self.cleaning_mode,
                "failed_stage": failed_stage,
                "failed_error": failed_error,
                "data_profile_before_cleaning": getattr(self, "profile_before_cleaning_", {}),
                "semantic_types": getattr(self, "semantic_types_", {}),
                **reports,
            }

            return current_df

        except (ValueError, DataIntegrityError) as e:
            if strict:
                raise

            warnings.warn(
                f"[SafeCleaningPipeline] Non-strict rejection avoided: {str(e)}. "
                f"Returning partial output.",
                UserWarning,
            )

            self.last_report_ = {
                "status": "partial_success",
                "cleaning_mode": self.cleaning_mode,
                "error": str(e),
                "data_profile_before_cleaning": getattr(self, "profile_before_cleaning_", {}),
                "semantic_types": getattr(self, "semantic_types_", {}),
                **reports,
            }

            return last_good_df  # ✅ NEVER return None

        except Exception as e:
           
            if strict:
                raise RuntimeError(f"[PIPELINE TRANSFORM FAILED] Unexpected error: {str(e)}") from e
            
            warnings.warn(
                f"[SafeCleaningPipeline] Critical unexpected failure: {str(e)}. Returning last good output.",
                UserWarning,
            )
        
            self.last_report_ = {
                "status": "failed",
                "cleaning_mode": self.cleaning_mode,
                "error": str(e),
                "failed_stage": "pipeline_level",
                "failed_error": str(e),
                "data_profile_before_cleaning": getattr(self, "profile_before_cleaning_", {}),
                "semantic_types": getattr(self, "semantic_types_", {}),
                **reports
            }
            return last_good_df

    # =====================================================
    # SAVE / LOAD
    # =====================================================

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

def build_pipeline(**configs):
    stage1_config = configs.get("stage1_config") or {}
    stage1_config = {
        **stage1_config,
        "allow_unseen_columns": True,
        "add_missing_schema_columns": False,
    }
    configs["stage1_config"] = stage1_config
    return SafeCleaningPipeline(**configs)


def fit_and_save_pipeline(df: pd.DataFrame, path="pipeline.pkl", **configs):
    pipe = build_pipeline(**configs)
    pipe.fit(df)
    pipe.save(path)
    return pipe


def load_pipeline(path="pipeline.pkl"):
    return SafeCleaningPipeline.load(path)