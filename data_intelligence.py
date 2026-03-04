import pandas as pd
from typing import Dict, Any

# ==============================================================================
# 🔌 IMPORTS
# Assuming the previous two files are named `profiler.py` and `semantic_inference.py`
# Adjust the import paths as necessary for your project structure.
# ==============================================================================
from .profiler import profile_dataframe
from .semantic_inference import infer_semantic_type

def build_data_intelligence(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Orchestrates structural profiling and semantic inference into a single,
    unified intelligence artifact. 
    
    Guarantees:
    - Stateless and deterministic
    - Zero mutation of the input DataFrame
    - 100% JSON-serializable output
    """
    
    # ==========================================
    # 🔬 STEP 1: Structural Profiling
    # ==========================================
    # Computes missingness, cardinality, outliers, and raw dtypes.
    # Guaranteed to return a JSON-safe dictionary.
    profile = profile_dataframe(df)
    
    # ==========================================
    # 🧠 STEP 2: Build Semantic Map
    # ==========================================
    # Infers business meaning (e.g., 'email', 'amount', 'date') per column.
    semantic_types = {}
    
    for col in df.columns:
        # Cast column name to string to guarantee strict JSON serialization
        # (Pandas columns can sometimes be integers or tuples)
        col_str = str(col)
        
        # Pass the extracted series to the inference engine safely
        semantic_types[col_str] = infer_semantic_type(col_str, df[col])
        
    # ==========================================
    # 🧱 STEP 3: Return Unified Contract
    # ==========================================
    # This object is the final payload consumed by the pipeline, UI, or LLMs.
    intelligence_artifact = {
        "profile": profile,
        "semantic_types": semantic_types
    }
    
    return intelligence_artifact