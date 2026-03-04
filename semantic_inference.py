import pandas as pd
import re

# ==============================================================================
# 🧩 EXTENSIBILITY REGISTRY: SEMANTIC_PATTERNS
# ==============================================================================
# 🔧 Improvement #3 applied: All regex patterns are now precompiled at module 
# load for maximum performance at enterprise scale.
#
# Constraints met:
# 1. Anchored (^...$)
# 2. Whitespace tolerant (\s*)
# 3. Float artifact tolerant for numeric strings like "9876543210.0" ((?:\.0)?)
# ==============================================================================

SEMANTIC_PATTERNS = {
    "email": re.compile(r"^\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\s*$"),
    
    # 10-15 digits, optional country code (+), optional hyphens/spaces, float artifact tolerant
    "phone": re.compile(r"^\s*(?:\+?\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\.0)?\s*$"),
    
    # Indian Pincode: 6 digits, cannot start with 0
    "pincode": re.compile(r"^\s*[1-9][0-9]{5}(?:\.0)?\s*$"),
    
    # Indian GSTIN: 2 Digits + 10 Char PAN + 1 Char/Digit + Z + 1 Char/Digit
    # (?i) handles case insensitivity
    "gstin": re.compile(r"(?i)^\s*[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\s*$"),
    
    # Tolerates YYYY-MM-DD, DD-MM-YYYY, and various separators
    "date": re.compile(r"^\s*(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\s*$"),
    
    # Amounts: Optional currency symbols, allows commas, decimals
    "amount": re.compile(r"^\s*(?:[$€£₹]\s*)?-?\d+(?:,\d{3})*(?:\.\d+)?\s*$"),
    
    # Standard UUID format for generic IDs
    "id": re.compile(r"(?i)^\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\s*$")
}

# Fast-path lookup for Stage A (Name-based Heuristics)
# NOTE: Removed multi-word keys (like 'e-mail' or 'contact_no') to strictly 
# support the highly accurate token-based matching strategy.
NAME_HEURISTICS = {
    "email": ["email"],
    "phone": ["phone", "mobile", "cell", "telephone", "contact"],
    "pincode": ["pin", "pincode", "zip", "zipcode", "postal"],
    "date": ["date", "time", "dob", "created", "updated"],
    "amount": ["amount", "price", "total", "salary", "revenue", "cost", "balance", "fee"],
    "gstin": ["gstin", "gst"],
    "id": ["uuid", "guid"]
}


def infer_semantic_type(col_name: str, series: pd.Series) -> str:
    """
    Infers the semantic meaning of a pandas Series using a two-stage hybrid approach.
    Answers: "What does this column represent?"
    """
    try:
        # ==========================================
        # 🏎️ STAGE A: Name-based Fast Path
        # ==========================================
        # Cheap, deterministic, O(1) checks.
        name_clean = str(col_name).lower().strip()
        
        # Exact or suffix match for 'id' (Prevents matching "width", "hidden")
        if name_clean == "id" or name_clean.endswith("_id") or name_clean.startswith("id_"):
            return "id"
            
        # 🔧 Improvement #1 applied: Tokenized word boundary matching.
        # Splits on underscores, spaces, and hyphens. Converts to set for O(1) lookups.
        # Defeats substring false-positives (e.g. "female" != "email", "phoneix" != "phone").
        tokens = set(re.split(r"[_\s\-]+", name_clean))
        
        for semantic, keywords in NAME_HEURISTICS.items():
            if any(kw in tokens for kw in keywords):
                return semantic
                
        # ==========================================
        # 🔬 STAGE B: Pattern-based Validation
        # ==========================================
        # Only reached if Stage A fails. Takes a small sample.
        
        # Extract up to first 50 valid (non-null) records
        sample_s = series.dropna().head(50)
        
        # If the column is completely empty, we cannot infer pattern
        if len(sample_s) == 0:
            return "unknown"
            
        # Safely cast to string to handle mixed dtypes and prevent regex crashes
        sample_str = sample_s.astype(str)
        
        best_semantic = "unknown"
        best_match_rate = 0.0
        
        # 📏 Recommended default threshold: 0.6 (tolerates up to 40% noise)
        MATCH_THRESHOLD = 0.6
        
        # Execute vectorized matching using precompiled regex
        for semantic, pattern in SEMANTIC_PATTERNS.items():
            
            # .str.match natively accepts pre-compiled re.Pattern objects
            matches = sample_str.str.match(pattern)
            
            # 🔧 Improvement #2 applied: Vectorized boolean mean. 
            # Cleaner, highly idiomatic pandas, and natively prevents integer division bugs.
            match_rate = float(matches.mean())
            
            if match_rate > MATCH_THRESHOLD and match_rate > best_match_rate:
                best_match_rate = match_rate
                best_semantic = semantic
                
                # Early exit if we hit a 100% confidence match
                if best_match_rate == 1.0:
                    break
                    
        return best_semantic

    except Exception:
        # ⚠️ Strict Guardrail: In a data pipeline, profiling failures should not crash the job.
        # Fallback cleanly to 'unknown'.
        return "unknown"