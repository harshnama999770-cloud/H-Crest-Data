import json
from pathlib import Path
from typing import Dict, Any

class CleaningLearningMemory:
    def __init__(self, path: str = "cleaning_memory.json"):
        self.path = Path(path)
        self.memory = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return {}
        return {}

    def save(self):
        try:
            self.path.write_text(json.dumps(self.memory, indent=2))
        except Exception:
            pass

    # -----------------------------
    # key builder (VERY IMPORTANT)
    # -----------------------------
    def build_key(self, col: str, role: str) -> str:
        return f"{col.lower()}|{role}"

    # -----------------------------
    # record feedback
    # -----------------------------
    def record_outcome(
        self,
        col: str,
        role: str,
        confidence: float,
        success: bool,
    ):
        key = self.build_key(col, role)

        stats = self.memory.setdefault(key, {
            "seen": 0,
            "failures": 0,
            "avg_confidence": 0.0,
        })

        stats["seen"] += 1
        if not success:
            stats["failures"] += 1

        # running average
        stats["avg_confidence"] = (
            stats["avg_confidence"] * (stats["seen"] - 1) + confidence
        ) / stats["seen"]

        self.save()

    # -----------------------------
    # get penalty
    # -----------------------------
    def get_penalty(self, col: str, role: str) -> float:
        key = self.build_key(col, role)
        stats = self.memory.get(key)
        if not stats:
            return 0.0

        failure_rate = stats["failures"] / max(stats["seen"], 1)

        # scale penalty
        return min(0.4, failure_rate * 0.5)