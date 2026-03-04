# ============================================================
# gui.py
# Advanced Data Cleaning Tool (FINAL FIXED VERSION + Stage6)
# ============================================================

import sys
import os
import json
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QTextEdit, QMessageBox, QTableView, QHeaderView,
    QTabWidget, QComboBox
)
from PyQt6.QtCore import Qt, QAbstractTableModel

from pipeline import load_pipeline
from train_pipeline import train_and_save_pipeline

# ✅ NEW: Runner that runs Stage 6 after pipeline
from runner import run_existing_pipeline_with_scorecard


# ============================================================
# PANDAS MODEL
# ============================================================

class PandasModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.columns[col])
        return None


# ============================================================
# HELPERS
# ============================================================

def _safe_deepcopy_jsonable(obj):
    """
    Deep-copy any python dict/list safely by JSON roundtrip.
    Also converts numpy/pandas objects to strings.
    """
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return {"warning": "Report could not be serialized", "raw": str(obj)}


def _align_schema_for_concat(dfs: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], dict]:
    """
    Makes all dataframes have same columns (union of all columns).
    Missing columns filled with NaN.
    Column order = sorted union for stability.

    Returns:
      aligned_dfs, schema_report
    """
    if not dfs:
        return dfs, {}

    all_cols = sorted(set().union(*[set(df.columns) for df in dfs]))

    schema_report = {
        "union_columns_count": int(len(all_cols)),
        "union_columns": list(all_cols),
        "missing_columns_per_file": {},
        "added_nan_cells_per_file": {},
    }

    aligned = []
    for i, df in enumerate(dfs):
        out = df.copy()
        missing = [c for c in all_cols if c not in out.columns]

        schema_report["missing_columns_per_file"][f"file_{i+1}"] = missing
        schema_report["added_nan_cells_per_file"][f"file_{i+1}"] = int(len(missing) * len(out))

        for c in missing:
            out[c] = np.nan

        out = out[all_cols]
        aligned.append(out)

    return aligned, schema_report


def _is_pipeline_fitted(pipe) -> bool:
    """
    Your pipeline object should have:
      - is_fitted (bool)
    But we also support older versions safely.
    """
    try:
        if pipe is None:
            return False
        if hasattr(pipe, "is_fitted"):
            return bool(pipe.is_fitted)
        # fallback: assume fitted if no flag exists
        return True
    except Exception:
        return False


# ============================================================
# MAIN APPLICATION
# ============================================================

class DataCleanerApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Data Cleaning Tool")
        self.resize(1150, 850)

        # Data
        self.df = None
        self.cleaned_df = None

        # Pipeline
        self.pipeline = None
        self.pipeline_path = "pipeline.pkl"

        # File storage
        self.loaded_files = {}  # filename -> df
        self.cleaned_files_single_mode = {}  # filename -> cleaned_df

        # Training info (for warning)
        self.last_training_filenames = set()

        # UI
        main_layout = QVBoxLayout(self)

        self.status = QLabel("Status: Initializing...")
        self.preview_note = QLabel("Preview: Showing first 100 rows only.")
        self.preview_note.setStyleSheet("color: gray;")

        self.btn_upload = QPushButton("Upload Data")
        self.btn_train = QPushButton("Train Pipeline")
        self.btn_clean = QPushButton("Run Cleaning")
        self.btn_save = QPushButton("Save Cleaned Data")

        self.mode_selector = QComboBox()
        self.mode_selector.addItem("Single Mode")
        self.mode_selector.addItem("Combined Mode")
        self.mode_selector.setCurrentText("Single Mode")
        self.processing_mode = "single"
        self.mode_selector.currentIndexChanged.connect(self._update_processing_mode)

        top = QHBoxLayout()
        controls = QVBoxLayout()
        controls.addWidget(self.status)
        controls.addWidget(self.preview_note)
        controls.addSpacing(10)
        controls.addWidget(self.btn_upload)
        controls.addWidget(self.btn_train)
        controls.addWidget(self.btn_clean)
        controls.addWidget(self.btn_save)
        controls.addSpacing(10)
        controls.addWidget(QLabel("Processing Mode:"))
        controls.addWidget(self.mode_selector)
        top.addLayout(controls)

        self.tabs = QTabWidget()
        self.original = QTableView()
        self.cleaned = QTableView()
        self.tabs.addTab(self.original, "Original")
        self.tabs.addTab(self.cleaned, "Cleaned")
        top.addWidget(self.tabs, 3)

        self.report = QTextEdit()
        self.report.setReadOnly(True)

        main_layout.addLayout(top)
        main_layout.addWidget(QLabel("Report"))
        main_layout.addWidget(self.report)

        # Events
        self.btn_upload.clicked.connect(self.load_data)
        self.btn_train.clicked.connect(self.train_pipeline)
        self.btn_clean.clicked.connect(self.run_cleaning)
        self.btn_save.clicked.connect(self.save_data)

        # Load pipeline at startup
        self._try_load_pipeline()

    # --------------------------------------------------------

    def _update_processing_mode(self, index):
        if index == 0:
            self.processing_mode = "single"
            self.status.setText("Status: Processing mode set to Single.")
        else:
            self.processing_mode = "combined"
            self.status.setText("Status: Processing mode set to Combined.")

    # --------------------------------------------------------

    def _try_load_pipeline(self):
        try:
            pipe = load_pipeline(self.pipeline_path)

            # IMPORTANT: ensure pipeline is fitted
            if not _is_pipeline_fitted(pipe):
                self.pipeline = None
                self.status.setText("Status: Pipeline found but NOT fitted. Train again.")
                return

            self.pipeline = pipe
            self.status.setText("Status: Trained pipeline loaded.")

        except Exception:
            self.pipeline = None
            self.status.setText("Status: No trained pipeline. Please train first.")

    # --------------------------------------------------------

    def load_data(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Data Files",
            "",
            "Data Files (*.csv *.xlsx)"
        )
        if not paths:
            return

        self.loaded_files = {}
        self.cleaned_files_single_mode = {}
        self.cleaned_df = None
        self.report.clear()

        for path in paths:
            filename = os.path.basename(path)
            try:
                if path.endswith(".csv"):
                    df = pd.read_csv(path)
                elif path.endswith(".xlsx"):
                    df = pd.read_excel(path)
                else:
                    QMessageBox.warning(self, "Unsupported Format", f"Skipping {filename}: Unsupported format.")
                    continue

                self.loaded_files[filename] = df

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load {filename}: {e}")

        if not self.loaded_files:
            self.status.setText("Status: No files loaded.")
            return

        first_file_name = list(self.loaded_files.keys())[0]
        self.df = self.loaded_files[first_file_name]

        self.original.setModel(PandasModel(self.df.head(100)))
        self.original.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.cleaned.setModel(None)

        self.status.setText(f"Loaded {len(self.loaded_files)} file(s). First preview: {first_file_name}")

    # --------------------------------------------------------

    def train_pipeline(self):
        if not self.loaded_files:
            QMessageBox.warning(self, "Warning", "Upload at least one file to train.")
            return

        self.last_training_filenames = set(self.loaded_files.keys())

        schema_report = None

        if len(self.loaded_files) >= 2:
            dfs = list(self.loaded_files.values())
            dfs, schema_report = _align_schema_for_concat(dfs)
            train_df = pd.concat(dfs, ignore_index=True)
            train_msg = "Trained using MERGED data from all uploaded files."
        else:
            first_file_name = list(self.loaded_files.keys())[0]
            train_df = self.loaded_files[first_file_name]
            train_msg = f"Trained using first file: {first_file_name}"

        try:
            self.pipeline = train_and_save_pipeline(train_df, path=self.pipeline_path)

            if not _is_pipeline_fitted(self.pipeline):
                raise RuntimeError("Training finished but pipeline is not marked fitted.")

            msg = (
                "Pipeline trained successfully.\n\n"
                f"{train_msg}\n\n"
                "IMPORTANT:\n"
                "For best results, upload a DIFFERENT file to clean.\n"
                "If you clean the same training file, results may look unrealistically perfect."
            )

            if schema_report:
                msg += "\n\nSchema merge note:\n"
                msg += f"- Union columns: {schema_report.get('union_columns_count', 0)}\n"
                msg += "- Missing columns were filled with NaN in some files."

            QMessageBox.information(self, "Training Complete", msg)
            self.status.setText("Status: Pipeline trained.")

            training_report = {
                "status": "trained",
                "trained_on_files": list(self.last_training_filenames),
                "training_rows": int(len(train_df)),
                "training_cols": int(len(train_df.columns)),
                "schema_alignment": schema_report,
            }
            self.report.setText(json.dumps(_safe_deepcopy_jsonable(training_report), indent=2, default=str))

        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"Pipeline training failed: {e}")

    # --------------------------------------------------------

    def run_cleaning(self):
        if not self.loaded_files:
            QMessageBox.warning(self, "Warning", "Upload data files to clean.")
            return

        if self.pipeline is None:
            self._try_load_pipeline()
            if self.pipeline is None:
                QMessageBox.critical(self, "Error", "No trained pipeline available.")
                return

        if not _is_pipeline_fitted(self.pipeline):
            QMessageBox.critical(self, "Error", "Pipeline is not fitted. Train pipeline first.")
            return

        # Warning: cleaning same file as training
        if self.last_training_filenames:
            cleaning_files = set(self.loaded_files.keys())
            overlap = cleaning_files.intersection(self.last_training_filenames)
            if overlap:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "You are cleaning the SAME file(s) used for training:\n\n"
                    f"{', '.join(sorted(overlap))}\n\n"
                    "This may produce overly perfect results.\n"
                    "For best accuracy, train on one dataset and clean a different dataset."
                )

        # =========================
        # SINGLE MODE
        # =========================
        if self.processing_mode == "single":
            self.cleaned_files_single_mode = {}
            cleaning_reports = {}

            for filename, df in self.loaded_files.items():
                try:
                    # ✅ NEW: pipeline cleaning + stage6 scorecard
                    result = run_existing_pipeline_with_scorecard(self.pipeline, df.copy())

                    cleaned_df = result["cleaned_df"]
                    issues = result["issues"]
                    scorecard = result["scorecard"]
                    reports = result["reports"]

                    if cleaned_df is None:
                        err_msg = reports.get("error", "Rejected by pipeline (unknown reason).")
                        raise ValueError(f"Cleaning failed for '{filename}': {err_msg}")

                    if not isinstance(cleaned_df, pd.DataFrame):
                        raise TypeError(
                            f"Cleaning returned type {type(cleaned_df)} for file '{filename}', expected DataFrame."
                        )

                    self.cleaned_files_single_mode[filename] = cleaned_df

                    cleaning_reports[filename] = {
                        "issue_count": int(len(issues)),
                        "scorecard": _safe_deepcopy_jsonable(scorecard),
                        "reports": _safe_deepcopy_jsonable(reports),
                    }

                except Exception as e:
                    cleaning_reports[filename] = {"error": str(e)}
                    QMessageBox.critical(self, "Cleaning Error", f"Failed to clean {filename}: {e}")

            if not self.cleaned_files_single_mode:
                self.status.setText("Status: No files were cleaned successfully.")
                self.cleaned.setModel(None)
                self.report.clear()
                return

            first_cleaned = list(self.cleaned_files_single_mode.keys())[0]
            self.cleaned_df = self.cleaned_files_single_mode[first_cleaned]

            self.cleaned.setModel(PandasModel(self.cleaned_df.head(100)))
            self.cleaned.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

            final_report = {
                "mode": "single",
                "files_cleaned": list(self.cleaned_files_single_mode.keys()),
                "reports_per_file": cleaning_reports,
                "note": "Preview shows first 100 rows only."
            }

            self.report.setText(json.dumps(_safe_deepcopy_jsonable(final_report), indent=2, default=str))
            self.status.setText("Status: Cleaning completed (Single Mode).")

        # =========================
        # COMBINED MODE
        # =========================
        elif self.processing_mode == "combined":
            if len(self.loaded_files) < 2:
                QMessageBox.warning(self, "Warning", "Combined mode requires at least two files.")
                return

            try:
                dfs = list(self.loaded_files.values())

                dfs, schema_report = _align_schema_for_concat(dfs)

                merged_df = pd.concat(dfs, ignore_index=True)

                # ✅ NEW: pipeline cleaning + stage6 scorecard
                result = run_existing_pipeline_with_scorecard(self.pipeline, merged_df.copy())

                cleaned_df = result["cleaned_df"]
                issues = result["issues"]
                scorecard = result["scorecard"]
                reports = result["reports"]

                if cleaned_df is None:
                    err_msg = reports.get("error", "Rejected by pipeline (unknown reason).")
                    raise ValueError(f"Cleaning failed in combined mode: {err_msg}")

                if not isinstance(cleaned_df, pd.DataFrame):
                    raise TypeError(f"Cleaning returned type {type(cleaned_df)} in combined mode, expected DataFrame.")

                self.cleaned_df = cleaned_df

                self.cleaned.setModel(PandasModel(self.cleaned_df.head(100)))
                self.cleaned.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

                combined_report = {
                    "mode": "combined",
                    "files_merged": list(self.loaded_files.keys()),
                    "schema_alignment": schema_report,
                    "issue_count": int(len(issues)),
                    "scorecard": _safe_deepcopy_jsonable(scorecard),
                    "reports": _safe_deepcopy_jsonable(reports),
                    "note": (
                        "Combined mode uses UNION of all columns.\n"
                        "Missing columns in some files were filled with NaN.\n"
                        "Preview shows first 100 rows only."
                    )
                }

                self.report.setText(json.dumps(_safe_deepcopy_jsonable(combined_report), indent=2, default=str))
                self.status.setText("Status: Cleaning completed (Combined Mode).")

            except Exception as e:
                QMessageBox.critical(self, "Cleaning Error", f"Failed to clean combined data: {e}")

    # --------------------------------------------------------

    def save_data(self):
        # =========================
        # SINGLE MODE SAVE
        # =========================
        if self.processing_mode == "single":
            if not self.cleaned_files_single_mode:
                QMessageBox.warning(self, "Warning", "No cleaned files to save in single mode.")
                return

            for original_filename, cleaned_df in self.cleaned_files_single_mode.items():
                default_save_name = f"cleaned_{original_filename}"

                path, _ = QFileDialog.getSaveFileName(
                    self,
                    f"Save Cleaned File: {original_filename}",
                    default_save_name,
                    "CSV Files (*.csv);;Excel Files (*.xlsx)"
                )
                if not path:
                    continue

                try:
                    if path.endswith(".csv"):
                        cleaned_df.to_csv(path, index=False)
                    elif path.endswith(".xlsx"):
                        cleaned_df.to_excel(path, index=False)
                    else:
                        cleaned_df.to_csv(path, index=False)

                    QMessageBox.information(self, "Saved", f"Saved {original_filename} to:\n{path}")

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save {original_filename}: {e}")

        # =========================
        # COMBINED MODE SAVE
        # =========================
        else:
            if self.cleaned_df is None:
                QMessageBox.warning(self, "Warning", "Nothing to save in combined mode.")
                return

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Combined Data File",
                "cleaned_combined.csv",
                "CSV Files (*.csv);;Excel Files (*.xlsx)"
            )
            if not path:
                return

            try:
                if path.endswith(".csv"):
                    self.cleaned_df.to_csv(path, index=False)
                elif path.endswith(".xlsx"):
                    self.cleaned_df.to_excel(path, index=False)
                else:
                    self.cleaned_df.to_csv(path, index=False)

                QMessageBox.information(self, "Saved", f"Saved combined data to:\n{path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save combined data: {e}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataCleanerApp()
    window.show()
    sys.exit(app.exec())
