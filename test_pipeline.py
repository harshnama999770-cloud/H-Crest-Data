import pytest
import pandas as pd
from pipeline import load_pipeline
import os


def read_data(path):
    """
    Reads data from a given path, supporting CSV and XLSX formats.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".csv"):
        return pd.read_csv(path, sep=",", encoding="utf-8", skipinitialspace=True)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")


@pytest.fixture(scope="module")
def trained_pipeline():
    """
    Loads the trained pipeline once per module.
    """
    if not os.path.exists("pipeline.pkl"):
        pytest.skip("pipeline.pkl not found. Run the training script first.")
    return load_pipeline("pipeline.pkl")


@pytest.mark.parametrize("file_extension", [".csv", ".xlsx"])
def test_pipeline_transform(trained_pipeline, file_extension):
    """
    Tests the full pipeline transform process on a sample dataset for both CSV and XLSX.
    """

    file_name = f"example_train{file_extension}"

    # Arrange
    df = read_data(file_name)

    original_rows = len(df)
    original_cols = list(df.columns)

    # Act
    clean_df = trained_pipeline.transform(df)

    # Assert 1: output must be a DataFrame
    assert isinstance(clean_df, pd.DataFrame), "Pipeline must return a pandas DataFrame."

    # Assert 2: row count must stay the same (your pipeline policy)
    assert len(clean_df) == original_rows, (
        f"Row count changed! Original={original_rows}, Cleaned={len(clean_df)}"
    )

    # ✅ Assert 3: column count must stay same
    assert len(clean_df.columns) == len(original_cols), (
        f"Column count changed! Original={len(original_cols)}, Cleaned={len(clean_df.columns)}"
    )

    # ✅ Assert 4: Stage0 can rename columns BUT must not create new columns
    # So we only check that cleaned columns are unique and not empty.
    assert len(set(clean_df.columns)) == len(clean_df.columns), (
        "Duplicate columns exist after Stage0 rename + merge."
    )

    assert all(str(c).strip() != "" for c in clean_df.columns), (
        "Empty column name detected after Stage0."
    )

    # Assert 5: report exists
    report = getattr(trained_pipeline, "last_report_", None)
    assert report is not None, "Pipeline must generate last_report_."

    # Assert 6: report contains pipeline status
    assert "status" in report, "Report must contain 'status'."

    # Assert 7: if success, it should be success/rejected only
    assert report["status"] in {"success", "rejected"}, (
        f"Unexpected pipeline status: {report['status']}"
    )

    # ✅ Assert 8: LLM report exists (if enabled)
    assert "llm_report" in report, "Report must contain llm_report."

    print(f"\n--- CLEANED DATA (TEST for {file_name}) ---")
    print(clean_df.head())

    print(f"\n--- REPORT (TEST for {file_name}) ---")
    print(report)
