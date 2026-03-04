import os
import json
import argparse
import pandas as pd
from pipeline import load_pipeline


def read_data(path: str) -> pd.DataFrame:
    """
    Reads data from CSV or XLSX.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".csv"):
        return pd.read_csv(path, sep=",", encoding="utf-8", skipinitialspace=True)

    if path.endswith(".xlsx"):
        return pd.read_excel(path)

    raise ValueError("Unsupported file format. Please provide .csv or .xlsx")


def save_report(report: dict, out_path: str):
    """
    Saves report as JSON safely.
    """
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Failed to save report JSON: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run SafeCleaningPipeline once on any file.")
    parser.add_argument("input_file", type=str, help="Input CSV/XLSX file path")
    parser.add_argument("--pipeline", type=str, default="pipeline.pkl", help="Pipeline .pkl path")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output folder")
    parser.add_argument("--strict", action="store_true", help="Strict mode (raise errors)")
    args = parser.parse_args()

    input_file = args.input_file
    pipeline_path = args.pipeline
    outdir = args.outdir
    strict = bool(args.strict)

    if not os.path.exists(pipeline_path):
        print(f"❌ Pipeline not found: {pipeline_path}")
        print("Train it first:")
        print("python train_pipeline.py example_train.csv --output pipeline.pkl")
        return

    os.makedirs(outdir, exist_ok=True)

    # -----------------------------------------
    # Load pipeline
    # -----------------------------------------
    print(f"Loading pipeline from: {pipeline_path}")
    pipe = load_pipeline(pipeline_path)

    # -----------------------------------------
    # Read input file
    # -----------------------------------------
    print(f"Reading input file: {input_file}")
    df = read_data(input_file)

    # SAFE: strip column whitespace
    df.columns = df.columns.astype(str).str.strip()

    print(f"Input rows={len(df)}, cols={len(df.columns)}")

    # -----------------------------------------
    # Run transform
    # -----------------------------------------
    print("\nRunning pipeline transform...")
    cleaned = pipe.transform(df, strict=strict)

    report = getattr(pipe, "last_report_", None)

    # -----------------------------------------
    # Save outputs
    # -----------------------------------------
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    report_path = os.path.join(outdir, f"{base_name}_report.json")
    csv_path = os.path.join(outdir, f"{base_name}_cleaned.csv")
    xlsx_path = os.path.join(outdir, f"{base_name}_cleaned.xlsx")

    save_report(report if isinstance(report, dict) else {}, report_path)

    if cleaned is None:
        print("\n❌ Pipeline rejected this file (cleaned output = None).")
        print(f"Report saved at: {report_path}")
        return

    cleaned.to_csv(csv_path, index=False, encoding="utf-8")
    cleaned.to_excel(xlsx_path, index=False)

    print("\n✅ Cleaning completed successfully!")
    print(f"Cleaned CSV saved:  {csv_path}")
    print(f"Cleaned XLSX saved: {xlsx_path}")
    print(f"Report JSON saved:  {report_path}")

    # -----------------------------------------
    # Quick preview
    # -----------------------------------------
    print("\n--- CLEANED DATA PREVIEW ---")
    print(cleaned.head(10))

    print("\n--- PIPELINE STATUS ---")
    if isinstance(report, dict):
        print("status:", report.get("status"))
        print("cleaning_mode:", report.get("cleaning_mode"))
        if "llm_report" in report:
            llm = report.get("llm_report") or {}
            if isinstance(llm, dict):
                print("llm_report.status:", llm.get("status"))


if __name__ == "__main__":
    main()
