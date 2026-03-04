import argparse
import pandas as pd
from pipeline import SafeCleaningPipeline
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


def train_and_save_pipeline(
    df,
    stage0_config=None,
    stage1_config=None,
    stage2_config=None,
    stage3_config=None,
    stage4_config=None,
    stage5_config=None,
    export_config=None,
    cleaning_mode="balanced",
    enable_llm=True,
    llm_model="deepseek-coder-7b-instruct-v1.5",
    llm_timeout=60,
    llm_max_rows=30,
    path="pipeline.pkl",
):
    """
    Initializes, trains, and saves the SafeCleaningPipeline.
    """

    print("Initializing pipeline...")

    pipe = SafeCleaningPipeline(
        stage0_config=stage0_config,
        stage1_config={**(stage1_config or {}), "allow_unseen_columns": True},
        stage2_config=stage2_config,
        stage3_config=stage3_config,
        stage4_config=stage4_config,
        stage5_config=stage5_config,
        export_config=export_config,

        # ✅ enterprise controls
        cleaning_mode=cleaning_mode,

        # ✅ LLM controls
        enable_llm=enable_llm,
        llm_model=llm_model,
        llm_timeout=llm_timeout,
        llm_max_rows=llm_max_rows,
    )

    print("Fitting pipeline on training data...")
    pipe.fit(df)

    print(f"Saving trained pipeline to {path}...")
    pipe.save(path)

    print("✅ Pipeline training complete and saved successfully.")

    # ✅ Print LLM status
    rep = getattr(pipe, "llm_report_", None)
    if rep is None:
        print("\n[LLM] No llm_report_ found.")
    else:
        print("\n[LLM REPORT]")
        print(rep)

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save the SafeCleaningPipeline.")

    parser.add_argument("input_file", type=str, help="Path to training file (CSV or XLSX).")
    parser.add_argument("--output", type=str, default="pipeline.pkl", help="Output pipeline path")

    # ✅ NEW: cleaning mode
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
        help="Cleaning intensity mode",
    )

    # ✅ NEW: LLM settings
    parser.add_argument("--enable-llm", action="store_true", help="Enable LLM schema inference")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM schema inference")
    parser.add_argument("--llm-model", type=str, default="deepseek-coder-7b-instruct-v1.5")
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--llm-max-rows", type=int, default=30)

    args = parser.parse_args()

    try:
        train_df = read_data(args.input_file)

        # SAFE: strip whitespace in column names only (no row drops)
        train_df.columns = train_df.columns.astype(str).str.strip()

        print(f"Loaded '{args.input_file}' for training.")
        print(f"Rows={len(train_df)}, Cols={len(train_df.columns)}")

        # safest default configs
        s0_config = {}
        s1_config = {}
        s2_config = {}
        s3_config = {}
        s4_config = {}
        s5_config = {}
        export_config = {}

        # Decide LLM enabled
        enable_llm = True
        if args.disable_llm:
            enable_llm = False
        elif args.enable_llm:
            enable_llm = True

        train_and_save_pipeline(
            train_df,
            stage0_config=s0_config,
            stage1_config=s1_config,
            stage2_config=s2_config,
            stage3_config=s3_config,
            stage4_config=s4_config,
            stage5_config=s5_config,
            export_config=export_config,
            cleaning_mode=args.mode,
            enable_llm=enable_llm,
            llm_model=args.llm_model,
            llm_timeout=args.llm_timeout,
            llm_max_rows=args.llm_max_rows,
            path=args.output,
        )

    except Exception as e:
        print(f"❌ Training failed: {e}")
