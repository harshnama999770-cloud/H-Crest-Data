# H-Crest Data

Enterprise-grade automated data cleaning and data observability pipeline designed to transform messy real-world datasets into reliable machine-learning-ready data.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline_Architecture-F7931E?logo=scikit-learn)
![AI](https://img.shields.io/badge/AI-Semantic_Routing-FF6F00)



## About

H-Crest Data is a modular pipeline designed to automate dataset preparation for machine learning and analytics workflows.

Real-world datasets are rarely clean or structured. They often contain missing values, inconsistent formats, incorrect relationships, and undocumented schemas. Preparing these datasets manually can consume a significant portion of ML development time.

This project explores how a structured pipeline can automatically analyze, clean, validate, and score datasets before they are used by downstream machine learning systems.

The pipeline combines rule-based data engineering techniques with optional LLM-assisted semantic reasoning to improve dataset understanding and validation.



## Motivation

In many machine learning projects, most of the effort is spent on data preparation rather than model training.

While existing tools provide simple cleaning utilities, they often lack deeper capabilities such as schema understanding, safety guards, and dataset quality monitoring.

H-Crest Data explores a pipeline architecture that can:

* automatically understand dataset structure
* apply safe cleaning transformations
* detect logical data inconsistencies
* intelligently handle missing values
* generate dataset quality scores
* optionally use AI for semantic schema inference



## Core Capabilities

### Automated Data Cleaning Pipeline

The system processes datasets through multiple structured stages. Each stage performs a specific transformation or validation task.

This modular architecture allows the system to remain transparent, extensible, and easier to debug.



### Semantic Dataset Understanding

The pipeline attempts to infer the semantic meaning of columns.

Examples include identifying:

* timestamps
* identifiers
* currency values
* contact information
* measurement values

When rule-based logic cannot determine column meaning confidently, an LLM-assisted inference module can be used.



### Intelligent Missing Value Handling

Instead of naive imputation, the system applies a multi-stage strategy:

1. group-based median imputation
2. similarity-based kNN imputation
3. global fallback values

This approach helps preserve statistical consistency in the dataset.



### Data Quality Scorecard

Each processed dataset receives a quality score based on multiple metrics including:

* completeness
* validity
* consistency
* uniqueness
* structural integrity

The score provides a quick overview of whether the dataset is reliable enough for downstream machine learning tasks.



### Safety Guards

The pipeline includes multiple protections to avoid destructive transformations:

* null inflation detection
* rollback of risky cleaning operations
* sparse column protection
* numeric stability checks

These safeguards help prevent accidental corruption of datasets during automated cleaning.



## LLM Integration

The pipeline optionally integrates a local LLM for semantic schema understanding.

Components responsible for LLM integration include:

* llm/llm_client.py
* llm/llm_schema_infer.py
* llm/semantic_inference.py

The model assists with:

* semantic column inference
* dataset schema interpretation
* validation rule suggestions

The system only calls the LLM when rule-based logic cannot confidently determine column meaning.

This hybrid design allows the pipeline to combine deterministic safety with AI-assisted flexibility.



## Pipeline Architecture

The system is implemented as a structured multi-stage pipeline.

| Stage     | File                | Responsibility                |
| --------- | ------------------- | ----------------------------- |
| Stage 0   | cleaningStage0.py   | Column normalization          |
| Stage 1   | cleaningStage1.py   | Structural dataset inspection |
| Stage 2   | cleaningStage2.py   | Data validation rules         |
| Stage 2.5 | cleaningStage2_5.py | Semantic profiling            |
| Stage 3   | cleaningStage3.py   | Semantic validation           |
| Stage 3.0 | cleaningStage3_0.py | Relationship validation       |
| Stage 4   | cleaningStage4.py   | Pattern validation            |
| Stage 5   | cleaningStage5.py   | Missing value imputation      |
| Stage 6   | cleaningStage6.py   | Dataset quality scoring       |

The pipeline controller coordinates these stages and produces the final cleaned dataset.



## Example Usage


import pandas as pd
from pipeline.pipeline import SafeCleaningPipeline

df = pd.read_csv("dataset.csv")

pipeline = SafeCleaningPipeline()

clean_df = pipeline.fit_transform(df)

print(clean_df.head())


## Project Structure

H-Crest-Data
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── app/
│   ├── app.py
│   ├── runner.py
│   └── run_pipeline_once.py
│
├── pipeline/
│   ├── pipeline.py
│   ├── pipeline_utils.py
│   ├── cleaningStage0.py
│   ├── cleaningStage1.py
│   ├── cleaningStage2.py
│   ├── cleaningStage2_5.py
│   ├── cleaningStage3.py
│   ├── cleaningStage3_0.py
│   ├── cleaningStage4.py
│   ├── cleaningStage5.py
│   └── cleaningStage6.py
│
├── intelligence/
│   ├── data_profiler.py
│   ├── data_intelligence.py
│   └── export_normalizer.py
│
├── llm/
│   ├── llm_client.py
│   ├── llm_schema_infer.py
│   └── semantic_inference.py
│
├── memory/
│   ├── learning_memory.py
│   ├── semantic_history.json
│   └── quality_history.json
│
├── tests/
│   ├── test_pipeline.py
│   └── test_llm.py
│
└── models/
    └── pipeline.pkl




## Tech Stack

Programming
Python

Data Processing
Pandas
NumPy

Pipeline Infrastructure
Scikit-Learn Transformers

Additional Components

* Regex validation
* JSON-based learning memory
* Local LLM semantic inference
* Joblib pipeline persistence


## Installation

Clone the repository


git clone https://github.com/harshnama999770-cloud/H-Crest-Data.git
cd H-Crest-Data


Install dependencies


pip install -r requirements.txt


Run the pipeline


python app/run_pipeline_once.py

## Example Data Quality Report

The pipeline produces a structured JSON report describing
dataset health, detected issues, and recommended actions.

Example output is available here:

reports/example_quality_report.json

Example summary:

Dataset Quality Score: 91.18
Health Grade: B
Total Issues Detected: 745

Top Problem Columns:
- discount
- phone
- email


## Learning Goals

This project explores several topics related to machine learning infrastructure:

* automated dataset preparation
* modular pipeline architecture
* semantic data validation
* hybrid rule-based and AI-assisted systems
* dataset observability



## Author

Harsh Nama
Founder — H-Crest Data



## License

MIT License
