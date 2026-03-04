# 🚀 H-Crest Data

### Enterprise-Grade Automated Data Cleaning & Observability Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline_Architecture-F7931E?logo=scikit-learn)
![AI](https://img.shields.io/badge/AI-Semantic_Routing-FF6F00)
![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)



# 🧭 About This Project

**H-Crest Data** is a modular data-cleaning and data-observability pipeline designed to transform messy real-world datasets into reliable machine-learning-ready data.

The project focuses on solving a common problem in ML systems: **raw datasets are rarely clean, structured, or trustworthy**.

Instead of using simple cleaning scripts, this project builds a **structured pipeline inspired by production data engineering systems**, combining:

* structural data validation
* semantic understanding of columns
* intelligent missing-value imputation
* automated quality scoring

The goal is to create a system that **reduces manual dataset preparation while improving reliability and transparency.**

---

# 🎯 Motivation

In real ML workflows, a large portion of time is spent cleaning data.
Most existing tools handle basic cleaning but lack **safety, observability, and semantic understanding**.

This project explores how a pipeline can:

* automatically understand datasets
* safely transform data without corruption
* monitor data quality over time
* prepare reliable inputs for ML pipelines

---

# ✨ Core Capabilities

## 📊 Data Quality Scorecard

The system produces an interpretable **dataset quality score** based on multiple factors.

Metrics include:

* Completeness
* Validity
* Consistency
* Uniqueness
* Pattern compliance

Each dataset receives:

* a **0-100 score**
* an **A–F quality grade**
* recommended improvement actions

This makes dataset quality visible to both engineers and analysts.



## 🤖 Semantic Column Understanding

The pipeline attempts to infer **column meaning and structure**, enabling intelligent processing.

Examples:

* detecting currency columns
* recognizing timestamps
* identifying contact information patterns
* detecting logical column relationships

Ambiguous columns can be routed to a lightweight semantic inference step.



## 🧠 Intelligent Missing Value Imputation

Instead of simple mean/median filling, the system uses a **multi-stage strategy**:

1. Group-based median imputation
2. k-Nearest Neighbor similarity imputation
3. Global median fallback

This approach preserves statistical integrity better than naive imputation.



## 🛡️ Safe Data Transformations

To prevent destructive cleaning operations, the pipeline includes safety checks such as:

* **Null inflation detection**
* automatic rollback of risky transformations
* skipping highly sparse columns
* numeric stability guards

These mechanisms help avoid accidental data corruption.

---

## 🔗 Cross-Column Relationship Validation

Logical relationships between columns can be automatically validated.

Examples include:

* `start_date` must occur before `end_date`
* `min_price` must be lower than `max_price`
* format validation for email or phone columns

When safe, the system can automatically correct detected inconsistencies.

---

# 🏗️ Pipeline Architecture

The project is built as a modular **Scikit-Learn style pipeline** using custom transformers.

This approach provides:

* clear separation of responsibilities
* reusable processing stages
* easier debugging and extension

| Stage      | Component             | Responsibility                            |
| ---------- | --------------------- | ----------------------------------------- |
| Stage 0    | ColumnNormalizer      | Normalize column names and aliases        |
| Stage 1    | StructuralAutoCleaner | Detect column types and structural issues |
| Stage 2    | QualityRuleCleaner    | Apply universal parsing rules             |
| Stage 2.5  | SemanticProfiler      | Build dataset intelligence layer          |
| Stage 3    | SemanticValidator     | Validate column meaning and risk          |
| Stage 3.0  | RelationshipValidator | Cross-column logic checks                 |
| Stage 4    | PatternValidator      | Regex-based validation                    |
| Stage 5    | OutlierAwareImputer   | Missing value handling                    |
| Stage 6    | DataQualityScorecard  | Dataset grading and reporting             |
| Controller | SafeCleaningPipeline  | Orchestrates all stages                   |

---

# 🧪 Example Usage


from pipeline.safe_cleaning_pipeline import SafeCleaningPipeline
import pandas as pd

df = pd.read_csv("data/sample_dirty_dataset.csv")

pipeline = SafeCleaningPipeline()

clean_df = pipeline.fit_transform(df)

print(clean_df.head())


# 📁 Project Structure


H-Crest-Data/
│
├── pipeline/
│   ├── column_normalizer.py
│   ├── structural_auto_cleaner.py
│   ├── quality_rule_cleaner.py
│   ├── semantic_profiler.py
│   ├── semantic_validator.py
│   ├── relationship_validator.py
│   ├── pattern_validator.py
│   ├── outlier_aware_imputer.py
│   ├── data_quality_scorecard.py
│   └── safe_cleaning_pipeline.py
│
├── data/
│   └── sample_dirty_dataset.csv
│
├── run_demo.py
├── requirements.txt
└── README.md

# 💻 Tech Stack

**Programming**

* Python

**Data Processing**

* Pandas
* NumPy

**Pipeline Architecture**

* Scikit-Learn Transformers

**Additional Components**

* Regex validation
* JSON memory storage
* Joblib pipeline persistence


# 🚀 Installation

Clone the repository


git clone https://github.com/YOUR-USERNAME/H-Crest-Data.git
cd H-Crest-Data


Install dependencies


pip install -r requirements.txt


Run the demo


python run_demo.py




# 🎓 Learning Goals

This project explores several concepts important in data engineering and ML infrastructure:

* automated data preparation pipelines
* safe transformation systems
* dataset observability
* semantic data validation
* modular pipeline design



# 👨‍💻 Author

**Harsh Nama**

Founder — H-Crest Data



# 📄 License

MIT License
