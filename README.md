# INSY 674 group project: proof of value study

## Team members

| Name | Role | GitHub ID |
|---|---|---|
| Zacharie Houle | Data preparation and feature engineering | `@...` |
| Teammate 2 | Data exploration and profiling | `@...` |
| Teammate 3 | Model comparison and error analysis | `@...` |
| Teammate 4 | Fine-tuning, ensembles, and evaluation | `@...` |

## Repository structure

```text
insy674-group-project/
├── README.md                        # Project overview and rubric documentation
├── requirements.txt                 # Python dependencies
├── data/
│   ├── raw/                         # Unmodified source data
│   ├── external/                    # External datasets joined during preparation
│   └── processed/                   # Leak-free splits: train.parquet, val.parquet, test.parquet
├── notebooks/
│   ├── 01_explore_data.ipynb        # Framing, data profiling, distributions, and correlations
│   ├── 02_prepare_data.ipynb        # Train/test split, cleaning, feature engineering, scaling
│   ├── 03_compare_models.ipynb      # Cross-validated baseline and model family comparison
│   └── 04_fine_tune_system.ipynb    # Hyperparameter search, ensembles, and final test evaluation
├── src/
│   ├── __init__.py
│   └── preprocessing.py             # Reusable scikit-learn Pipeline and transformers
└── reports/
    ├── figures/                     # Plots exported for the README and slides
    └── presentation/                # Final 10-minute presentation deck
```

## 1. Frame the problem and look at the big picture

### Business objective and decision context

When a workers’ compensation claim is first reported, a claims adjuster sets an initial reserve called InitialCaseEstimate. This is the amount the insurer expects the claim may eventually cost. The challenge is that this estimate is made early, before the full outcome of the claim is known. Adjusters must rely on limited information, experience, and professional judgment. As a result, some claims may be over-reserved, while others may be significantly under-reserved.

Our dataset includes both the original InitialCaseEstimate and the final UltimateIncurredClaimCost. This allows us to measure how accurate the initial reserve was. Our objective is to build a machine-learning decision-support system that helps identify when the initial reserve is likely to be inaccurate. Our central research question is: Can machine learning use information available when a workers’ compensation claim is first reported to identify and reduce errors in the initial reserves set by claims adjusters?

The goal is not to replace the claims adjuster. Instead, the model acts as a second layer of review by identifying patterns in historical claims that may be difficult to detect consistently. If successful, the system could help insurers improve reserve accuracy, identify claims that need closer review, and make better decisions earlier in the claims process.

### Current workaround and baseline
*(Describe how the organization solves this problem today without machine learning, and define the baseline score to beat.)*

### Problem framing and performance metric
*(Justify the framing: supervised vs. unsupervised, regression vs. classification, offline vs. online learning. Define the primary evaluation metric and explain why it matches the economic cost of prediction errors.)*

## 2. Explore the data

Full profiling and visualizations are in `notebooks/01_explore_data.ipynb`.

### Attribute summary and target variable
*(Summarize attribute types, missing value percentages, noise patterns, and distribution shapes. Identify the target variable and its distribution.)*

### Correlations and external data
*(Summarize the main findings from the correlation analysis and document the external dataset joined to enrich the raw features.)*

## 3. Prepare the data

The automated, leak-free preparation workflow is implemented in `notebooks/02_prepare_data.ipynb` and modularized in `src/preprocessing.py`.

### Data splitting and anti-leakage protocol
The raw dataset is split into `train.parquet`, `val.parquet`, and `test.parquet` before any imputation, encoding, or scaling is fitted. All transformers are fitted strictly on `train` and applied to `val` and `test` through a unified scikit-learn `Pipeline`.

### Cleaning, feature engineering, and scaling
*(Document the rationale for outlier handling, missing value imputation, domain feature engineering, discretization, mathematical transformations for skewed distributions, and feature scaling.)*

## 4. Measure and compare performance

Model shortlisting and error forensics are in `notebooks/03_compare_models.ipynb`.

### Cross-validation results across model families
*(Report N-fold cross-validation mean and standard deviation across distinct model families: linear, tree-based, boosting, and neural networks.)*

### Feature importance and error analysis
*(Summarize the top predictive variables per algorithm, compare the types of errors made by each family, and explain why 3 to 5 specific models were shortlisted.)*

## 5. Fine-tune the system

Hyperparameter optimization, ensembling, and held-out test evaluation are in `notebooks/04_fine_tune_system.ipynb`.

### Hyperparameter search and preprocessing tuning
*(Document the search strategy used, including which data preparation choices were treated as tunable hyperparameters inside cross-validation.)*

### Ensemble methods and final test set performance
*(Compare the best individual models against Voting/Stacking ensembles and report the final generalization performance on the held-out test set against the baseline.)*

### Assumptions and system limitations
*(List specific model assumptions, data limitations, their operational impact, and how they can be mitigated in production.)*

## Setup and reproduction

Create a virtual environment, install dependencies, and run the notebooks in order (`01` through `04`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
