# INSY 674 group project: proof of value study

## Team members

| Name | Role | GitHub ID |
|---|---|---|
| Nebiyou Timotewos | Problem framing and business context | `@nebiyoutimotewos` |
| Pranjal Singh | Data exploration and profiling | `@pranjallll` |
| Zacharie Houle | Data preparation and feature engineering | `@coredium98` |
| Joel Tomy | Model comparison, fine-tuning, ensembles, and evaluation | `@joeltomy34` |

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
│   ├── preprocessing.py             # Reusable scikit-learn Pipeline and transformers
│   └── modeling.py                  # Shared split, features, metrics and CV helpers for Notebooks 03-04
└── reports/
    ├── figures/                     # Plots exported for the README and slides
    └── presentation/                # Final 10-minute presentation deck
```

## 1. Frame the problem and look at the big picture

### Business objective and decision context

When a workers’ compensation claim is first reported, a claims adjuster sets an initial reserve called `InitialIncurredCalimsCost` (spelled with that typo in the source CSV). This is the amount the insurer expects the claim may eventually cost. The challenge is that this estimate is made early, before the full outcome of the claim is known. Adjusters must rely on limited information, experience, and professional judgment. As a result, some claims may be over-reserved, while others may be significantly under-reserved.

Our dataset includes both the original `InitialIncurredCalimsCost` and the final `UltimateIncurredClaimCost`. This allows us to measure how accurate the initial reserve was. Our objective is to build a machine-learning decision-support system that helps identify when the initial reserve is likely to be inaccurate. Our central research question is: Can machine learning use information available when a workers’ compensation claim is first reported to identify and reduce errors in the initial reserves set by claims adjusters?

The goal is not to replace the claims adjuster. Instead, the model acts as a second layer of review by identifying patterns in historical claims that may be difficult to detect consistently. If successful, the system could help insurers improve reserve accuracy, identify claims that need closer review, and make better decisions earlier in the claims process.

### Current workaround and baseline
The current process relies on the claims adjuster’s initial estimate, recorded in the dataset as `InitialIncurredCalimsCost`. When a claim is first reported, the adjuster reviews the information available at that stage and sets an initial reserve based on the details of the claim and their professional judgment.

Since this estimate is made before the claim has fully developed, it may differ from the amount the claim ultimately costs. Our dataset also includes `UltimateIncurredClaimCost`, which gives us the final outcome and allows us to measure how accurate the original reserve was.

For this project, `InitialIncurredCalimsCost` will serve as the main baseline. This gives us a practical comparison because we can evaluate whether the machine-learning model improves on the estimate already being made by the human adjuster. The model will be considered useful if it can reduce prediction error compared with this baseline on unseen claims.

### Problem framing and performance metric

The project is a supervised regression problem because we’re using historical claims with known final outcomes to predict a continuous value. It’s supervised because the model has a known target to learn from, and regression makes more sense than classification because we’re trying to estimate how large the reserving error is, not sort claims into categories. The project uses offline learning because the model is trained on historical claims and doesn’t update every time a new claim is reported. Instead, it could be retrained periodically as more completed claims and final outcomes become available.

Our primary evaluation metric will be Mean Absolute Error (MAE) because it shows, on average, how far the model’s prediction is from the actual outcome. It also gives us a straightforward way to compare the model with the original `InitialIncurredCalimsCost`. We’ll also report Root Mean Squared Error (RMSE) because it puts more weight on larger prediction errors, which matters because a major reserving mistake can have a much greater financial impact than a small one.

## 2. Explore the data

Full profiling and visualizations are in `notebooks/01_explore_data.ipynb`.

### Attribute summary and target variable
The source dataset (`data/raw/train.csv`) contains 36,176 historical workers' compensation claims and 15 original attributes recorded between 1988 and 2005. Every attribute is profiled in `notebooks/01_explore_data.ipynb` by data type, missingness, noise, distribution shape, and task usefulness. Missing values are sparse and restricted to three columns: `WeeklyWages` (56 rows, 0.15%), `HoursWorkedPerWeek` (49 rows, 0.14%), and `MaritalStatus` (22 rows, 0.06%). Data-quality screening identified 75 rows with `HoursWorkedPerWeek > 168`, 16 rows with zero hours worked, 4 rows with `Age` outside the 14 to 100 range, 2 rows with negative reporting delays (`DateReported < DateTimeOfAccident`), and 12,560 rows sharing a repeated `ClaimNumber`. The free-text `ClaimDescription` field has 20,596 distinct uppercase phrases averaging 7 words per claim.

Both cost columns are strictly positive and heavily right-skewed: `InitialIncurredCalimsCost` has a median of \$2,000, a mean of \$7,743.59, and a maximum of \$830,000, whereas the final outcome `UltimateIncurredClaimCost` has a median of \$3,373.86, a mean of \$10,952.82, and a maximum of \$4,027,135.94. From these two columns we define the dollar reserve error `ReserveShortfall = UltimateIncurredClaimCost - InitialIncurredCalimsCost` (median \$82.70, mean +\$3,209.23, ranging from -\$663,152.27 to +\$3,984,135.94, with 54% of claims under-reserved) and the scale-normalized `LogDevelopmentRatio = log(UltimateIncurredClaimCost / InitialIncurredCalimsCost)`. Because `InitialIncurredCalimsCost` is known at First Notice of Loss (FNOL), predicting `UltimateIncurredClaimCost` and subtracting `InitialIncurredCalimsCost` (`predicted shortfall = predicted ultimate - initial reserve`) yields the exact same dollar MAE and RMSE as predicting `ReserveShortfall` directly, while keeping the regression target strictly positive so `log1p` and Tweedie loss functions can be evaluated on the same scale.

### Correlations and external data
Spearman rank correlation and log-log scatter analysis show that `InitialIncurredCalimsCost` has the strongest monotonic relationship with `UltimateIncurredClaimCost`, followed by `WeeklyWages` and `Age`, but substantial dispersion around the diagonal confirms that the initial reserve alone does not capture how claims develop. Reporting lag (`DateReported - DateTimeOfAccident`) is also right-skewed with extreme outliers over one year. These findings motivated five preparation choices: parsing UTC timestamps to extract `ReportingLagDays` and calendar components, applying `log1p` transforms to skewed monetary predictors, computing hourly wage and reserve-to-wage ratios, extracting TF-IDF and SVD features from `ClaimDescription`, and joining annual US Medical Care CPI data (`CPIMEDSL`) by `AccidentYear` to express nominal reserves and wages from 1988 to 2005 in constant 2005 dollars.

## 3. Prepare the data

The automated, leak-free preparation workflow is implemented in `notebooks/02_prepare_data.ipynb` and modularized in `src/preprocessing.py`.

### Data splitting and anti-leakage protocol
Before fitting any imputation, transformation, or scaling parameters, we compute the row-level audit targets (`ReserveShortfall` and `LogDevelopmentRatio`) and split the 36,176 claims into `train.parquet` (70%), `val.parquet` (15%), and `test.parquet` (15%) using `GroupShuffleSplit` on `ClaimNumber`. Grouping by `ClaimNumber` guarantees that the 12,560 rows sharing repeated claim identifiers never cross between training and evaluation sets. `UltimateIncurredClaimCost` and all derived targets are excluded from the feature matrix $X$, while `InitialIncurredCalimsCost` is retained in $X$ because the adjuster's initial reserve is recorded at First Notice of Loss (FNOL).

### Cleaning, feature engineering, and scaling
1. Recording anomalies and outliers: Rather than dropping rows (which would prevent scoring incomplete or extreme claims in production), we flag and sanitize physical recording anomalies identified during exploration (`HoursWorkedPerWeek` outside `(0, 168]` across 91 rows and 2 negative reporting lags) and apply train-only quantile clipping (`QuantileOutlierClipper` at the 1st and 99th percentiles, tunable as a hyperparameter).
2. Missing values: Missing entries in `WeeklyWages` (56 rows), `HoursWorkedPerWeek` (49 rows plus sanitized anomalies), and `MaritalStatus` (22 rows) are paired with boolean missingness indicators (`add_indicator=True`) and imputed using train-only median or `IterativeImputer` strategies.
3. External data enrichment and domain features: We join annual US Medical Care CPI data from FRED (`data/external/fred_medical_cpi_annual.csv`, series `CPIMEDSL`, 1988 to 2005) on `AccidentYear` to express initial reserves and wages in constant 2005 dollars (`InitialCost_Real2005`, `WeeklyWages_Real2005`). We decompose UTC accident and reporting timestamps, engineer `ReportingLagDays`, `ApproxHourlyWage`, `HoursPerDay`, `TotalDependents`, and `InitialReserveToWageRatio`, discretize `Age` via `KBinsDiscretizer`, apply `log1p` transforms to heavy-tailed monetary predictors, and extract 15 latent injury-description features from `ClaimDescription` using `TfidfVectorizer` + `TruncatedSVD`.
4. Scaling and feature selection: Continuous predictors are standardized (`StandardScaler`) and filtered for redundant collinearity ($|r| \ge 0.95$ learned on `train` via `CollinearityDropper`).

## 4. Measure and compare performance

Model shortlisting and error analysis are in `notebooks/03_compare_models.ipynb`. Both modelling notebooks reproduce Notebook 02's claim-grouped 70/15/15 split exactly (verified row by row against `data/processed/`), use Notebook 02's feature engineering and `src/preprocessing.py` pipeline, and share helpers in `src/modeling.py`. All preprocessing sits inside each model's pipeline, so it is re-fitted on every cross-validation fold. CV folds use `GroupKFold` on `ClaimNumber` (no claim number on both sides of a fold). The test split is not touched until the final evaluation.

### Cross-validation results across model families

* **Baseline to beat:** the adjuster's initial reserve, 5-fold CV **MAE $6,161 ± 316**. It under-reserves 54% of claims. A constant prediction does far worse (median: $9,588), and multiplying the reserve by one development factor does not help ($6,309).
* **Iteration 1 (structured features only):** 8 families with near-default settings: Ridge, LinearSVR, KNN, decision tree, RandomForest, LightGBM, XGBoost, MLP (log-cost target). Seven beat the adjuster; best were RandomForest ($5,722 ± 397), LinearSVR ($5,745 ± 413) and LightGBM ($5,757 ± 398). KNN was worse than the adjuster ($6,562).
* **Iteration 2 (feature engineering + selection):** TF-IDF features from `ClaimDescription` improved every strong model (up to −$310); LightGBM ($5,516) and RandomForest ($5,517) reached **~10.5% below the adjuster**. Removing `Gender` and `MaritalStatus` did not hurt (LightGBM $5,516 → $5,504), so they are excluded from all final models.
* **Iteration 3 (loss function):** for LightGBM, an absolute-error (L1) loss on dollars gave the best MAE ($5,189) but predicts only 62% of total dollars; squared error on dollars gets totals right (102%) but is worse than the adjuster per claim. The loss became a tuned hyperparameter in Notebook 04.

### Feature importance and error analysis

* **Most significant variables** (permutation importance on the validation split, every algorithm): the **initial reserve** dominates (shuffling it adds $3,500–7,000 of MAE). Notebook 02's **`InitialReserveToWageRatio`** and **CPI-adjusted wages** form the second tier for linear and forest models, and the **claim description** adds ≈ $720 once included. Words that raise predicted cost: *shoulder, head, slipped disc, fracture, back*; words that lower it: *eye, foreign (body), lacerated, bruised, burn, finger*.
* **Error types:** models beat the adjuster on mid-sized claims, are slightly worse on the very smallest, and, like the adjuster, cannot anticipate rare catastrophic claims (the top 1% of claims cause 28–31% of all absolute error).
* **Error diversity:** LightGBM, XGBoost and RandomForest make nearly identical errors (log-residual correlation 0.95–0.98); LinearSVR and MLP err differently (≈ 0.87–0.89). Averaging LightGBM with LinearSVR already lowers out-of-fold MAE to $5,321.
* **Shortlist:** LightGBM, RandomForest, LinearSVR, MLP (four families, diverse errors). XGBoost was dropped as redundant with LightGBM, Ridge as dominated by LinearSVR.

## 5. Fine-tune the system

Hyperparameter optimization, ensembling, held-out test evaluation and interpretation are in `notebooks/04_fine_tune_system.ipynb`.

### Hyperparameter search and preprocessing tuning

Tuning uses 5-fold claim-grouped CV on train + validation (85% of claims), optimising MAE. The adjuster scores $6,349 on these folds.

| Model | Search method (why) | Default → tuned CV MAE |
|---|---|---|
| LightGBM | Bayesian optimisation, Optuna TPE, 30 trials (many continuous/conditional parameters, ~20 s per trial) | $5,722 → **$5,377** |
| RandomForest | Random search, 6 configurations (robust model, slowest to fit) | $5,717 → $5,674 |
| MLP | Bayesian optimisation, Optuna TPE, 15 trials (log-scaled continuous parameters) | $6,006 → $5,738 |
| LinearSVR | Grid search, 12 configurations (only three discrete choices) | $5,797 → $5,752 |

Data-preparation choices tuned as hyperparameters: loss / target transform, outlier clipping, imputation strategy, scaler, TF-IDF vocabulary size and n-grams, number of SVD text topics. Key findings: the **L1 loss on dollars** won every top LightGBM trial (≈ $5,380 vs ≈ $5,690 for log + L2); a **compact text representation** (single words, 30 topics) generalised best; outlier clipping helped the MLP but not the tree models.

### Ensemble methods and final test set performance

* **Ensembles (CV):** stacking with a median-regression meta-model was best ($5,354 ± 616), statistically tied with tuned LightGBM alone ($5,377). The meta-model weights LightGBM 0.78, LinearSVR 0.14 and MLP 0.10, and ignores RandomForest. Simple-average voting was worse than LightGBM alone ($5,416+). Stacking was selected by the pre-declared lowest-CV-MAE rule.
* **Held-out test set (5,431 claims, same rows as `data/processed/test.parquet`, used once):**

| | MAE | Median AE | RMSE |
|---|---|---|---|
| **Final model** | **$4,960** | **$441** | $24,714 |
| Adjuster initial reserve | $6,065 | $1,026 | $25,488 |

* **−18.2% MAE vs the adjuster: $1,105 saved per claim (95% bootstrap CI $911–$1,333)**; closer to the final cost than the adjuster on **74%** of claims; 45–62% more accurate on claims up to ~$10,000.
* **As a review tool:** reviewing the 2% of claims with the largest predicted shortfall finds a materially under-reserved claim 50% of the time, **6.0× the hit rate of random review**; the top 10% catches 34% of all materially under-reserved claims.
* **Main drivers:** initial reserve, claim description, then Notebook 02's reserve-to-wage ratio (permutation importance ≈ $4,600, ≈ $960 and ≈ $360 of MAE). SHAP shows hand/finger laceration wording lowers predicted cost while back, shoulder, lifting and fall wording raises it.

### Assumptions and system limitations

1. **Claim-grouped random split rather than a time-based split**, while accident year is a predictor: production error on future claims is likely higher. CPI adjustment removes medical-price inflation but not changes in claim practice. *Fix:* validate on the latest accident years; retrain periodically.
2. **The L1 loss predicts the typical outcome (conditional median rather than the portfolio mean)**, so predictions sum to only 62.7% of actual total cost on the test set compared with 71.8% for the adjuster. On a right-skewed distribution where 9 routine back injuries cost \$2,000 and 1 unpredictable surgical complication costs \$100,000, minimizing per-claim MAE predicts \$2,000 so the model is right on 9 out of 10 individual claims (and closer than the adjuster on 74% of test claims), whereas forcing predictions to sum to 100% of total portfolio dollars (`raw + L2`, MAE \$7,804) requires artificially inflating every routine claim and performs worse than the human adjuster. *Fix:* use the model strictly for claim-level triage and audit ranking, while keeping actuarial portfolio methods (chain-ladder development triangles) for aggregate balance-sheet reserves.
3. **Catastrophic claims remain unpredictable at first notice** (top-quintile gain only 10%). *Fix:* a second model at 30–90 days using medical codes, time off work and litigation information.
4. **The model learns to correct current adjusters**; a change in reserving guidelines would break those corrections. *Fix:* monitor the reserve distribution and retrain on drift.
5. **Synthetic competition data** with no jurisdiction, policy or employer information, and US medical CPI assumed as the right deflator; results need a pilot on real closed claims.
6. **Short, noisy free text** (~7 words); wording conventions elsewhere would weaken the text features.
7. **Fairness:** gender and marital status are excluded, but age and wages remain and can act as proxies. *Fix:* audit errors and flag rates by group; keep a human decision-maker.
8. **Complexity (Stacking vs. single LightGBM):** stacking (\$5,354 ± 616 CV MAE) beats the single tuned LightGBM (\$5,377 ± 617) by only \$23, well inside one fold standard deviation, and the meta-model assigns 78% of its weight to LightGBM, 14% to LinearSVR, 10% to MLP, and 0% to RandomForest. Stacking was retained on the test set strictly to honor the pre-declared lowest-CV-MAE selection rule without post-test snooping, but in production a single tuned LightGBM is the preferred deployment choice for lower maintenance and direct SHAP explanations.
9. **Claim maturity and right-censoring on recent accident years:** workers' compensation claims with long medical or legal tails take several years to reach their true `UltimateIncurredClaimCost`. Claims from the latest accident years in the dataset (2003 to 2005) have had less development time before dataset closure, which can introduce right-censoring bias and partly explains why `AccidentYear` ranks fourth in permutation importance (\$156 MAE). *Fix:* restrict training to fully mature accident cohorts or model development snapshots at fixed lags (such as 12, 24, and 36 months).
10. **Symmetric MAE loss vs. asymmetric business risk and observational limits:** MAE penalizes a \$10,000 under-reserve and a \$10,000 over-reserve equally, even though severe under-reserving carries higher solvency and case-management risk. Moreover, flagging an under-reserved claim at FNOL improves accounting visibility but only reduces actual payout if paired with an operational intervention (such as early nurse case management) whose causal effect must be validated in a prospective randomized pilot.

The final model is saved by Notebook 04 to `models/final_model.joblib` (git-ignored; re-create it by running the notebook) with `models/model_card.json`. To score new claims: `model.predict(prepare_features(raw_rows, drop_sensitive=True))` using `src/modeling.py`.

## Setup and reproduction

Create a virtual environment, install dependencies, and run the notebooks in order (`01` through `04`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
