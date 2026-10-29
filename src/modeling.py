"""Shared helpers for Notebook 03 (compare models) and Notebook 04 (fine-tune).

Why this module exists
----------------------
Notebooks 03 and 04 must train every model on *exactly* the same rows and the
same feature definitions as Notebook 02, otherwise their scores are not
comparable. Instead of reading the pre-transformed parquet files (which would
freeze one set of preparation choices), we:

1. Re-create Notebook 02's claim-grouped 70 / 15 / 15 split
   (``GroupShuffleSplit`` on ``ClaimNumber``, ``random_state=42``), so the
   held-out test rows are identical to ``data/processed/test.parquet``.
2. Apply Notebook 02's stateless feature engineering (``engineer_features``:
   anomaly flags, CPI join, date decomposition, domain ratios). It learns
   nothing from data, so it cannot leak.
3. Return Notebook 02's *stateful* preprocessing pipeline (imputation, clipping,
   log transforms, Age bins, encoding, scaling) plus our tunable claim-description
   text block, placed **inside** each model pipeline. It is re-fitted on the
   training folds of every cross-validation split, which keeps CV leak-free and
   lets Notebook 04 tune preparation choices as hyperparameters (Geron,
   Appendix B: "treat your data transformation choices as hyperparameters").
4. Build cross-validation folds with ``GroupKFold`` on ``ClaimNumber`` so that,
   like Notebook 02's split, rows sharing a claim number never sit on both sides.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessing_pipeline, compute_targets, decompose_datetime_features

REPO_ROOT = Path(__file__).resolve().parents[1]
CPI_PATH = REPO_ROOT / "data" / "external" / "fred_medical_cpi_annual.csv"

RANDOM_STATE = 42
TARGET_COL = "UltimateIncurredClaimCost"
INITIAL_COL = "InitialIncurredCalimsCost"  # sic: the CSV header is misspelled
TEXT_COL = "ClaimDescription"
ID_COL = "ClaimNumber"
# Columns that are outcomes (or derived from the outcome) and must never be predictors
OUTCOME_COLS = [TARGET_COL, "ReserveShortfall", "LogDevelopmentRatio"]

# Column groups exactly as defined in Notebook 02 (cells 12 and 14)
LOW_CARD_CAT_COLS = ["Gender", "MaritalStatus", "PartTimeFullTime"]
DISCRETIZE_COLS = ["Age"]
SKEWED_NUMERIC_COLS = [
    INITIAL_COL,
    "InitialCost_Real2005",
    "WeeklyWages",
    "WeeklyWages_Real2005",
    "ApproxHourlyWage",
    "InitialReserveToWageRatio",
    "ReportingLagDays",
]
# Notebook 03 (iteration 2) tests whether these personal attributes add accuracy;
# the shortlisted / final models drop them for fairness and privacy reasons.
SENSITIVE_COLS = ["Gender", "MaritalStatus"]


# ---------------------------------------------------------------------------
# 1. Loading and splitting (mirrors Notebook 02, cell 3, exactly)
# ---------------------------------------------------------------------------
def load_splits(repo_root: Path = REPO_ROOT) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) using Notebook 02's claim-grouped split."""
    raw_df = pd.read_csv(Path(repo_root) / "data" / "raw" / "train.csv")
    data_copy = compute_targets(raw_df.copy())

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss_test.split(data_copy, groups=data_copy[ID_COL]))
    train_val_df = data_copy.iloc[train_val_idx].reset_index(drop=True)
    test_df = data_copy.iloc[test_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[ID_COL]))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def group_folds(X: pd.DataFrame, groups: pd.Series, n_splits: int = 5):
    """5-fold CV that keeps all rows of a ClaimNumber in the same fold."""
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(X, groups=np.asarray(groups)))


_CPI_CACHE: Optional[pd.DataFrame] = None


def _cpi() -> pd.DataFrame:
    global _CPI_CACHE
    if _CPI_CACHE is None:
        _CPI_CACHE = pd.read_csv(CPI_PATH)
    return _CPI_CACHE


def engineer_features(df: pd.DataFrame, external_cpi: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Stateless feature engineering - a copy of ``engineer_features`` in Notebook 02 (cell 11).

    Kept here so Notebooks 03/04 (and the UI) can import it. If Notebook 02's
    version changes, update this function too (or move both into
    ``src/preprocessing.py``).
    """
    external_cpi = _cpi() if external_cpi is None else external_cpi
    out = df.copy()

    # 1. Parse timestamps in UTC and compute exact ReportingLagDays as in Notebook 01
    acc_dt = pd.to_datetime(out["DateTimeOfAccident"], errors="coerce", utc=True)
    rep_dt = pd.to_datetime(out["DateReported"], errors="coerce", utc=True)
    raw_lag = (rep_dt - acc_dt).dt.total_seconds() / 86400.0

    out["negative_lag_flag"] = (raw_lag < 0).astype(int)
    out["lag_over_1yr_flag"] = (raw_lag > 365).astype(int)
    out["ReportingLagDays"] = raw_lag.clip(lower=0.0)
    out["AccidentYear"] = acc_dt.dt.year

    # 2. Join external FRED Medical Care CPI by AccidentYear
    idx = out.index
    out = out.merge(external_cpi, on="AccidentYear", how="left")
    out.index = idx
    out["CPI_Multiplier_to_2005"] = out["CPI_Multiplier_to_2005"].fillna(1.0)
    out["InitialCost_Real2005"] = out[INITIAL_COL] * out["CPI_Multiplier_to_2005"]
    out["WeeklyWages_Real2005"] = out["WeeklyWages"] * out["CPI_Multiplier_to_2005"]

    # 3. Decompose DateTimeOfAccident and DateReported into calendar/time components
    out = decompose_datetime_features(
        out, date_cols=["DateTimeOfAccident", "DateReported"], drop_original=True
    )

    # 4. Sanitize implausible HoursWorkedPerWeek (<= 0 or > 168) flagged in Notebook 01
    invalid_hours = (~out["HoursWorkedPerWeek"].between(0.001, 168.0)) & out["HoursWorkedPerWeek"].notna()
    out["invalid_hours_flag"] = invalid_hours.astype(int)
    out.loc[invalid_hours, "HoursWorkedPerWeek"] = np.nan
    out["age_under_14_flag"] = (out["Age"] < 14).astype(int)

    # 5. Domain ratios and aggregations
    out["ApproxHourlyWage"] = out["WeeklyWages"] / out["HoursWorkedPerWeek"]
    days_clean = out["DaysWorkedPerWeek"].replace(0, np.nan)
    out["HoursPerDay"] = out["HoursWorkedPerWeek"] / days_clean
    out["TotalDependents"] = out["DependentChildren"].fillna(0) + out["DependentsOther"].fillna(0)
    wages_pos = out["WeeklyWages"].where(out["WeeklyWages"] > 0, np.nan)
    out["InitialReserveToWageRatio"] = out[INITIAL_COL] / wages_pos
    return out


def prepare_features(raw_df: pd.DataFrame, drop_sensitive: bool = False) -> pd.DataFrame:
    """Turn raw claim rows (same columns as train.csv, target optional) into model inputs.

    This is also the entry point for scoring *new* claims (e.g. from the UI):
    ``model.predict(prepare_features(new_claims, drop_sensitive=True))``.
    """
    eng = engineer_features(raw_df.drop(columns=[ID_COL] + OUTCOME_COLS, errors="ignore"))
    eng[TEXT_COL] = eng[TEXT_COL].fillna("").astype(str).str.upper()
    if drop_sensitive:
        eng = eng.drop(columns=SENSITIVE_COLS, errors="ignore")
    return eng


def make_xy(df: pd.DataFrame, drop_sensitive: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Engineer features, drop the ID and all outcome columns, and return X, y."""
    return prepare_features(df, drop_sensitive=drop_sensitive), df[TARGET_COL].copy()


def numeric_cols_of(X: pd.DataFrame) -> List[str]:
    """Plain numeric columns = every numeric column not in the skewed list (Notebook 02's rule)."""
    return [
        c for c in X.select_dtypes(include=[np.number]).columns
        if c not in SKEWED_NUMERIC_COLS
    ]


# ---------------------------------------------------------------------------
# 2. Feature pipeline (structured columns + optional claim-description text)
# ---------------------------------------------------------------------------
def build_feature_pipeline(
    X: pd.DataFrame,
    use_text: bool = True,
    text_max_features: int = 2000,
    text_ngram_max: int = 2,
    text_svd_components: Optional[int] = 50,
    **prep_kwargs,
) -> ColumnTransformer:
    """Notebook 02's structured pipeline, plus our tunable claim-description text block.

    Notebook 02 vectorises the text inside its own pipeline (15 SVD topics). We
    switch that off (``text_col=None``) and add our own block so the vocabulary
    size, n-grams and number of topics can be tuned, and so linear models can use
    the sparse TF-IDF words directly; Notebook 02's 15-topic setting is one point
    of that search space.

    ``prep_kwargs`` are passed through to ``build_preprocessing_pipeline`` so that
    Notebook 04 can tune clipping, imputation, skew transform and scaling.
    ``text_svd_components=None`` keeps the sparse TF-IDF matrix (good for
    linear models); an integer compresses it to dense latent topics with
    TruncatedSVD (good for trees, KNN and neural nets).
    """
    prep_defaults = dict(
        clip_outliers=True,
        num_imputer_strategy="median",
        add_missing_indicator=True,
        skew_transform="log1p",
        scaler_type="standard",
        n_bins=5,
        collinearity_threshold=0.95,
    )
    prep_defaults.update(prep_kwargs)

    structured = build_preprocessing_pipeline(
        numeric_cols=numeric_cols_of(X),
        skewed_numeric_cols=[c for c in SKEWED_NUMERIC_COLS if c in X.columns],
        low_card_cat_cols=[c for c in LOW_CARD_CAT_COLS if c in X.columns],
        discretize_cols=[c for c in DISCRETIZE_COLS if c in X.columns],
        text_col=None,
        **prep_defaults,
    )
    structured.set_output(transform="default")  # numpy, so it can sit next to sparse text

    blocks = [("structured", structured, [c for c in X.columns if c != TEXT_COL])]
    if use_text:
        text_steps = [
            ("tfidf", TfidfVectorizer(
                max_features=text_max_features,
                ngram_range=(1, text_ngram_max),
                min_df=3,
                sublinear_tf=True,
            ))
        ]
        if text_svd_components:
            text_steps.append(("svd", TruncatedSVD(n_components=text_svd_components, random_state=RANDOM_STATE)))
        blocks.append(("text", Pipeline(text_steps), TEXT_COL))

    return ColumnTransformer(blocks, remainder="drop", sparse_threshold=0.3)


def feature_names(fitted_features: ColumnTransformer) -> List[str]:
    """Readable names for the columns produced by a *fitted* feature pipeline."""
    names: List[str] = []
    for name, trans, _ in fitted_features.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        if name == "structured":
            # The column step outputs numpy here, so the collinearity filter only knows
            # positions; map the positions it kept back to the column step's real names.
            all_names = list(trans.named_steps["columns"].get_feature_names_out())
            dropped = {int(c) for c in trans.named_steps["collinearity_filter"].cols_to_drop_}
            names += [n for i, n in enumerate(all_names) if i not in dropped]
        elif name == "text":
            if "svd" in trans.named_steps:
                names += [f"text_topic_{i}" for i in range(trans.named_steps["svd"].n_components)]
            else:
                names += [f"word:{w}" for w in trans.named_steps["tfidf"].get_feature_names_out()]
    return names


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
def regression_report(y_true, y_pred) -> Dict[str, float]:
    """Dollar-scale metrics used throughout Notebooks 03 and 04."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedianAE": float(np.median(np.abs(y_true - y_pred))),
        # Positive bias = the model under-predicts on average (under-reserving).
        "MeanBias(actual-pred)": float(np.mean(y_true - y_pred)),
    }


# ---------------------------------------------------------------------------
# 4. Cross-validation helper (same logic as the inline version in Notebook 03)
# ---------------------------------------------------------------------------
def _fit_predict_fold(pipe, X, y, tr_idx, va_idx):
    from sklearn.base import clone
    model = clone(pipe)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    return va_idx, model.predict(X.iloc[va_idx])


def cv_evaluate(pipe, X: pd.DataFrame, y: pd.Series, folds, n_jobs: int = -1):
    """Fit ``pipe`` on each training fold and predict the held-out fold.

    Returns ``(fold_metrics_df, oof_predictions)``. Every preprocessing step inside
    ``pipe`` is re-fitted per fold, so the estimate is leak-free.
    """
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_fold)(pipe, X, y, tr, va) for tr, va in folds
    )
    oof = np.zeros(len(y))
    rows = []
    for va_idx, pred in results:
        oof[va_idx] = pred
        rows.append(regression_report(y.iloc[va_idx], pred))
    return pd.DataFrame(rows), oof
