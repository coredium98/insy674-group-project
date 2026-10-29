"""Shared helpers for Notebook 03 (compare models) and Notebook 04 (fine-tune).

Why this module exists
----------------------
Notebooks 03 and 04 must train every model on *exactly* the same rows and the
same feature definitions, otherwise their scores are not comparable. Instead of
reading pre-transformed parquet files (which would freeze one set of
preparation choices), we:

1. Re-create the 70 / 15 / 15 train / val / test split from Notebook 02 with the
   same ``random_state`` so the held-out test rows are identical.
2. Apply the same stateless feature engineering as Notebook 02
   (``engineer_features``) - it learns nothing from data, so it cannot leak.
3. Return a *stateful* preprocessing pipeline (imputation, clipping, scaling,
   text vectorisation) that is placed **inside** each model pipeline. That way
   it is re-fitted on the training folds of every cross-validation split, which
   keeps CV leak-free and lets Notebook 04 tune preparation choices as
   hyperparameters (Geron, Appendix B: "treat your data transformation choices
   as hyperparameters").
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessing_pipeline, decompose_datetime_features

RANDOM_STATE = 42
TARGET_COL = "UltimateIncurredClaimCost"
INITIAL_COL = "InitialIncurredCalimsCost"  # sic: the CSV header is misspelled
TEXT_COL = "ClaimDescription"
ID_COL = "ClaimNumber"

LOW_CARD_CAT_COLS = ["Gender", "MaritalStatus", "PartTimeFullTime"]
# Notebook 03 (iteration 2) shows these two personal attributes add no accuracy,
# so the shortlisted / final models drop them for fairness and privacy reasons.
SENSITIVE_COLS = ["Gender", "MaritalStatus"]
SKEWED_NUMERIC_COLS = [INITIAL_COL, "WeeklyWages", "hourly_wage", "reporting_delay_days"]


# ---------------------------------------------------------------------------
# 1. Loading and splitting (mirrors Notebook 02 exactly)
# ---------------------------------------------------------------------------
def load_splits(repo_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) using Notebook 02's split recipe."""
    raw_df = pd.read_csv(Path(repo_root) / "data" / "raw" / "train.csv")
    train_val_df, test_df = train_test_split(
        raw_df.copy(), test_size=0.15, random_state=RANDOM_STATE
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.15 / 0.85, random_state=RANDOM_STATE
    )
    return train_df, val_df, test_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering - a copy of ``engineer_features`` in Notebook 02.

    Kept here so Notebooks 03/04 can import it. If Notebook 02's version changes,
    update this function too (or move both into ``src/preprocessing.py``).
    """
    out = df.copy()
    out["HoursWorkedPerWeek"] = out["HoursWorkedPerWeek"].clip(lower=0, upper=168)

    acc_dt = pd.to_datetime(out["DateTimeOfAccident"], errors="coerce")
    rep_dt = pd.to_datetime(out["DateReported"], errors="coerce")
    out["reporting_delay_days"] = (rep_dt - acc_dt).dt.days.clip(lower=0)
    out = decompose_datetime_features(
        out, date_cols=["DateTimeOfAccident", "DateReported"], drop_original=True
    )

    hours = out["HoursWorkedPerWeek"].replace(0, np.nan)
    out["hourly_wage"] = out["WeeklyWages"] / hours
    days = out["DaysWorkedPerWeek"].replace(0, np.nan)
    out["hours_per_day"] = out["HoursWorkedPerWeek"] / days
    out["total_dependents"] = out["DependentChildren"].fillna(0) + out["DependentsOther"].fillna(0)
    return out


def prepare_features(raw_df: pd.DataFrame, drop_sensitive: bool = False) -> pd.DataFrame:
    """Turn raw claim rows (same columns as train.csv, target optional) into model inputs.

    This is also the entry point for scoring *new* claims (e.g. from the UI):
    ``model.predict(prepare_features(new_claims, drop_sensitive=True))``.
    """
    eng = engineer_features(raw_df.drop(columns=[ID_COL, TARGET_COL], errors="ignore"))
    eng[TEXT_COL] = eng[TEXT_COL].fillna("").astype(str).str.upper()
    if drop_sensitive:
        eng = eng.drop(columns=SENSITIVE_COLS, errors="ignore")
    return eng


def make_xy(df: pd.DataFrame, drop_sensitive: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Engineer features, drop the ID and the target, and return X, y."""
    return prepare_features(df, drop_sensitive=drop_sensitive), df[TARGET_COL].copy()


def numeric_cols_of(X: pd.DataFrame) -> List[str]:
    """Plain numeric columns = every numeric column that is not in the skewed list."""
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
    """Structured features from Notebook 02's pipeline, plus TF-IDF text features.

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
        collinearity_threshold=0.95,
    )
    prep_defaults.update(prep_kwargs)

    structured = build_preprocessing_pipeline(
        numeric_cols=numeric_cols_of(X),
        skewed_numeric_cols=[c for c in SKEWED_NUMERIC_COLS if c in X.columns],
        low_card_cat_cols=[c for c in LOW_CARD_CAT_COLS if c in X.columns],
        # Age is intentionally NOT discretised here: it is already in numeric_cols,
        # and listing it twice makes the ColumnTransformer emit a duplicate "Age"
        # column (the error that currently stops Notebook 02 from running).
        discretize_cols=[],
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
