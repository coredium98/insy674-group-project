"""Modular, leak-free data preparation functions and custom transformers.

Structured directly around Aurélien Géron's Machine Learning Project Checklist
(Appendix B, "Prepare the Data"), INSY 674 Modules 3 and 4, and the findings
from `notebooks/01_explore_data.ipynb`:
- Notes: Work on copies (`df.copy()`), perform a claim-grouped split
  (`GroupShuffleSplit` on `ClaimNumber`) to prevent leakage across the 12,560
  rows sharing claim IDs, and encapsulate every step in functions or
  `BaseEstimator`/`TransformerMixin` classes.
- Step 1: Data cleaning (domain anomaly flags and sanitization for the 75 rows
  with `HoursWorkedPerWeek > 168`, 16 rows with `0` hours, and 2 negative
  reporting lags; train-only quantile clipping via `QuantileOutlierClipper`;
  missingness indicators plus configurable imputation for the 56 missing
  `WeeklyWages`, 49 `HoursWorkedPerWeek`, and 22 `MaritalStatus` records).
- Step 2: Feature selection (dropping `ClaimNumber` and outcome columns from X,
  plus automated collinearity pruning at |r| >= 0.95 via `CollinearityDropper`).
- Step 3: Feature engineering (external FRED Medical Care CPI join by
  `AccidentYear`, UTC timestamp decomposition and reporting lag in days,
  employment ratios, `KBinsDiscretizer` on `Age`, mathematical transforms
  `log1p`/`sqrt`/`Yeo-Johnson`, and TF-IDF + TruncatedSVD on `ClaimDescription`).
- Step 4: Feature scaling (`StandardScaler`, `MinMaxScaler`, or `RobustScaler`).
"""

from typing import List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)


class QuantileOutlierClipper(BaseEstimator, TransformerMixin):
    """Step 1 (Data cleaning): Clip numeric outliers using train-set quantiles.

    Because `clip_outliers` is a hyperparameter, cross-validation in Notebook 04
    can test whether clipping extreme values improves validation error.
    """

    def __init__(
        self,
        clip_outliers: bool = True,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ):
        self.clip_outliers = clip_outliers
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y=None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower_quantile)
        self.upper_bounds_ = X_df.quantile(self.upper_quantile)
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = pd.DataFrame(X).copy()
        if not self.clip_outliers:
            return X_copy
        return X_copy.clip(
            lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1
        )

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class ClaimTextSVDTransformer(BaseEstimator, TransformerMixin):
    """Step 3 (Feature engineering): TF-IDF + TruncatedSVD for ClaimDescription.

    As found in Notebook 01, `ClaimDescription` has 20,596 unique phrases with an
    average length of 7 words. This transformer fits a TF-IDF vocabulary and
    latent semantic components strictly on the training split and returns dense
    numeric columns (`text_svd_0`, ..., `text_svd_k`).
    """

    def __init__(
        self,
        n_components: int = 15,
        max_features: int = 500,
        ngram_range: tuple = (1, 2),
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.ngram_range = ngram_range

    def _extract_series(self, X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(np.asarray(X).ravel())
        return s.fillna("").astype(str)

    def fit(self, X, y=None):
        texts = self._extract_series(X)
        self.tfidf_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            lowercase=True,
        )
        tfidf_mat = self.tfidf_.fit_transform(texts)
        eff_components = min(self.n_components, max(1, tfidf_mat.shape[1] - 1))
        self.svd_ = TruncatedSVD(n_components=eff_components, random_state=42)
        self.svd_.fit(tfidf_mat)
        self.feature_names_out_ = np.array(
            [f"text_svd_{i}" for i in range(eff_components)], dtype=object
        )
        return self

    def transform(self, X) -> pd.DataFrame:
        texts = self._extract_series(X)
        tfidf_mat = self.tfidf_.transform(texts)
        svd_mat = self.svd_.transform(tfidf_mat)
        idx = X.index if hasattr(X, "index") else None
        return pd.DataFrame(svd_mat, columns=self.feature_names_out_, index=idx)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class CollinearityDropper(BaseEstimator, TransformerMixin):
    """Step 2 (Feature selection): Drop one of any pair of features with |r| >= threshold.

    Learns the correlated columns strictly on the training fold to prevent leakage.
    """

    def __init__(self, threshold: float = 0.95, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)
        if not self.enabled or X_df.shape[1] <= 1:
            self.cols_to_drop_ = []
            return self

        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.cols_to_drop_ = [
            col for col in upper.columns if any(upper[col] >= self.threshold)
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = pd.DataFrame(X).copy()
        if not self.cols_to_drop_:
            return X_copy
        return X_copy.drop(columns=self.cols_to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features=None):
        cols = (
            list(input_features)
            if input_features is not None
            else list(self.feature_names_in_)
        )
        return np.array(
            [c for c in cols if c not in self.cols_to_drop_], dtype=object
        )


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the row-level audit targets defined in Notebook 01.

    - `ReserveShortfall` = `UltimateIncurredClaimCost - InitialIncurredCalimsCost`
      (primary dollar audit target; positive means under-reserving)
    - `LogDevelopmentRatio` = `log(UltimateIncurredClaimCost / InitialIncurredCalimsCost)`
      (scale-normalized log target; defined for all 36,176 rows since both costs > 0)
    """
    out = df.copy()
    out["ReserveShortfall"] = (
        out["UltimateIncurredClaimCost"] - out["InitialIncurredCalimsCost"]
    )
    out["LogDevelopmentRatio"] = np.log(
        out["UltimateIncurredClaimCost"] / out["InitialIncurredCalimsCost"]
    )
    return out


def drop_uninformative_columns(
    df: pd.DataFrame, cols_to_drop: Sequence[str]
) -> pd.DataFrame:
    """Step 2 (Feature selection): Return a copy with ID or outcome columns removed."""
    out = df.copy()
    existing = [c for c in cols_to_drop if c in out.columns]
    return out.drop(columns=existing)


def decompose_datetime_features(
    df: pd.DataFrame,
    date_cols: Sequence[str],
    drop_original: bool = True,
) -> pd.DataFrame:
    """Step 3 (Feature engineering): Decompose UTC timestamps into calendar parts."""
    out = df.copy()
    for col in date_cols:
        if col not in out.columns:
            continue
        dt = pd.to_datetime(out[col], errors="coerce", utc=True)
        out[f"{col}_year"] = dt.dt.year
        out[f"{col}_month"] = dt.dt.month
        out[f"{col}_day"] = dt.dt.day
        out[f"{col}_dayofweek"] = dt.dt.dayofweek
        out[f"{col}_hour"] = dt.dt.hour
        out[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        if drop_original:
            out = out.drop(columns=[col])
    return out


def build_preprocessing_pipeline(
    numeric_cols: List[str],
    skewed_numeric_cols: Optional[List[str]] = None,
    discretize_cols: Optional[List[str]] = None,
    low_card_cat_cols: Optional[List[str]] = None,
    ordinal_cat_cols: Optional[List[str]] = None,
    text_col: Optional[str] = None,
    clip_outliers: bool = True,
    num_imputer_strategy: str = "median",
    add_missing_indicator: bool = True,
    skew_transform: str = "log1p",
    n_bins: int = 5,
    n_text_svd_components: int = 15,
    scaler_type: str = "standard",
    collinearity_threshold: float = 0.95,
) -> Pipeline:
    """Build a complete, leak-free preparation pipeline covering Steps 1 to 4.

    Every choice (outlier clipping, imputation strategy, missingness flags,
    skew transformation, bin count, text SVD dimensions, scaling method, and
    collinearity threshold) is exposed as a parameter so it can be tuned as a
    hyperparameter in Notebook 04.
    """
    skewed_numeric_cols = skewed_numeric_cols or []
    discretize_cols = discretize_cols or []
    low_card_cat_cols = low_card_cat_cols or []
    ordinal_cat_cols = ordinal_cat_cols or []

    # Step 1: Missing value imputer choice
    if num_imputer_strategy == "iterative":
        num_imputer = IterativeImputer(
            random_state=42,
            max_iter=10,
            add_indicator=add_missing_indicator,
        )
    elif num_imputer_strategy == "zero":
        num_imputer = SimpleImputer(
            strategy="constant",
            fill_value=0,
            add_indicator=add_missing_indicator,
        )
    else:
        num_imputer = SimpleImputer(
            strategy=num_imputer_strategy,
            add_indicator=add_missing_indicator,
        )

    # Step 3: Promising mathematical transformation choice for skewed features
    if skew_transform == "log1p":
        math_transformer = FunctionTransformer(
            np.log1p, feature_names_out="one-to-one"
        )
    elif skew_transform == "sqrt":
        math_transformer = FunctionTransformer(
            np.sqrt, feature_names_out="one-to-one"
        )
    else:
        math_transformer = PowerTransformer(
            method="yeo-johnson", standardize=False
        )

    # Step 4: Feature scaling choice (Standardize vs. Normalize vs. Robust)
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = "passthrough"

    transformers = []

    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("clipper", QuantileOutlierClipper(clip_outliers=clip_outliers)),
                ("imputer", num_imputer),
                ("scaler", scaler),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    if skewed_numeric_cols:
        skew_pipe = Pipeline(
            steps=[
                ("clipper", QuantileOutlierClipper(clip_outliers=clip_outliers)),
                ("imputer", num_imputer),
                ("math_transform", math_transformer),
                ("scaler", scaler),
            ]
        )
        transformers.append(("skewed_num", skew_pipe, skewed_numeric_cols))

    if discretize_cols:
        disc_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "binner",
                    KBinsDiscretizer(
                        n_bins=n_bins,
                        encode="ordinal",
                        strategy="quantile",
                    ),
                ),
            ]
        )
        transformers.append(("discretized", disc_pipe, discretize_cols))

    if low_card_cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        min_frequency=10,
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat_ohe", cat_pipe, low_card_cat_cols))

    if ordinal_cat_cols:
        ord_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )
        transformers.append(("cat_ord", ord_pipe, ordinal_cat_cols))

    if text_col:
        text_pipe = Pipeline(
            steps=[
                (
                    "tfidf_svd",
                    ClaimTextSVDTransformer(n_components=n_text_svd_components),
                ),
                ("scaler", scaler),
            ]
        )
        transformers.append(("text", text_pipe, [text_col]))

    col_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    col_transformer.set_output(transform="pandas")

    full_prep_pipeline = Pipeline(
        steps=[
            ("columns", col_transformer),
            (
                "collinearity_filter",
                CollinearityDropper(threshold=collinearity_threshold),
            ),
        ]
    )
    full_prep_pipeline.set_output(transform="pandas")
    return full_prep_pipeline
