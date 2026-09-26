

# ============================================================
# LEAKAGE-FREE NESTED CROSS-VALIDATION
# Diabetes Classification
#
# Features:
#   - Missing-value handling using training-fold-only imputation
#   - Multiple imbalance strategies
#   - 10 classifiers
#   - 7 feature-ranking methods
#   - Greedy XGBoost feature selection
#   - Nested cross-validation
#   - Outer-fold-only final evaluation
#   - Statistical significance testing
#   - Reproducibility configuration
#
# IMPORTANT
# ------------------------------------------------------------
# NO INFORMATION FROM AN OUTER TEST FOLD IS USED FOR:
#
#   * Missing-value imputation
#   * Scaling
#   * Resampling
#   * Feature ranking
#   * Feature selection
#   * Classifier selection
#   * Model fitting
#
# Missing-value imputation is learned ONLY from training data.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wilcoxon

from sklearn.base import clone

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import mutual_info_classif

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from sklearn.inspection import permutation_importance

from imblearn.over_sampling import ADASYN

from imblearn.under_sampling import TomekLinks

from xgboost import XGBClassifier

from boruta import BorutaPy

import shap

from lime.lime_tabular import LimeTabularExplainer


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "data.csv"

OUTPUT_DIR = "nested_cv_results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_SEED = 42

N_OUTER_SPLITS = 10

N_INNER_SPLITS = 5

TOP_K = 6

MIN_SELECTED_FEATURES = 1

# ------------------------------------------------------------
# SHAP
# ------------------------------------------------------------

SHAP_BACKGROUND_SIZE = 10

SHAP_SAMPLE_SIZE = 100

SHAP_NSAMPLES = 100

# ------------------------------------------------------------
# LIME
# ------------------------------------------------------------

LIME_SAMPLE_SIZE = 100

# ------------------------------------------------------------
# Feature-ranking tree counts
# ------------------------------------------------------------

RF_TREES = 500

PDI_TREES = 500

BORUTA_TREES = 200

# ------------------------------------------------------------
# Expensive explainers
# ------------------------------------------------------------

RUN_EXPENSIVE_EXPLAINERS = True


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# 4. CLEAN STRING COLUMNS
# ============================================================

for col in df.select_dtypes(
    include=["object"]
).columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# 5. BINARY ENCODING
# ============================================================

binary_map = {

    "gender": {
        "male": 1,
        "female": 0
    },

    "family_diabetes": {
        "yes": 1,
        "no": 0
    },

    "hypertensive": {
        "yes": 1,
        "no": 0
    },

    "family_hypertension": {
        "yes": 1,
        "no": 0
    },

    "cardiovascular_disease": {
        "yes": 1,
        "no": 0
    },

    "stroke": {
        "yes": 1,
        "no": 0
    },

    "diabetic": {
        "yes": 1,
        "no": 0
    }
}


for col, mapping in binary_map.items():

    if col in df.columns:

        df[col] = df[col].replace(
            mapping
        )


# ============================================================
# 6. TARGET VALIDATION
# ============================================================

TARGET = "diabetic"

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' "
        f"was not found in the dataset."
    )


# ============================================================
# 7. CONVERT TARGET
# ============================================================

if df[TARGET].dtype == "object":

    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "yes": 1,
            "no": 0
        })
    )


# ============================================================
# 8. CHECK TARGET
# ============================================================

if df[TARGET].isna().any():

    print("\nWARNING:")
    print(
        "Rows with missing target values will be removed."
    )

    df = df.dropna(
        subset=[TARGET]
    ).reset_index(
        drop=True
    )


df[TARGET] = df[TARGET].astype(int)


# ============================================================
# 9. CONVERT REMAINING CATEGORICAL VARIABLES
# ============================================================

X_raw = df.drop(
    columns=[TARGET]
).copy()

y = df[TARGET].copy()


# ------------------------------------------------------------
# Try numeric conversion for object columns.
# ------------------------------------------------------------

for col in X_raw.columns:

    if X_raw[col].dtype == "object":

        converted = pd.to_numeric(
            X_raw[col],
            errors="coerce"
        )

        # If conversion produces at least some valid
        # numeric values, use it.
        if converted.notna().sum() > 0:

            X_raw[col] = converted


# ============================================================
# 10. CHECK NON-NUMERIC COLUMNS
# ============================================================

non_numeric = X_raw.select_dtypes(
    exclude=np.number
).columns.tolist()


if len(non_numeric) > 0:

    raise ValueError(
        "\nNon-numeric predictor columns remain:\n"
        f"{non_numeric}\n\n"
        "These columns must be encoded before "
        "running the nested CV."
    )


# ============================================================
# 11. REPLACE INF VALUES WITH NaN
# ============================================================

X_raw = X_raw.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 12. MISSING-VALUE REPORT
# ============================================================

missing_counts = (
    X_raw
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

missing_counts = missing_counts[
    missing_counts > 0
]


print("\n" + "=" * 70)
print("MISSING-VALUE REPORT")
print("=" * 70)

if len(missing_counts) == 0:

    print(
        "No missing predictor values detected."
    )

else:

    print(
        "\nColumns containing missing values:"
    )

    print(
        missing_counts
    )

    print(
        "\nTotal missing predictor values:",
        int(X_raw.isna().sum().sum())
    )

    print(
        "Rows containing at least one missing predictor:",
        int(X_raw.isna().any(axis=1).sum())
    )


# ============================================================
# IMPORTANT
# ------------------------------------------------------------
# DO NOT IMPUTE HERE.
#
# X_raw remains unchanged.
#
# Imputation will occur separately inside every training
# partition.
# ============================================================

X = X_raw.copy()

feature_names = X.columns.tolist()


# ============================================================
# 13. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(
    "\nDataset shape:",
    X.shape
)

print(
    "\nNumber of features:",
    len(feature_names)
)

print(
    "\nClass distribution:"
)

print(
    y.value_counts()
)

print(
    "\nClass proportions:"
)

print(
    y.value_counts(
        normalize=True
    )
)


# ============================================================
# 14. IMPUTATION FUNCTION
# ============================================================

def fit_imputer(
    X_train
):

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imputed = imputer.fit_transform(
        X_train
    )

    X_imputed = pd.DataFrame(
        X_imputed,
        columns=X_train.columns,
        index=X_train.index
    )

    return (
        imputer,
        X_imputed
    )


def transform_with_imputer(
    imputer,
    X_data
):

    X_imputed = imputer.transform(
        X_data
    )

    X_imputed = pd.DataFrame(
        X_imputed,
        columns=X_data.columns,
        index=X_data.index
    )

    return X_imputed


# ============================================================
# 15. MODEL DEFINITIONS
# ============================================================

def create_models():

    models = {

        "LogisticRegression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ),

        "SVM":
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ),

        "KNN":
            KNeighborsClassifier(),

        "DecisionTree":
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_SEED
            ),

        "NaiveBayes":
            GaussianNB(),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),

        "ExtraTrees":
            ExtraTreesClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),

        "GradientBoosting":
            GradientBoostingClassifier(
                random_state=RANDOM_SEED
            ),

        "AdaBoost":
            AdaBoostClassifier(
                random_state=RANDOM_SEED
            ),

        "XGBoost":
            XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                scale_pos_weight=1,
                n_jobs=-1
            )
    }

    return models


# ============================================================
# 16. SAMPLING
# ============================================================

def apply_sampler(
    X_train,
    y_train,
    strategy
):

    if strategy == "Original":

        return (
            X_train.copy(),
            y_train.copy()
        )


    elif strategy == "ADASYN":

        sampler = ADASYN(
            random_state=RANDOM_SEED
        )

        X_new, y_new = sampler.fit_resample(
            X_train,
            y_train
        )

        X_new = pd.DataFrame(
            X_new,
            columns=X_train.columns
        )

        y_new = pd.Series(
            y_new,
            name=y_train.name
        )

        return (
            X_new,
            y_new
        )


    elif strategy == "TomekLinks":

        sampler = TomekLinks()

        X_new, y_new = sampler.fit_resample(
            X_train,
            y_train
        )

        X_new = pd.DataFrame(
            X_new,
            columns=X_train.columns
        )

        y_new = pd.Series(
            y_new,
            name=y_train.name
        )

        return (
            X_new,
            y_new
        )


    elif strategy == "ADASYN+TomekLinks":

        adasyn = ADASYN(
            random_state=RANDOM_SEED
        )

        X_temp, y_temp = adasyn.fit_resample(
            X_train,
            y_train
        )

        tomek = TomekLinks()

        X_new, y_new = tomek.fit_resample(
            X_temp,
            y_temp
        )

        X_new = pd.DataFrame(
            X_new,
            columns=X_train.columns
        )

        y_new = pd.Series(
            y_new,
            name=y_train.name
        )

        return (
            X_new,
            y_new
        )


    else:

        raise ValueError(
            f"Unknown imbalance strategy: {strategy}"
        )


STRATEGIES = [
    "Original",
    "ADASYN",
    "TomekLinks",
    "ADASYN+TomekLinks"
]


# ============================================================
# 17. FEATURE RANKING
# ============================================================

def rank_mutual_information(
    X_train,
    y_train
):

    scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=RANDOM_SEED
    )

    return pd.Series(
        scores,
        index=X_train.columns
    ).sort_values(
        ascending=False
    )


# ============================================================
# CORRELATION
# ============================================================

def rank_correlation(
    X_train,
    y_train
):

    temp = X_train.copy()

    temp["diabetic"] = (
        y_train.values
    )

    corr = (
        temp.corr()["diabetic"]
        .drop("diabetic")
        .abs()
        .sort_values(
            ascending=False
        )
    )

    return corr


# ============================================================
# RANDOM FOREST
# ============================================================

def rank_random_forest(
    X_train,
    y_train
):

    model = RandomForestClassifier(
        n_estimators=RF_TREES,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(
        ascending=False
    )


# ============================================================
# BORUTA
# ============================================================

def rank_boruta(
    X_train,
    y_train
):

    model = RandomForestClassifier(
        n_estimators=BORUTA_TREES,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=6,
        random_state=RANDOM_SEED
    )

    boruta = BorutaPy(
        model,
        n_estimators="auto",
        random_state=RANDOM_SEED,
        verbose=0
    )

    boruta.fit(
        X_train.values,
        y_train.values
    )

    ranking = pd.Series(
        boruta.ranking_,
        index=X_train.columns
    )

    return ranking.sort_values(
        ascending=True
    )


# ============================================================
# SHAP
# ============================================================

def rank_shap(
    X_train,
    y_train
):

    model = RandomForestClassifier(
        n_estimators=RF_TREES,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    background_size = min(
        SHAP_BACKGROUND_SIZE,
        len(X_train)
    )

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X_train)
    )

    background = shap.kmeans(
        X_train,
        background_size
    )

    explainer = shap.KernelExplainer(
        model.predict_proba,
        background
    )

    X_sample = X_train.iloc[
        :sample_size
    ]

    shap_values = explainer.shap_values(
        X_sample,
        nsamples=SHAP_NSAMPLES
    )

    if isinstance(
        shap_values,
        list
    ):

        shap_class1 = shap_values[1]

    else:

        shap_values = np.asarray(
            shap_values
        )

        if shap_values.ndim == 3:

            shap_class1 = (
                shap_values[:, :, 1]
            )

        elif shap_values.ndim == 2:

            shap_class1 = shap_values

        else:

            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{shap_values.shape}"
            )

    importance = np.abs(
        shap_class1
    ).mean(
        axis=0
    )

    return pd.Series(
        importance,
        index=X_train.columns
    ).sort_values(
        ascending=False
    )


# ============================================================
# LIME
# ============================================================

def rank_lime(
    X_train,
    y_train
):

    model = RandomForestClassifier(
        n_estimators=RF_TREES,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=[
            "non-diabetic",
            "diabetic"
        ],
        discretize_continuous=True,
        random_state=RANDOM_SEED
    )

    importance = np.zeros(
        len(X_train.columns)
    )

    n_samples = min(
        LIME_SAMPLE_SIZE,
        len(X_train)
    )

    for i in range(
        n_samples
    ):

        explanation = (
            explainer.explain_instance(
                X_train.iloc[i].values,
                model.predict_proba,
                num_features=len(
                    X_train.columns
                )
            )
        )

        for condition, value in (
            explanation.as_list()
        ):

            for feature in X_train.columns:

                if feature in condition:

                    idx = (
                        X_train.columns
                        .get_loc(feature)
                    )

                    importance[idx] += (
                        abs(value)
                    )

                    break

    return pd.Series(
        importance,
        index=X_train.columns
    ).sort_values(
        ascending=False
    )


# ============================================================
# PERMUTATION DROP IMPORTANCE
# ============================================================

def rank_pdi(
    X_train,
    y_train
):

    model = RandomForestClassifier(
        n_estimators=PDI_TREES,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    result = permutation_importance(
        model,
        X_train,
        y_train,
        n_repeats=20,
        random_state=RANDOM_SEED,
        scoring="accuracy",
        n_jobs=-1
    )

    return pd.Series(
        result.importances_mean,
        index=X_train.columns
    ).sort_values(
        ascending=False
    )


# ============================================================
# 18. MASTER FEATURE RANKING
# ============================================================

def calculate_feature_rankings(
    X_train,
    y_train,
    run_expensive=True
):

    rankings = {}

    print(
        "      Mutual Information..."
    )

    rankings["MutualInfo"] = (
        rank_mutual_information(
            X_train,
            y_train
        )
    )

    print(
        "      Correlation..."
    )

    rankings["Correlation"] = (
        rank_correlation(
            X_train,
            y_train
        )
    )

    print(
        "      Random Forest..."
    )

    rankings["RandomForest"] = (
        rank_random_forest(
            X_train,
            y_train
        )
    )

    print(
        "      Boruta..."
    )

    rankings["Boruta"] = (
        rank_boruta(
            X_train,
            y_train
        )
    )

    if run_expensive:

        print(
            "      SHAP..."
        )

        rankings["SHAP"] = (
            rank_shap(
                X_train,
                y_train
            )
        )

        print(
            "      LIME..."
        )

        rankings["LIME"] = (
            rank_lime(
                X_train,
                y_train
            )
        )

    else:

        print(
            "      SHAP/LIME disabled."
        )


    print(
        "      PDI..."
    )

    rankings["PDI"] = (
        rank_pdi(
            X_train,
            y_train
        )
    )

    return rankings


# ============================================================
# 19. GLOBAL FEATURE LIST
# ============================================================

def create_global_feature_list(
    rankings,
    top_k=TOP_K
):

    feature_counter = Counter()

    rank_positions = {}

    for method, ranking in rankings.items():

        top_features = (
            ranking.index[:top_k]
        )

        feature_counter.update(
            top_features
        )

        for position, feature in enumerate(
            top_features,
            start=1
        ):

            if feature not in rank_positions:

                rank_positions[
                    feature
                ] = []

            rank_positions[
                feature
            ].append(
                position
            )


    frequency_df = pd.DataFrame.from_dict(
        feature_counter,
        orient="index",
        columns=["Frequency"]
    )


    frequency_df["MeanRank"] = [
        np.mean(
            rank_positions[f]
        )
        for f in frequency_df.index
    ]


    frequency_df = (
        frequency_df
        .sort_values(
            by=[
                "Frequency",
                "MeanRank"
            ],
            ascending=[
                False,
                True
            ]
        )
    )


    return (
        frequency_df.index.tolist(),
        frequency_df
    )


# ============================================================
# 20. XGBOOST
# ============================================================

def create_xgb():

    return XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        scale_pos_weight=1,
        n_jobs=-1
    )


# ============================================================
# 21. GREEDY XGBOOST FEATURE SELECTION
# ============================================================

def greedy_xgb_feature_selection(
    X_train,
    y_train,
    candidate_features,
    inner_cv
):

    selected = []

    best_f1 = -np.inf

    progress = []

    for feature in candidate_features:

        current_features = (
            selected + [feature]
        )

        X_candidate = (
            X_train[current_features]
        )

        fold_scores = []

        for (
            inner_train_idx,
            inner_val_idx
        ) in inner_cv.split(
            X_candidate,
            y_train
        ):

            X_inner_train = (
                X_candidate.iloc[
                    inner_train_idx
                ]
            )

            y_inner_train = (
                y_train.iloc[
                    inner_train_idx
                ]
            )

            X_inner_val = (
                X_candidate.iloc[
                    inner_val_idx
                ]
            )

            y_inner_val = (
                y_train.iloc[
                    inner_val_idx
                ]
            )

            # ------------------------------------------------
            # IMPORTANT
            #
            # X_train reaching this function has already been
            # imputed using the OUTER-TRAINING data only.
            #
            # Therefore no outer-test information is present.
            #
            # ------------------------------------------------

            X_balanced, y_balanced = (
                apply_sampler(
                    X_inner_train,
                    y_inner_train,
                    "Original"
                )
            )

            model = Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler()
                    ),

                    (
                        "xgb",
                        create_xgb()
                    )
                ]
            )

            model.fit(
                X_balanced,
                y_balanced
            )

            prediction = (
                model.predict(
                    X_inner_val
                )
            )

            fold_scores.append(
                f1_score(
                    y_inner_val,
                    prediction,
                    zero_division=0
                )
            )


        mean_f1 = np.mean(
            fold_scores
        )


        if (
            len(selected) == 0
            or mean_f1 >= best_f1
        ):

            selected.append(
                feature
            )

            best_f1 = mean_f1

            progress.append(
                {
                    "NumberFeatures":
                        len(selected),

                    "Feature":
                        feature,

                    "F1":
                        mean_f1
                }
            )


    return (
        selected,
        progress
    )


# ============================================================
# 22. INNER CLASSIFIER EVALUATION
# ============================================================

def evaluate_classifier_inner_cv(
    X_train,
    y_train,
    features,
    classifier_name,
    strategy,
    inner_cv
):

    model = create_models()[
        classifier_name
    ]

    X_selected = (
        X_train[features]
    )

    scores = []

    for (
        inner_train_idx,
        inner_val_idx
    ) in inner_cv.split(
        X_selected,
        y_train
    ):

        X_inner_train = (
            X_selected.iloc[
                inner_train_idx
            ]
        )

        y_inner_train = (
            y_train.iloc[
                inner_train_idx
            ]
        )

        X_inner_val = (
            X_selected.iloc[
                inner_val_idx
            ]
        )

        y_inner_val = (
            y_train.iloc[
                inner_val_idx
            ]
        )


        X_balanced, y_balanced = (
            apply_sampler(
                X_inner_train,
                y_inner_train,
                strategy
            )
        )


        pipe = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    clone(model)
                )
            ]
        )


        pipe.fit(
            X_balanced,
            y_balanced
        )


        prediction = (
            pipe.predict(
                X_inner_val
            )
        )


        scores.append(
            f1_score(
                y_inner_val,
                prediction,
                zero_division=0
            )
        )


    return np.mean(
        scores
    )


# ============================================================
# 23. CLASSIFIER SELECTION
# ============================================================

def select_best_classifier(
    X_train,
    y_train,
    features,
    strategy,
    inner_cv
):

    models = create_models()

    classifier_scores = {}

    for name in models.keys():

        score = (
            evaluate_classifier_inner_cv(
                X_train,
                y_train,
                features,
                name,
                strategy,
                inner_cv
            )
        )

        classifier_scores[
            name
        ] = score

        print(
            f"        {name}: "
            f"F1={score:.4f}"
        )


    best_classifier = max(
        classifier_scores,
        key=classifier_scores.get
    )


    return (
        best_classifier,
        classifier_scores
    )


# ============================================================
# 24. OUTER-FOLD EVALUATION
# ============================================================

def evaluate_outer_fold(
    X_train,
    y_train,
    X_test,
    y_test,
    features,
    classifier_name,
    strategy
):

    X_train_selected = (
        X_train[features]
    )

    X_test_selected = (
        X_test[features]
    )


    X_balanced, y_balanced = (
        apply_sampler(
            X_train_selected,
            y_train,
            strategy
        )
    )


    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                create_models()[
                    classifier_name
                ]
            )
        ]
    )


    model.fit(
        X_balanced,
        y_balanced
    )


    prediction = (
        model.predict(
            X_test_selected
        )
    )


    probability = None


    if hasattr(
        model,
        "predict_proba"
    ):

        probability = (
            model.predict_proba(
                X_test_selected
            )[:, 1]
        )


    cm = confusion_matrix(
        y_test,
        prediction
    )


    result = {

        "Accuracy":
            accuracy_score(
                y_test,
                prediction
            ),

        "Precision":
            precision_score(
                y_test,
                prediction,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                prediction,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                prediction,
                zero_division=0
            ),

        "TN":
            cm[0, 0],

        "FP":
            cm[0, 1],

        "FN":
            cm[1, 0],

        "TP":
            cm[1, 1]
    }


    if probability is not None:

        try:

            result["ROC_AUC"] = (
                roc_auc_score(
                    y_test,
                    probability
                )
            )

        except Exception:

            result["ROC_AUC"] = np.nan

    else:

        result["ROC_AUC"] = np.nan


    return (
        result,
        prediction
    )


# ============================================================
# 25. OUTER CV
# ============================================================

print("\n")

print("=" * 70)
print("STARTING LEAKAGE-FREE NESTED CROSS-VALIDATION")
print("=" * 70)


outer_cv = StratifiedKFold(
    n_splits=N_OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)


all_results = []

all_selected_features = []

all_classifier_selection = []

outer_predictions = []


for strategy in STRATEGIES:

    print("\n")

    print(
        "#" * 70
    )

    print(
        f"IMBALANCE STRATEGY: {strategy}"
    )

    print(
        "#" * 70
    )


    strategy_dir = os.path.join(
        OUTPUT_DIR,
        strategy.replace(
            "+",
            "_"
        )
    )


    os.makedirs(
        strategy_dir,
        exist_ok=True
    )


    for outer_fold, (
        outer_train_idx,
        outer_test_idx
    ) in enumerate(
        outer_cv.split(X, y),
        start=1
    ):

        print("\n")

        print(
            "=" * 60
        )

        print(
            f"{strategy} | "
            f"OUTER FOLD "
            f"{outer_fold}/"
            f"{N_OUTER_SPLITS}"
        )

        print(
            "=" * 60
        )


        # ====================================================
        # OUTER SPLIT
        # ====================================================

        X_outer_train_raw = (
            X.iloc[
                outer_train_idx
            ].copy()
        )

        y_outer_train = (
            y.iloc[
                outer_train_idx
            ].copy()
        )


        X_outer_test_raw = (
            X.iloc[
                outer_test_idx
            ].copy()
        )

        y_outer_test = (
            y.iloc[
                outer_test_idx
            ].copy()
        )


        print(
            "Outer training samples:",
            len(X_outer_train_raw)
        )

        print(
            "Outer test samples:",
            len(X_outer_test_raw)
        )


        # ====================================================
        # OUTER-TRAINING-ONLY IMPUTATION
        #
        # THIS IS THE CRITICAL FIX
        # ====================================================

        print(
            "\n  STEP 0: "
            "Training-fold-only median imputation"
        )


        outer_imputer, X_outer_train = (
            fit_imputer(
                X_outer_train_raw
            )
        )


        X_outer_test = (
            transform_with_imputer(
                outer_imputer,
                X_outer_test_raw
            )
        )


        # ----------------------------------------------------
        # Verification
        # ----------------------------------------------------

        if X_outer_train.isna().any().any():

            raise ValueError(
                "NaN values remain in "
                "outer training data after imputation."
            )


        if X_outer_test.isna().any().any():

            raise ValueError(
                "NaN values remain in "
                "outer test data after imputation."
            )


        # ====================================================
        # INNER CV
        # ====================================================

        inner_cv = StratifiedKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED
        )


        # ====================================================
        # STEP 1
        # FEATURE RANKING
        #
        # Only outer training data
        # ====================================================

        print(
            "\n  STEP 1: Feature ranking"
        )


        # ----------------------------------------------------
        # Resampling is performed only on outer training data.
        # ----------------------------------------------------

        X_rank_train, y_rank_train = (
            apply_sampler(
                X_outer_train,
                y_outer_train,
                strategy
            )
        )


        rankings = (
            calculate_feature_rankings(
                X_rank_train,
                y_rank_train,
                run_expensive=
                    RUN_EXPENSIVE_EXPLAINERS
            )
        )


        # ====================================================
        # SAVE FEATURE RANKINGS
        # ====================================================

        ranking_table = pd.DataFrame(
            {
                method:
                    ranking.reindex(
                        feature_names
                    )

                for method, ranking
                in rankings.items()
            }
        )


        ranking_table.to_csv(
            os.path.join(
                strategy_dir,
                f"outer_fold_"
                f"{outer_fold}_"
                f"feature_rankings.csv"
            )
        )


        # ====================================================
        # GLOBAL FEATURE LIST
        # ====================================================

        (
            global_features,
            frequency_df
        ) = create_global_feature_list(
            rankings,
            top_k=TOP_K
        )


        frequency_df.to_csv(
            os.path.join(
                strategy_dir,
                f"outer_fold_"
                f"{outer_fold}_"
                f"feature_frequency.csv"
            )
        )


        all_selected_features.append(
            {
                "Strategy":
                    strategy,

                "OuterFold":
                    outer_fold,

                "Features":
                    "|".join(
                        global_features
                    )
            }
        )


        # ====================================================
        # STEP 2
        # GREEDY XGBOOST
        # ====================================================

        print(
            "\n  STEP 2: "
            "Greedy XGBoost feature selection"
        )


        greedy_features, greedy_progress = (
            greedy_xgb_feature_selection(
                X_outer_train,
                y_outer_train,
                global_features,
                inner_cv
            )
        )


        if len(greedy_features) == 0:

            greedy_features = [
                global_features[0]
            ]


        print(
            "\n  Selected features:"
        )

        print(
            greedy_features
        )


        pd.DataFrame(
            greedy_progress
        ).to_csv(
            os.path.join(
                strategy_dir,
                f"outer_fold_"
                f"{outer_fold}_"
                f"greedy_selection.csv"
            ),
            index=False
        )


        # ====================================================
        # STEP 3
        # CLASSIFIER SELECTION
        # ====================================================

        print(
            "\n  STEP 3: "
            "Classifier selection"
        )


        (
            best_classifier,
            classifier_scores
        ) = select_best_classifier(
            X_outer_train,
            y_outer_train,
            greedy_features,
            strategy,
            inner_cv
        )


        print(
            "\n  Selected classifier:",
            best_classifier
        )


        classifier_record = {

            "Strategy":
                strategy,

            "OuterFold":
                outer_fold,

            "SelectedClassifier":
                best_classifier
        }


        for (
            name,
            score
        ) in classifier_scores.items():

            classifier_record[
                f"InnerF1_{name}"
            ] = score


        all_classifier_selection.append(
            classifier_record
        )


        # ====================================================
        # STEP 4
        # OUTER TEST EVALUATION
        # ====================================================

        print(
            "\n  STEP 4: "
            "Outer-test evaluation"
        )


        (
            outer_result,
            prediction
        ) = evaluate_outer_fold(

            X_outer_train,

            y_outer_train,

            X_outer_test,

            y_outer_test,

            greedy_features,

            best_classifier,

            strategy
        )


        # ====================================================
        # SAVE METADATA
        # ====================================================

        outer_result[
            "Strategy"
        ] = strategy

        outer_result[
            "OuterFold"
        ] = outer_fold

        outer_result[
            "Classifier"
        ] = best_classifier

        outer_result[
            "NumberFeatures"
        ] = len(
            greedy_features
        )

        outer_result[
            "SelectedFeatures"
        ] = "|".join(
            greedy_features
        )


        all_results.append(
            outer_result
        )


        # ====================================================
        # OUTER PREDICTIONS
        # ====================================================

        for (
            true_value,
            pred_value
        ) in zip(
            y_outer_test,
            prediction
        ):

            outer_predictions.append(
                {

                    "Strategy":
                        strategy,

                    "OuterFold":
                        outer_fold,

                    "Actual":
                        true_value,

                    "Predicted":
                        pred_value
                }
            )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            "\n  OUTER TEST RESULTS"
        )


        print(
            f"  Accuracy : "
            f"{outer_result['Accuracy']:.4f}"
        )

        print(
            f"  Precision: "
            f"{outer_result['Precision']:.4f}"
        )

        print(
            f"  Recall   : "
            f"{outer_result['Recall']:.4f}"
        )

        print(
            f"  F1       : "
            f"{outer_result['F1']:.4f}"
        )

        print(
            f"  ROC-AUC  : "
            f"{outer_result['ROC_AUC']:.4f}"
        )


# ============================================================
# 26. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_cv_outer_results.csv"
    ),
    index=False
)


predictions_df = pd.DataFrame(
    outer_predictions
)


predictions_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_cv_outer_predictions.csv"
    ),
    index=False
)


classifier_selection_df = (
    pd.DataFrame(
        all_classifier_selection
    )
)


classifier_selection_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_cv_classifier_selection.csv"
    ),
    index=False
)


selected_features_df = (
    pd.DataFrame(
        all_selected_features
    )
)


selected_features_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_cv_selected_features.csv"
    ),
    index=False
)


# ============================================================
# 27. PERFORMANCE SUMMARY
# ============================================================

metric_columns = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1",

    "ROC_AUC"
]


summary = (
    results_df
    .groupby(
        "Strategy"
    )[metric_columns]
    .agg(
        [
            "mean",
            "std"
        ]
    )
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_cv_summary_mean_std.csv"
    )
)


print("\n")

print(
    "=" * 70
)

print(
    "NESTED CV SUMMARY"
)

print(
    "=" * 70
)

print(
    summary
)


# ============================================================
# 28. CONFIDENCE INTERVALS
# ============================================================

def confidence_interval(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    mean = np.mean(
        values
    )

    std = np.std(
        values,
        ddof=1
    )

    n = len(
        values
    )

    margin = (
        1.96 *
        std /
        np.sqrt(n)
    )

    return (
        mean,
        mean - margin,
        mean + margin
    )


ci_records = []


for strategy in STRATEGIES:

    strategy_results = (
        results_df[
            results_df["Strategy"]
            == strategy
        ]
    )


    record = {
        "Strategy":
            strategy
    }


    for metric in metric_columns:

        mean, lower, upper = (
            confidence_interval(
                strategy_results[
                    metric
                ]
            )
        )


        record[
            f"{metric}_Mean"
        ] = mean


        record[
            f"{metric}_CI_Lower"
        ] = lower


        record[
            f"{metric}_CI_Upper"
        ] = upper


    ci_records.append(
        record
    )


ci_df = pd.DataFrame(
    ci_records
)


ci_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_cv_confidence_intervals.csv"
    ),
    index=False
)


# ============================================================
# 29. CONFUSION MATRICES
# ============================================================

for strategy in STRATEGIES:

    subset = predictions_df[
        predictions_df["Strategy"]
        == strategy
    ]


    cm = confusion_matrix(
        subset["Actual"],
        subset["Predicted"]
    )


    cm_percentage = (
        cm /
        cm.sum(
            axis=1,
            keepdims=True
        ) *
        100
    )


    plt.figure(
        figsize=(7, 6)
    )


    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        xticklabels=[
            "Non-Diabetic",
            "Diabetic"
        ],
        yticklabels=[
            "Non-Diabetic",
            "Diabetic"
        ]
    )


    plt.xlabel(
        "Predicted",
        fontsize=14
    )


    plt.ylabel(
        "Actual",
        fontsize=14
    )


    plt.title(
        f"Nested CV Outer-Test "
        f"Confusion Matrix (%)\n"
        f"{strategy}",
        fontsize=15
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix_"
            f"{strategy.replace('+', '_')}.png"
        ),
        dpi=300
    )


    plt.close()


# ============================================================
# 30. PERFORMANCE COMPARISON
# ============================================================

mean_performance = (
    results_df
    .groupby(
        "Strategy"
    )[metric_columns]
    .mean()
)


plt.figure(
    figsize=(12, 7)
)


mean_performance.plot(
    kind="bar"
)


plt.ylabel(
    "Score"
)

plt.xlabel(
    "Imbalance Strategy"
)

plt.title(
    "Leakage-Free Nested CV Performance"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=30
)

plt.legend(
    bbox_to_anchor=(
        1.02,
        1
    ),
    loc="upper left"
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "nested_cv_performance_comparison.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# 31. F1 DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(11, 7)
)


sns.boxplot(
    data=results_df,
    x="Strategy",
    y="F1"
)


plt.xlabel(
    "Imbalance Strategy"
)

plt.ylabel(
    "Outer-Test F1"
)

plt.title(
    "Outer-Test F1 Distribution "
    "Across Nested CV Folds"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "nested_cv_f1_boxplot.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# 32. CLASSIFIER SELECTION FREQUENCY
# ============================================================

classifier_frequency = (
    classifier_selection_df
    .groupby(
        [
            "Strategy",
            "SelectedClassifier"
        ]
    )
    .size()
    .reset_index(
        name="Frequency"
    )
)


classifier_frequency.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classifier_selection_frequency.csv"
    ),
    index=False
)


# ============================================================
# 33. FEATURE SELECTION FREQUENCY
# ============================================================

feature_counter_by_strategy = {}


for strategy in STRATEGIES:

    subset = selected_features_df[
        selected_features_df[
            "Strategy"
        ] == strategy
    ]


    counter = Counter()


    for feature_string in subset[
        "Features"
    ]:

        if pd.isna(
            feature_string
        ):

            continue


        features = (
            feature_string.split("|")
        )


        counter.update(
            features
        )


    feature_counter_by_strategy[
        strategy
    ] = counter


    frequency_df = (
        pd.DataFrame.from_dict(
            counter,
            orient="index",
            columns=[
                "Frequency"
            ]
        )
        .sort_values(
            "Frequency",
            ascending=False
        )
    )


    frequency_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_frequency_"
            f"{strategy.replace('+', '_')}.csv"
        )
    )


# ============================================================
# 34. WILCOXON TEST
# ============================================================

wilcoxon_records = []

reference_strategy = "Original"


reference_scores = (
    results_df[
        results_df["Strategy"]
        == reference_strategy
    ]
    .sort_values(
        "OuterFold"
    )["F1"]
    .values
)


for strategy in STRATEGIES:

    if strategy == reference_strategy:

        continue


    current_scores = (
        results_df[
            results_df["Strategy"]
            == strategy
        ]
        .sort_values(
            "OuterFold"
        )["F1"]
        .values
    )


    if (
        len(reference_scores)
        == len(current_scores)
    ):

        try:

            statistic, p_value = (
                wilcoxon(
                    reference_scores,
                    current_scores
                )
            )

        except ValueError:

            statistic = np.nan

            p_value = np.nan

    else:

        statistic = np.nan

        p_value = np.nan


    wilcoxon_records.append(
        {

            "Reference":
                reference_strategy,

            "Comparison":
                strategy,

            "Statistic":
                statistic,

            "PValue":
                p_value
        }
    )


wilcoxon_df = pd.DataFrame(
    wilcoxon_records
)


wilcoxon_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "wilcoxon_outer_fold_f1.csv"
    ),
    index=False
)


print("\n")

print(
    "=" * 70
)

print(
    "WILCOXON TEST"
)

print(
    "=" * 70
)

print(
    wilcoxon_df
)


# ============================================================
# 35. HYPERPARAMETER REPORT
# ============================================================

hyperparameter_report = pd.DataFrame(
    [

        {
            "Classifier":
                "LogisticRegression",

            "Hyperparameters":
                "max_iter=1000, "
                "class_weight=balanced",

            "ClassWeight":
                "Yes"
        },

        {
            "Classifier":
                "SVM",

            "Hyperparameters":
                "kernel=rbf, "
                "probability=True, "
                "class_weight=balanced",

            "ClassWeight":
                "Yes"
        },

        {
            "Classifier":
                "KNN",

            "Hyperparameters":
                "Default sklearn parameters",

            "ClassWeight":
                "No"
        },

        {
            "Classifier":
                "DecisionTree",

            "Hyperparameters":
                "class_weight=balanced",

            "ClassWeight":
                "Yes"
        },

        {
            "Classifier":
                "NaiveBayes",

            "Hyperparameters":
                "Default GaussianNB parameters",

            "ClassWeight":
                "No"
        },

        {
            "Classifier":
                "RandomForest",

            "Hyperparameters":
                "n_estimators=200, "
                "class_weight=balanced",

            "ClassWeight":
                "Yes"
        },

        {
            "Classifier":
                "ExtraTrees",

            "Hyperparameters":
                "n_estimators=200, "
                "class_weight=balanced",

            "ClassWeight":
                "Yes"
        },

        {
            "Classifier":
                "GradientBoosting",

            "Hyperparameters":
                "Default sklearn parameters",

            "ClassWeight":
                "No"
        },

        {
            "Classifier":
                "AdaBoost",

            "Hyperparameters":
                "Default sklearn parameters",

            "ClassWeight":
                "No"
        },

        {
            "Classifier":
                "XGBoost",

            "Hyperparameters":
                "eval_metric=logloss, "
                "scale_pos_weight=1",

            "ClassWeight":
                "No"
        }
    ]
)


hyperparameter_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_hyperparameters.csv"
    ),
    index=False
)


# ============================================================
# 36. MISSING-VALUE REPORT
# ============================================================

missing_report = pd.DataFrame(
    {
        "Feature":
            X.columns,

        "MissingBeforeImputation":
            X.isna().sum().values,

        "MissingPercentage":
            (
                X.isna().mean().values
                * 100
            )
    }
)


missing_report = (
    missing_report
    .sort_values(
        "MissingBeforeImputation",
        ascending=False
    )
)


missing_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missing_value_report.csv"
    ),
    index=False
)


# ============================================================
# 37. REPRODUCIBILITY CONFIGURATION
# ============================================================

reproducibility_config = {

    "data_file":
        DATA_FILE,

    "target":
        TARGET,

    "random_seed":
        RANDOM_SEED,

    "missing_value_strategy":
        "Median imputation",

    "missing_value_leakage_control":
        "Imputer fitted separately on training data only",

    "outer_cv":
        {
            "type":
                "StratifiedKFold",

            "n_splits":
                N_OUTER_SPLITS,

            "shuffle":
                True,

            "random_state":
                RANDOM_SEED
        },

    "inner_cv":
        {
            "type":
                "StratifiedKFold",

            "n_splits":
                N_INNER_SPLITS,

            "shuffle":
                True,

            "random_state":
                RANDOM_SEED
        },

    "imbalance_strategies":
        STRATEGIES,

    "top_k":
        TOP_K,

    "shap_background_size":
        SHAP_BACKGROUND_SIZE,

    "shap_sample_size":
        SHAP_SAMPLE_SIZE,

    "shap_nsamples":
        SHAP_NSAMPLES,

    "lime_sample_size":
        LIME_SAMPLE_SIZE,

    "nested_cross_validation":
        True,

    "outer_test_used_for_imputation":
        False,

    "outer_test_used_for_scaling":
        False,

    "outer_test_used_for_resampling":
        False,

    "outer_test_used_for_feature_ranking":
        False,

    "outer_test_used_for_feature_selection":
        False,

    "outer_test_used_for_classifier_selection":
        False,

    "outer_test_used_for_model_fitting":
        False
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "reproducibility_config.json"
    ),
    "w"
) as f:

    json.dump(
        reproducibility_config,
        f,
        indent=4
    )


# ============================================================
# 38. FINAL VALIDATION
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "FINAL VALIDATION"
)

print(
    "=" * 70
)


if results_df.isna().all().all():

    raise ValueError(
        "No valid nested-CV results were generated."
    )


print(
    "\nOuter-fold evaluations completed:",
    len(results_df)
)


print(
    "Expected evaluations:",
    len(STRATEGIES)
    * N_OUTER_SPLITS
)


# ============================================================
# 39. FINAL REPORT
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "PIPELINE COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nResults directory:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)


print(
    "\nFiles generated:"
)


for filename in sorted(
    os.listdir(
        OUTPUT_DIR
    )
):

    print(
        "  -",
        filename
    )


print("\n")

print(
    "IMPORTANT:"
)

print(
    "1. Missing values were handled using "
    "median imputation."
)

print(
    "2. Imputation was fitted only on "
    "training partitions."
)

print(
    "3. Outer test folds were not used "
    "for imputation."
)

print(
    "4. Outer test folds were not used "
    "for feature ranking."
)

print(
    "5. Outer test folds were not used "
    "for feature selection."
)

print(
    "6. Outer test folds were not used "
    "for classifier selection."
)

print(
    "7. Resampling was restricted to "
    "training data."
)

print(
    "8. Final performance is based only "
    "on outer-test predictions."
)

print(
    "\nNested cross-validation completed successfully."
)





