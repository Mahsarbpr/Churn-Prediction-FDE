import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier


MODEL_FEATURES = [
    "recency_days",
    "has_meaningful_history",
    "sessions_30d",
    "sessions_90d",
    "purchase_count_90d",
    "revenue_90d",
]


def split_by_customer(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    train_index, test_index = next(
        splitter.split(
            dataset,
            groups=dataset["customer_id"],
        )
    )

    train = dataset.iloc[train_index].reset_index(drop=True)
    test = dataset.iloc[test_index].reset_index(drop=True)

    return train, test


def train_logistic_regression(
    train: pd.DataFrame,
) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        train[MODEL_FEATURES],
        train["churn"],
    )

    return model


def evaluate_model(
    model,
    test: pd.DataFrame,
) -> dict:
    probabilities = model.predict_proba(
        test[MODEL_FEATURES]
    )[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(
            test["churn"],
            probabilities,
        ),
        "pr_auc": average_precision_score(
            test["churn"],
            probabilities,
        ),
        "precision": precision_score(
            test["churn"],
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            test["churn"],
            predictions,
            zero_division=0,
        ),
    }


def evaluate_recency_baseline(
    test: pd.DataFrame,
    threshold_days: float = 60.0,
) -> dict:
    scores = test["recency_days"].astype(float)

    predictions = (
        scores >= threshold_days
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            test["churn"],
            scores,
        ),
        "pr_auc": average_precision_score(
            test["churn"],
            scores,
        ),
        "precision": precision_score(
            test["churn"],
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            test["churn"],
            predictions,
            zero_division=0,
        ),
    }

def evaluate_thresholds(
    model,
    test: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

    probabilities = model.predict_proba(
        test[MODEL_FEATURES]
    )[:, 1]

    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    test["churn"],
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    test["churn"],
                    predictions,
                    zero_division=0,
                ),
                "flagged_customers_pct": predictions.mean(),
            }
        )

    return pd.DataFrame(rows)

def train_random_forest(
    train: pd.DataFrame,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[MODEL_FEATURES],
        train["churn"],
    )

    return model

def train_gradient_boosting(
    train: pd.DataFrame,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42,
    )

    model.fit(
        train[MODEL_FEATURES],
        train["churn"],
    )

    return model

def train_xgboost(
    train: pd.DataFrame,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[MODEL_FEATURES],
        train["churn"],
    )

    return model

def predict_churn_probability(
    model,
    features: pd.DataFrame,
) -> pd.Series:
    probabilities = model.predict_proba(
        features[MODEL_FEATURES]
    )[:, 1]

    return pd.Series(
        probabilities,
        index=features.index,
        name="churn_probability",
    )