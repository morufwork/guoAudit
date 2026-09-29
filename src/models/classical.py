"""Classical baseline models: thin factory over sklearn/xgboost estimators."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def build_model(name: str, params: dict, seed: int):
    params = dict(params)
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if name == "xgboost":
        return XGBClassifier(random_state=seed, **params)
    if name == "mlp":
        return MLPClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown model: {name}")
