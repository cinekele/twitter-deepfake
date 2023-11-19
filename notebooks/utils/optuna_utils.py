import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from optuna import Trial
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier


def instantiate_lgbm(trial: Trial) -> LGBMClassifier:
    params = {
        "boosting_type": trial.suggest_categorical('lgbm_boosting_type', ['gbdt', 'dart']),
        "max_depth": trial.suggest_int('lgbm_max_depth', 1, 15),
        "n_estimators": trial.suggest_int('lgbm_n_estimators', 10, 500, log=True),
        "subsample": trial.suggest_float('lgbm_subsample', 0.6, 1),
        'n_jobs': 6
    }
    return LGBMClassifier(**params)


def instantiate_xgb(trial: Trial) -> XGBClassifier:
    params = {
        "booster": trial.suggest_categorical('xgb_booster', ['gbtree', 'dart']),
        "max_depth": trial.suggest_int('xgb_max_depth', 1, 15),
        "n_estimators": trial.suggest_int('xgb_n_estimators', 10, 500, log=True),
        "subsample": trial.suggest_float('xgb_subsample', 0.6, 1),
        'n_jobs': 6
    }
    return XGBClassifier(**params)


def instantiate_rf(trial: Trial) -> RandomForestClassifier:
    params = {
        "max_depth": trial.suggest_int('rf_max_depth', 1, 15),
        "n_estimators": trial.suggest_int('rf_n_estimators', 10, 500, log=True),
        "criterion": trial.suggest_categorical('rf_criterion', ['gini', 'entropy', 'log_loss']),
        "min_samples_split": trial.suggest_float('rf_min_samples_split', 0.01, 0.1),
        'n_jobs': 6
    }
    return RandomForestClassifier(**params)


def instantiate_svc(trial: Trial) -> SVC:
    params = {
        "kernel": trial.suggest_categorical('svc_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        "C": trial.suggest_float('svc_C', 1e-2, 1e2, log=True)
    }
    return SVC(**params)


def instantiate_lr(trial: Trial) -> LogisticRegression:
    params = {
        "solver": 'saga',
        "penalty": trial.suggest_categorical('lr_penalty', ['l1', 'l2']),
        "C": trial.suggest_float('lr_C', 1e-2, 1e2, log=True),
        "n_jobs": 6
    }
    return LogisticRegression(**params)


def instantiate_tfidf(trial: Trial) -> TfidfVectorizer:
    ngram = trial.suggest_categorical('tfidf_ngram_range', ["unigram", "digram", "trigram"])
    if ngram == "unigram":
        ngram_range = (1, 1)
    elif ngram == "digram":
        ngram_range = (1, 2)
    else:
        ngram_range = (1, 3)
    params = {
        "max_features": trial.suggest_int('tfidf_max_features', 1000, 10000, log=True),
        "ngram_range": ngram_range,
        "max_df": trial.suggest_float('tfidf_max_df', 0.8, 1.0),
        "min_df": trial.suggest_float('tfidf_min_df', 0.0, 0.2),
    }
    return TfidfVectorizer(**params)


MODELS = {
    'LGBM': instantiate_lgbm,
    'XGB': instantiate_xgb,
    'RF': instantiate_rf,
    'SVC': instantiate_svc,
    'LR': instantiate_lr
}

ENCODERS = {
    'TFIDF': instantiate_tfidf
}


def objective(trial: Trial, X: pd.DataFrame, y: pd.Series, n_splits: int, model: str, encoder: str,
              random_state: int = 42) -> float:
    type_of_model = MODELS[model]
    type_of_encoder = ENCODERS[encoder]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = type_of_model(trial)
    encoder = type_of_encoder(trial)
    pipeline = Pipeline([
        ('encoder', encoder),
        ('model', model)
    ])
    scorer = make_scorer(balanced_accuracy_score)
    scores = cross_val_score(pipeline, X, y, scoring=scorer, cv=skf)

    return np.mean(scores)
