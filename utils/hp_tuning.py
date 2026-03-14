from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from typing import Any, Callable
import numpy as np


def tune_model_hyperparameters( 
    model: BaseEstimator, 
    param_space: dict[str, list[Any]], 
    scoring_metric: str | Callable, 
    cross_val_splits: int, 
    train_set_features: np.ndarray, 
    train_set_labels: np.ndarray
) -> RandomizedSearchCV:
    """
    Tune model hyperparameters using GridSearchCV

    :param model: BaseEstimator - A scikit-learn estimator
    :param param_space: dict - The chosen hyperparameters and their values
    :param scoring_metric: str | Callable - The evaluation metric to evaluate the estimator
    :param cross_val_splits: int - The number of cross-validation splits to perform
    :param train_set_features: np.ndarray - A 2d matrix of the predictor features
    :param train_set_labels: np.ndarray - A 1D array of the target labels

    :return: RandomizedSearchCV
    """
    rand_search = RandomizedSearchCV(model, param_space, scoring=scoring_metric, cv=cross_val_splits, n_jobs=-1, n_iter=10)
    rand_search.fit(train_set_features, train_set_labels)

    return rand_search