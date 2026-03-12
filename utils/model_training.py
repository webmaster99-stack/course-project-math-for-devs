from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np


def train_model(model: BaseEstimator, train_set_features: np.ndarray, train_set_labels: np.ndarray) -> BaseEstimator:
    """
    Train a model with chosen predictor features and target labels

    :param model: BaseEstimator - The scikit-learn estimator to train
    :param train_set_features: np.ndarray - A 2d matrix of the predictor features
    :param train_set_labels: np.ndarray - A 1D array of the target labels
    """
    model.fit(train_set_features, train_set_labels)
    return model


def get_predictions(model: BaseEstimator, test_set_features: np.ndarray) -> np.ndarray:
    """
    Get predictions from a trained model

    :param model: BaseEstimator - The already trained scikit-learn estimator
    :param test_set_features: np.ndarray - A 2d matrix of the predictor features from the test set

    :return: np.ndarray - An array of the predicted labels
    """
    return model.predict(test_set_features)


SCORING_METHODS = {
    "accuracy": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score, average="weighted")
}


def evaluate_model(model: BaseEstimator, features: np.ndarray, labels: np.ndarray, scoring_method: str) -> None:
    """
    Evaluates a model on test or train data

    :param model: BaseEstimator - A trained scikit-learn estimator
    :param features: np.ndarray - A matrix of the chosen predictor features
    :param labels: np.ndarray - An array of target labels 
    """
    # predictions = get_predictions(model, features)
    scorer = SCORING_METHODS[scoring_method]
    score = scorer(model, X=features, y_true=labels)

    print(f"{model}'s {scoring_method} score")
    print(f"{score:.4f}")
