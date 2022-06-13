from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    split_X = np.array_split(X, cv)
    split_y = np.array_split(y, cv)
    train_score, v_score = [], []
    for i in range(cv):
        v_set = split_X.pop(i)
        v_labels = split_y.pop(i)
        train_X = flatten_2D_array(split_X)
        train_y = flatten_2D_array(split_y)
        estimator.fit(train_X, train_y)
        predicted_train = estimator.predict(train_X)
        predicted_v = estimator.predict(v_set)
        train_score.append(scoring(train_y, predicted_train))
        v_score.append(scoring(v_labels, predicted_v))
        split_X.insert(i, v_set)
        split_y.insert(i, v_labels)
    return float(np.mean(train_score)), float(np.mean(v_score))


def flatten_2D_array(arr):
    new_arr = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            new_arr.append(arr[i][j])
    return np.array(new_arr)
