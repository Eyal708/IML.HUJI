import math

import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.metrics.loss_functions import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.models_, self.weights_ = [], np.zeros(self.iterations_)
        self.D_ = np.zeros(self.iterations_ * X.shape[0]).reshape(self.iterations_, X.shape[0])
        self.D_[0] = np.repeat(1 / X.shape[0], X.shape[0])
        for i in range(self.iterations_):
            learner = self.wl_()
            learner.fit(X, y * self.D_[i])
            self.models_.append(learner)
            err = learner.loss(X, y * self.D_[i])
            self.weights_[i] = 0.5 * math.log((1 / err) - 1)
            if i < self.iterations_ - 1:  # we don't need this in the final iteration
                y_pred = learner.predict(X)
                self.D_[i + 1] = np.array([self.D_[i, j] * math.exp(-1 * y[j] * self.weights_[i] * y_pred[j])
                                           for j in range(X.shape[0])])
                self.D_[i + 1] = self.D_[i + 1] / np.sum(self.D_[i + 1])  # normalize

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        return self.partial_predict(X, len(self.models_))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self.partial_loss(X, y, len(self.models_))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        ensemble_pred = np.zeros(X.shape[0])
        for t in range(T):
            y_pred = self.models_[t].predict(X)
            ensemble_pred += self.weights_[t] * y_pred
        return np.sign(ensemble_pred)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.partial_predict(X, T)
        return misclassification_error(y, y_pred)
