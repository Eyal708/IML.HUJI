from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """

    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given samples as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input samples to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        samples = Perceptron.include_intercept(X, self.include_intercept_)  # adjusts the data if needed
        self.coefs_ = np.repeat(0, samples.shape[1])  # initialize coefficients
        self.fitted_ = True
        iteration = 0
        while iteration < self.max_iter_:
            flag = False
            for i in range(samples.shape[0]):
                if y[i] * np.matmul(self.coefs_, samples[i, :]) <= 0:
                    self.coefs_ = self.coefs_ + y[i] * samples[i, :]
                    self.callback_(self, samples[i, :], y[i])
                    flag = True
                    break
            if not flag:  # no mis-classification, so we can break the loop
                break
            iteration += 1

    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        data = Perceptron.include_intercept(X, self.include_intercept_)
        responses = []
        for i in range(data.shape[0]):
            if np.matmul(data[i, :], self.coefs_) < 0:
                responses.append(-1)
            else:
                responses.append(1)
        return np.array(responses)

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
        from ...metrics import misclassification_error
        responses = self.predict(X)
        return misclassification_error(responses, y)

    @staticmethod
    def include_intercept(X, is_included) -> np.ndarray:
        """
        adjusts the data depending if we is_included is true or false.
        if true: adds a columns of "1" to X and returns it(because we want to include an intercept
        if false: returns X as it is(we don't want to include an intercept
        """
        if is_included:
            return np.append(X, np.array([np.repeat(1, X.shape[0])]).T, axis=1)
        return X
