from __future__ import annotations

import time
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_err = np.inf
        signs = [1, -1]
        features = list(range(X.shape[1]))
        signs_features = product(signs, features)  # all possible combinations of signs X features
        for sign, feat in signs_features:
            thr, err = self._find_threshold(X[:, feat], y, sign)  # find best threshold  for each sign and feat
            if err < min_err:  # set attributes if we found a better split
                self.threshold_ = thr
                self.j_ = feat
                self.sign_ = sign
                min_err = err

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        values = X[:, self.j_]  # column for feature best suited for split found in fit
        return np.where(values < self.threshold_, -1 * self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        min_err, best_thr = np.inf, 0
        values = values.reshape(values.shape[0], 1)
        labels = labels.reshape(labels.shape[0], 1)
        joined = np.concatenate((values, labels), 1)
        joined = joined[joined[:, 0].argsort()]
        values = joined[:, 0]
        labels = joined[:, 1]
        for i in range(values.shape[0]):
            pred_labels = np.concatenate((np.repeat(-1 * sign, i), np.repeat(sign, values.shape[0] - i)), 0)
            err = DecisionStump.misclassification_error(labels, np.array(pred_labels))  # calculate error
            if err < min_err:  # if a better threshold is found, save it, and it's error
                min_err = err
                best_thr = values[i]
        return best_thr, min_err

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
        pred_labels = self.predict(X)
        return DecisionStump.misclassification_error(y, pred_labels)

    @staticmethod
    def misclassification_error(y_true, y_pred):
        """
        return weighted loss over predicted labels
        """
        # err = 0
        # for i in range(y_true.shape[0]):
        #     if np.sign(y_true[i]) != np.sign(y_pred[i]):
        #         err += np.abs(y_true[i])
        loss = np.sum(np.where(np.sign(y_true) != np.sign(y_pred), np.abs(y_true), 0))
        return float(loss)
