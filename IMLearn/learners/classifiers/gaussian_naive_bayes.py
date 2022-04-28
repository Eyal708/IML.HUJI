from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
import math


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.pi_ = counts / y.shape[0]
        self.mu_ = np.zeros(shape=(self.classes_.shape[0], X.shape[1]))
        self.vars_ = np.zeros(shape=(self.classes_.shape[0], X.shape[1]))
        for i in range(self.classes_.shape[0]):
            group = X[y == self.classes_[i]]
            self.mu_[i] = np.mean(group, axis=0)
            self.vars_[i] = np.var(group, axis=0)

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
        return self.likelihood(X).argmax(axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihood_mat = []
        for sample in X:
            row = []
            for i in range(self.classes_.shape[0]):
                cov_mat = np.diag(self.vars_[i])
                cov_mat_inv = np.linalg.inv(cov_mat)
                Z = math.sqrt(math.pow(2 * math.pi, X.shape[1]) * np.linalg.det(cov_mat))
                # a_k = cov_mat_inv @ self.mu_[i, :].T
                # b_k = \
                #     math.log(self.pi_[i]) - 0.5 * np.linalg.multi_dot([self.mu_[i, :], cov_mat_inv, self.mu_[i, :]])
                l = self.pi_[i] * (1 / Z) * math.exp(
                      np.linalg.multi_dot([-0.5*(sample - self.mu_[i, :]), cov_mat_inv, sample - self.mu_[i, :]]))
                # row.append(a_k.T @ sample + b_k)
                row.append(l)
            likelihood_mat.append(row)
        return np.array(likelihood_mat)

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
        predictions = self.predict(X)
        return misclassification_error(y, predictions)
