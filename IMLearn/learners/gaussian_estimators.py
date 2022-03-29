from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        if self.biased_:
            m = X.shape[0]
        else:
            m = X.shape[0] - 1
        self.mu_ = np.mean(X)
        self.var_ = np.sum(np.power(X - self.mu_, 2)) / m
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        args = [self.mu_, self.var_]
        B = np.apply_along_axis(self.normal_dist_density, 0, [X], *args).reshape(X.shape[0], )
        return B

    @staticmethod
    def normal_dist_density(x: float, mu: float, sigma: float) -> float:
        power = -1 * (x - mu) ** 2 / 2 * sigma
        return 1 / (2 * math.pi * sigma) ** 0.5 * math.exp(power)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # args = (mu, sigma)
        # A = np.apply_along_axis(UnivariateGaussian.normal_dist_density, 0, [X], *args).reshape(X.shape[0], )
        # return math.log(np.product(A))
        sum = np.sum((X - mu) ** 2)
        m = X.shape[0]
        return math.log(1 / ((2 * math.pi * math.sqrt(sigma)) ** (m / 2))) - 1 / (2 * sigma) * sum


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        # """
        self.mu_ = np.mean(X, 0)
        # # self.cov_ = np.matmul(np.transpose(X - self.mu_), X - self.mu_) * 1 / (X.shape[0] - 1)
        self.cov_ = np.cov(np.transpose(X))
        self.fitted_ = True
        return self


    @staticmethod
    def mult_gaus_density(X: np.ndarray, sigma: np.ndarray, meu: np.ndarray):
        d = meu.shape[0]
        power = np.linalg.multi_dot([np.transpose(X - meu), np.linalg.inv(sigma), X - meu]) * (-0.5)
        return 1 / ((2 * math.pi) ** d * np.linalg.det(sigma)[1]) * math.exp(power)

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        args = (self.cov_, self.mu_)
        B = np.apply_along_axis(MultivariateGaussian.mult_gaus_density, 1, X, *args)
        return B

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        d = X.shape[1]
        n = X.shape[0]
        sum = np.sum(np.multiply(np.transpose(X - mu), np.matmul(np.linalg.inv(cov), np.transpose(X - mu))))
        return n / 2 * (math.log(1) - math.log((math.pi * 2) ** d) + np.linalg.slogdet(cov)[1]) - 0.5 * sum
     #closed expression to calculate log likelihood
# from __future__ import annotations
# import numpy as np
#
#
# class UnivariateGaussian:
#     """
#     Class for univariate Gaussian Distribution Estimator
#     """
#     def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
#         """
#         Estimator for univariate Gaussian mean and variance parameters
#
#         Parameters
#         ----------
#         biased_var : bool, default=False
#             Should fitted estimator of variance be a biased or unbiased estimator
#
#         Attributes
#         ----------
#         fitted_ : bool
#             Initialized as false indicating current estimator instance has not been fitted.
#             To be set as True in `UnivariateGaussian.fit` function.
#
#         mu_: float
#             Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
#             function.
#
#         var_: float
#             Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
#             function.
#         """
#         self.biased_ = biased_var
#         self.fitted_, self.mu_, self.var_ = False, None, None
#
#     def fit(self, X: np.ndarray) -> UnivariateGaussian:
#         """
#         Estimate Gaussian expectation and variance from given samples
#
#         Parameters
#         ----------
#         X: ndarray of shape (n_samples, )
#             Training data
#
#         Returns
#         -------
#         self : returns an instance of self.
#
#         Notes
#         -----
#         Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
#         estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
#         """
#         self.mu_ = UnivariateGaussian._calc_mu(X)
#         if self.biased_:
#             self.var_ = UnivariateGaussian._calc_unbiased_var(self.mu_, X)
#         else:
#             self.var_ = UnivariateGaussian._calc_biased_var(self.mu_, X)
#
#         self.fitted_ = True
#         return self
#
#     def pdf(self, X: np.ndarray) -> np.ndarray:
#         """
#         Calculate PDF of observations under Gaussian model with fitted estimators
#         Parameters
#         ----------
#         X: ndarray of shape (n_samples, )
#             Samples to calculate PDF for
#         Returns
#         -------
#         pdfs: ndarray of shape (n_samples, )
#             Calculated values of given samples for PDF function of N(mu_, var_)
#         Raises
#         ------
#         ValueError: In case function was called prior fitting the model
#         """
#         if not self.fitted_:
#             raise ValueError("Estimator must first be fitted before calling `pdf` function")
#         return UnivariateGaussian._calc_pdf(self.mu_, self.var_, X)
#
#     @staticmethod
#     def _calc_pdf(mu: float, var: float, X: np.ndarray) -> np.ndarray:
#         """
#         Calculate PDF of observations under Gaussian model if random variable X~N(mu, var)
#         Parameters
#         ----------
#         mu : float
#             Expectation of Gaussian
#         var : float
#             Variance of Gaussian
#         X: ndarray of shape (n_samples, )
#             Samples to calculate PDF for
#         Returns
#         -------
#         pdfs: ndarray of shape (n_samples, )
#             Calculated values of given samples for PDF function of N(mu_, var_)
#         """
#         scalar1 = 1 / np.sqrt(2 * np.pi * var)
#         scalar2 = -2 * var
#         return scalar1 * np.exp((np.power((X - mu), 2)) / scalar2)
#
#     @staticmethod
#     def _calc_mu(X: np.ndarray) -> float:
#         """
#         Calculate mean estimator of observations under Gaussian model
#
#         Parameters
#         ----------
#         X: ndarray of shape (n_samples, )
#             Samples to calculate mean for
#
#         Returns
#         -------
#         mean: float
#            mean calculated
#         """
#         if len(X) == 0:  # Don't divide by 0, if the array is empty return mean = 0
#             return 0
#         return X.sum() / len(X)
#
#     @staticmethod
#     def _calc_biased_var(mu: float, X: np.ndarray) -> float:
#         """
#         Calculate variance estimator of observations under Gaussian model for biased estimator
#
#         Parameters
#         ----------
#         mu : float
#             Expectation of Gaussian
#         X: ndarray of shape (n_samples, )
#             Samples to calculate mean for
#
#         Returns
#         -------
#         variance: float
#            variance calculated
#         """
#         if len(X) == 0:  # Don't divide by 0, if the array is empty return var = 0
#             return 0
#         scalar = 1 / len(X)
#         return scalar * np.sum(np.power((X - mu), 2))
#
#     @staticmethod
#     def _calc_unbiased_var(mu: float, X: np.ndarray) -> float:
#         """
#         Calculate variance estimator of observations under Gaussian model for unbiased estimator
#
#         Parameters
#         ----------
#         mu : float
#             Expectation of Gaussian
#         X: ndarray of shape (n_samples, )
#             Samples to calculate mean for
#
#         Returns
#         -------
#         variance: float
#            variance calculated
#         """
#         if len(X) <= 1:  # Don't divide by 0, if the array is empty or single sample return var = 0
#             return 0
#         scalar = 1 / (len(X) - 1)
#         return scalar * np.sum(np.power((X - mu), 2))
#
#     @staticmethod
#     def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
#         """
#         Calculate the log-likelihood of the data under a specified Gaussian model
#
#         Parameters
#         ----------
#         mu : float
#             Expectation of Gaussian
#         sigma : float
#             Variance of Gaussian
#         X : ndarray of shape (n_samples, )
#             Samples to calculate log-likelihood with
#
#         Returns
#         -------
#         log_likelihood: float
#             log-likelihood calculated
#         """
#         return np.sum(np.log(UnivariateGaussian._calc_pdf(mu, sigma, X)))
#
#
# class MultivariateGaussian:
#     """
#     Class for multivariate Gaussian Distribution Estimator
#     """
#     def __init__(self):
#         """
#         Initialize an instance of multivariate Gaussian estimator
#
#         Attributes
#         ----------
#         fitted_ : bool
#             Initialized as false indicating current estimator instance has not been fitted.
#             To be set as True in `MultivariateGaussian.fit` function.
#
#         mu_: ndarray of shape (n_features,)
#             Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
#             function.
#
#         cov_: ndarray of shape (n_features, n_features)
#             Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
#             function.
#         """
#         self.mu_, self.cov_ = None, None
#         self.fitted_ = False
#
#     def fit(self, X: np.ndarray) -> MultivariateGaussian:
#         """
#         Estimate Gaussian expectation and covariance from given samples
#
#         Parameters
#         ----------
#         X: ndarray of shape (n_samples, n_features)
#             Training data
#
#         Returns
#         -------
#         self : returns an instance of self
#
#         Notes
#         -----
#         Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
#         Then sets `self.fitted_` attribute to `True`
#         """
#
#         self.mu_ = MultivariateGaussian._calc_mu(X)
#         self.cov_ = MultivariateGaussian._calc_unbiased_cov(self.mu_, X)
#         self.fitted_ = True
#         return self
#
#     def pdf(self, X: np.ndarray):
#         """
#         Calculate PDF of observations under Gaussian model with fitted estimators
#
#         Parameters
#         ----------
#         X: ndarray of shape (n_samples, n_features)
#             Samples to calculate PDF for
#
#         Returns
#         -------
#         pdfs: ndarray of shape (n_samples, )
#             Calculated values of given samples for PDF function of N(mu_, cov_)
#
#         Raises
#         ------
#         ValueError: In case function was called prior fitting the model
#         """
#         if not self.fitted_:
#             raise ValueError("Estimator must first be fitted before calling `pdf` function")
#         # from scipy.stats import multivariate_normal
#         # var = multivariate_normal(self.mu_, self.cov_)
#         # real_pdf = var.pdf(X)
#
#         # m = X - self.mu_
#         # scalar = 1 / (np.sqrt(np.power(2 * np.pi, len(self.mu_)) * np.linalg.det(self.cov_)))
#         # pdf = np.zeros(len(X))
#         # for i in range(len(pdf)):
#         #     pdf[i] = scalar * np.exp((-1 / 2) * (m[i].T @ np.linalg.inv(self.cov_) @ m[i]))
#         return MultivariateGaussian._calc_pdf(self.mu_, self.cov_, X)
#
#     @staticmethod
#     def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
#         """
#         Calculate the log-likelihood of the data under a specified Gaussian model
#
#         Parameters
#         ----------
#         mu : ndarray of shape (n_features,)
#             Expectation of Gaussian
#         cov : ndarray of shape (n_features, n_features)
#             covariance matrix of Gaussian
#         X : ndarray of shape (n_samples, n_features)
#             Samples to calculate log-likelihood with
#
#         Returns
#         -------
#         log_likelihood: float
#             log-likelihood calculated over all input data and under given parameters of Gaussian
#         """
#
#         return np.float(np.sum(np.log(MultivariateGaussian._calc_pdf(mu, cov, X))))
        # loglikelihood = -0.5 * (
        #         np.log(np.linalg.det(cov))
        #         + np.einsum('...j,jk,...k', X, np.linalg.inv(cov), X)
        #         + len(mu) * np.log(2 * np.pi)
        # )
        # loglikelihood = np.sum(loglikelihood)

        # if a == loglikelihood:
        #     print(a)

        # from scipy.stats import multivariate_normal
        # y = multivariate_normal.pdf(X, mean=mu, cov=cov)
        # y_log = np.log(y)
        # from scipy.stats import norm
        # t, s = norm.fit(X)
        # log_likelihood_2 = np.sum(np.log(norm.pdf(X, t, s)))

        # def calc_loglikelihood(residuals):
        #     return -0.5 * (np.log(np.linalg.det(cov)) + residuals.T.dot(np.linalg.inv(cov)).dot(residuals) + 2 * np.log(
        #         2 * np.pi))
        #
        # m = X - mu
        # b = np.apply_along_axis(calc_loglikelihood, 1, m)
        # c = b.sum()
        # if a == c:
        #     print(a)

        # vals, vecs = np.linalg.eigh(cov)
        # logdet = np.sum(np.log(vals))
        # valsinv = np.array([1. / v for v in vals])
        # # `vecs` is R times D while `vals` is a R-vector where R is the matrix
        # # rank. The asterisk performs element-wise multiplication.
        # U = vecs * np.sqrt(valsinv)
        # rank = len(vals)
        # dev = X - mu
        # # "maha" for "Mahalanobis distance".
        # maha = np.square(np.dot(dev, U)).sum()
        # log2pi = np.log(2 * np.pi)
        # log_l = -0.5 * (rank * log2pi + maha + logdet)
        # if a == log_l:
        #     print(a)
        # return c

    # @staticmethod
    # def _calc_mu(X: np.ndarray) -> np.ndarray:
    #     """
    #     Calculate multivariate sample mean estimator under Gaussian model
    #
    #     Parameters
    #     ----------
    #     X: ndarray of shape (n_samples, n_features)
    #         Training data
    #
    #     Returns
    #     -------
    #     mean: ndarray of shape (n_features,)
    #        mean calculated
    #     """
    #     if len(X) == 0:  # Don't divide by 0, if the array is empty return mean = 0
    #         raise ValueError("Non valid samples were given to fit function, samples size must be > 1")
    #     return np.mean(X, axis=0)
    #
    # @staticmethod
    # def _calc_unbiased_cov(mu: np.ndarray, X: np.ndarray) -> np.ndarray:
    #     """
    #     Calculate covariance multivariate estimator of observations under Gaussian model for unbiased estimator
    #
    #     Parameters
    #     ----------
    #     mu : ndarray of shape (n_features,)
    #         Expectation of Gaussian
    #     X: ndarray of shape (n_samples, n_features)
    #         Training data
    #
    #     Returns
    #     -------
    #     covariance: ndarray of shape (n_features,)
    #        covariance calculated
    #     """
    #     if len(X) <= 1:  # Don't divide by 0, if the array is empty or single sample return var = 0
    #         raise ValueError("Non valid samples were given to fit function, samples size must be > 1")
    #     scalar = 1 / (len(X) - 1)
    #     m = X - mu
    #     try:
    #         cov = np.cov(X.T)  # Reminder: come back here and decide which implementation to choose
    #         # cov = scalar * m.T @ m
    #     except:
    #         try:
    #             cov = scalar * m @ m.T
    #         except:
    #             raise ValueError("Non valid samples were given to pdf")
    #     return cov
    #
    # @staticmethod
    # def _calc_pdf(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> np.ndarray:
    #     """
    #     Calculate PDF of observations under Gaussian model if random variable X~N(mu, var)
    #     Parameters
    #     ----------
    #     mu : ndarray of shape (n_features,)
    #         Expectation of Gaussian
    #     cov : ndarray of shape (n_features, n_features)
    #         covariance matrix of Gaussian
    #     X: ndarray of shape (n_samples, n_features)
    #         Training data
    #     Returns
    #     -------
    #     pdfs: ndarray of shape (n_samples, )
    #         Calculated values of given samples for PDF function of N(mu_, var_)
    #     """
    #     m = X - mu
    #     scalar = 1 / (np.sqrt(np.power(2 * np.pi, len(mu)) * np.linalg.det(cov)))
    #     pdf = np.zeros(len(X))
    #     for i in range(len(pdf)):
    #         pdf[i] = scalar * np.exp((-0.5) * (m[i].T @ np.linalg.inv(cov) @ m[i]))
    #     return pdf
    #
    #
