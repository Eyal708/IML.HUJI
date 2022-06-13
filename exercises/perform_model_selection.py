from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    low, high = -1.2, 2
    mu = 0
    title_txt = "(noise = " + str(noise) + "  n_samples = " + str(n_samples) + ")"
    samples = np.linspace(low, high, n_samples)
    noises = np.random.normal(mu, noise, n_samples)
    true_labels = np.apply_along_axis(lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2), 0, samples)
    noise_labels = true_labels + noises
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(samples), pd.Series(noise_labels), 2 / 3)
    train_X = np.array(train_X).reshape(train_X.shape[0], )
    test_X = np.array(test_X).reshape(test_X.shape[0], )
    train_y = np.array(train_y).reshape(train_y.shape[0], )
    test_y = np.array(test_y).reshape(test_y.shape[0], )
    fig_1 = go.Figure([go.Scatter(name="True model", x=samples, y=true_labels, mode="markers"),
                       go.Scatter(name="Train samples", x=train_X, y=np.array(train_y), mode="markers"),
                       go.Scatter(name="Test samples", x=test_X, y=test_y, mode="markers")])
    fig_1.update_layout(title="True model compared to noisy train and test sets" + title_txt
                        , xaxis_title="Feature", yaxis_title="Label")
    fig_1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = list(range(0, 11))
    train_errors, val_errors = [], []
    for k in range(11):
        estimator = PolynomialFitting(k)
        errors = cross_validate(estimator, train_X, train_y, mean_square_error)
        train_errors.append(errors[0])
        val_errors.append(errors[1])
    fig_2 = go.Figure([go.Scatter(name="Train errors", x=degrees, y=train_errors),
                       go.Scatter(name="Validation errors", x=degrees, y=val_errors)])
    fig_2.update_layout(title="Train and validation errors over different polynomial degrees" + title_txt,
                        xaxis_title="Polynomial degree", yaxis_title="Error")
    fig_2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = int(np.argmin(val_errors, 0))
    model = PolynomialFitting(best_k)
    model.fit(train_X, train_y)
    loss = model.loss(test_X, test_y)
    print("Results for noise = " + str(noise) + " and n_samples = " + str(n_samples))
    print("Best k:", best_k, "\nTest loss:", round(loss, 2), " Validation loss:", round(np.min(val_errors), 2))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    #train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), n_samples / X.shape[0])
    train_X = X[0:50, :]
    test_X = X[50:, :]
    train_y = y[0:50]
    test_y = y[50:]
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    low_ridge, high_ridge = 0, 5
    low_lasso, high_lasso = 0, 5
    ridge_lam_values = np.linspace(low_ridge, high_ridge, n_evaluations)
    lasso_lam_values = np.linspace(low_lasso, high_lasso, n_evaluations)
    title_txt = "Train and validation errors over different values of lambda for Lasso/Ridge"
    lasso_train_err, lasso_val_err, ridge_train_err, ridge_val_err = [], [], [], []
    for i in range(n_evaluations):
        lasso = Lasso(lasso_lam_values[i])
        ridge = RidgeRegression(ridge_lam_values[i])
        lasso_errors = cross_validate(lasso, np.array(train_X), np.array(train_y), mean_square_error)
        ridge_errors = cross_validate(ridge, np.array(train_X), np.array(train_y), mean_square_error)
        lasso_train_err.append(lasso_errors[0])
        lasso_val_err.append(lasso_errors[1])
        ridge_train_err.append(ridge_errors[0])
        ridge_val_err.append(ridge_errors[1])
    fig_3 = go.Figure(
        [go.Scatter(name="Ridge: Train error", x=ridge_lam_values, y=ridge_train_err, mode="markers"),
         go.Scatter(name="Ridge: Validation error", x=ridge_lam_values, y=ridge_val_err, mode="markers"),
         go.Scatter(name="Lasso: Train error", x=lasso_lam_values, y=lasso_train_err, mode="markers"),
         go.Scatter(name="Lasso: Validation error", x=lasso_lam_values, y=lasso_val_err, mode="markers")])
    fig_3.update_layout(title=title_txt, xaxis_title="Lambda value", yaxis_title="Error")
    fig_3.show()
    min_ind_lasso, min_ind_ridge = np.argmin(lasso_val_err), np.argmin(ridge_val_err)
    best_reg_lasso, best_reg_ridge = lasso_lam_values[min_ind_lasso], ridge_lam_values[min_ind_ridge]
    print("Q7\nBest lambda for Lasso:", best_reg_lasso, "\nBest lambda for Ridge:", best_reg_ridge)
    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    lasso = Lasso(best_reg_lasso).fit(train_X, train_y)
    ridge = RidgeRegression(best_reg_ridge).fit(np.array(train_X), np.array(train_y))
    ls = LinearRegression().fit(np.array(train_X), np.array(train_y))
    lasso_err = mean_square_error(lasso.predict(test_X), np.array(test_y))
    ridge_err, ls_err = ridge.loss(np.array(test_X), np.array(test_y)), ls.loss(np.array(test_X), np.array(test_y))
    print("(Q8)Errors for each algorithm:\nLasso: " + str(lasso_err) + " Ridge: " + str(ridge_err) +
          " Least squares: " + str(ls_err))


# raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Q1-3
    select_polynomial_degree()
    # Q4
    select_polynomial_degree(noise=0)
    # Q5
    select_polynomial_degree(n_samples=1500, noise=10)
    # Q 6-8
    select_regularization_parameter()
