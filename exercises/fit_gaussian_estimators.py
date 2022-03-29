import plotly
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    m, interval, meo, sigma = 1000, 10, 10, 1
    X = np.random.normal(meo, sigma, m)
    estimator = UnivariateGaussian()
    estimator.fit(X)
    print("Question 1:")
    print("(", estimator.mu_, ",", estimator.var_, ")")

    # Question 2 - Empirically showing sample mean is consistent
    num_of_samples = list(range(meo, m + 1, interval))
    # mean_estimator = [abs(estimator.fit(np.random.choice(X, i)).mu_ - meo) for i in num_of_samples]
    mean_estimator = [abs(estimator.fit(X[:i]).mu_ - meo) for i in num_of_samples]
    fig_1 = px.scatter(x=num_of_samples, y=mean_estimator, labels={"x": "Number of Samples", "y":
        "Distance from true expectancy"}, height=500, width=600, title="Distance of estimated expectancy"
                                                                       " from true value")
    fig_1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    estimator.fit(X)
    PDF = estimator.pdf(X)
    fig_2 = px.scatter(x=X, y=PDF, labels={"x": "sample value", "y": "pdf"}, height=500, width=800,
                       title="Samples drawn from ~N(10,1) and their pdf")
    fig_2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    n = 1000
    mult_est = MultivariateGaussian()
    X = np.random.multivariate_normal(mu, sigma, n)
    mult_est.fit(X)
    print("Question 4:")
    print(mult_est.mu_)
    print(mult_est.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    values = np.transpose(np.array([np.repeat(f1, len(f1)), np.tile(f1, len(f1))]))  # cartesean product of f1
    log_likelihood = np.array([MultivariateGaussian.log_likelihood(np.array([row[0], 0, row[1], 0]), sigma, X)
                               for row in values]).reshape(f1.shape[0], f1.shape[0])
    fig3 = px.imshow(log_likelihood, x=f1, y=f1, labels={"x": "feature 3", "y": "feature 1", "color": "log likelihood"}
                     , title="Log likelihood for expectancy [f1,0,f3,0]")

    fig3.show()

    # Question 6 - Maximum likelihood
    form = "{:.3f}"
    max_liklihood = np.max(log_likelihood)
    helper = np.where(log_likelihood == max_liklihood)
    max_indices = (helper[0][0], helper[1][0])
    print("Question 6:")
    print("f1:", form.format(f1[max_indices[0]]), " f3:", form.format(f1[max_indices[1]]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

    #This is just the code for solving the quiz
    # a = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #               -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    # first = UnivariateGaussian.log_likelihood(1, 1, a)
    # second = UnivariateGaussian.log_likelihood(10, 1, a)
    # print("first is ", first)
    # print("second is ", second)
