import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data, response = load_dataset("../datasets/" + f)

        # raise NotImplementedError()

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_callback(perceptron: Perceptron, X: np.ndarray, y: int):
            """
            callback function to calculate the loss after each perceptron algorithm iteration
            """
            losses.append(perceptron.loss(data, response))

        perceptron = Perceptron(callback=loss_callback,include_intercept=False)
        perceptron.fit(data, response)

        # raise NotImplementedError()
        # Plot figure of loss as function of fitting iteration
        iterations = [i + 1 for i in range(len(losses))]
        figure = px.line(x=iterations, y=losses, labels={"x": "Iterations", "y": "Loss", },
                         title="Loss over " + n + " data in relation to numer of iterations of Perceptron algorithm")
        figure.show()
        # raise NotImplementedError()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data, response = load_dataset("../datasets/" + f)
        # Fit models and predict over training set
        lda = LDA().fit(data, response)
        naive = GaussianNaiveBayes().fit(data, response)
        lda_pred = lda.predict(data)
        naive_pred = naive.predict(data)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy

        # Create subplots
        from IMLearn.metrics import accuracy
        lda_acc, naive_acc = accuracy(response, lda_pred), accuracy(response, naive_pred)
        lda_centers, naive_centers = lda.mu_, naive.mu_
        titles = ["Model: Naive Gaussian   Accuracy: " + str(naive_acc), "Model: LDA   Accuracy: " + str(lda_acc)]
        figure = go.Figure(make_subplots(rows=1, cols=2, subplot_titles=titles))

        # Add traces for data-points setting symbols and colors
        figure.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers", name="Naive",
                                    marker=dict(color=naive_pred, symbol=response)), row=1, col=1)
        figure.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers", name="LDA",
                                    marker=dict(color=lda_pred, symbol=response)), row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        figure.add_trace(go.Scatter(x=lda_centers[:, 0], y=lda_centers[:, 1], showlegend=False, mode="markers"
                                    , marker=dict(color="black", symbol="x", size=10)), col=1, row=1)
        figure.add_trace(go.Scatter(x=naive_centers[:, 0], y=naive_centers[:, 1], showlegend=False, mode="markers"
                                    , marker=dict(color="black", symbol="x", size=10)), col=2, row=1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for k in lda.classes_:
            lda_ellipse = get_ellipse(lda.mu_[k], lda.cov_)
            naive_ellipse = get_ellipse(naive.mu_[k], np.diag(naive.vars_[k]))
            figure.add_trace(naive_ellipse, row=1, col=1)
            figure.add_trace(lda_ellipse, row=1, col=2)

        figure.update_xaxes(title_text="feature1")
        figure.update_yaxes(title_text="feature2")
        figure.update_layout(title_text="Data Set: " + f, title_x=0.5, title_y=1, height=500)
        figure.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    # compare_gaussian_classifiers()
    # quiz
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(8, 1)
    y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    naive = GaussianNaiveBayes()
    naive.fit(X, y)
    print(naive.pi_[0])
    print(naive.mu_[1])
    X_2 = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    y_2 = np.array([0, 0, 1, 1, 1, 1])
    naive_2 = GaussianNaiveBayes()
    naive_2.fit(X_2, y_2)
    print(naive_2.vars_)
