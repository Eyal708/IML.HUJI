# %%
import time

import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    start_time = time.time()
    symbols = np.array(["square", "x", "circle"])
    mult = [100, 10]

    for itr, noise_val in enumerate(noise):

        (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise_val), \
                                               generate_data(test_size, noise_val)

        # Question 1: Train- and test errors of AdaBoost in noiseless case
        adaboost = AdaBoost(DecisionStump, n_learners)
        adaboost.fit(train_X, train_y)
        train_errors = [adaboost.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)]
        test_errors = [adaboost.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)]
        models_num = list(range(1, n_learners + 1))
        fig_1 = go.Figure([go.Scatter(x=models_num, y=train_errors, name="Train Error"),
                           go.Scatter(x=models_num, y=test_errors, name="Test Error")])
        fig_1.update_layout(xaxis_title="Number of fitted learners", yaxis_title="Error",
                            title="Train & Test Errors Over Number Of Fitted Learners")
        fig_1.show()
        # Question 2: Plotting decision surfaces
        T = [5, 50, 100, 250]
        lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array(
            [-.1, .1])
        fig_2 = make_subplots(rows=2, cols=2, subplot_titles=["Fitted Learners: " + str(i) for i in T],
                              horizontal_spacing=0.01, vertical_spacing=.03)
        for i, n in enumerate(T):
            fig_2.add_traces([decision_surface(adaboost.partial_predict, lims[0], lims[1], n, showscale=False),
                              go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                         marker=dict(color=test_y, symbol=symbols[test_y.astype(int)],
                                                     colorscale=[custom[0], custom[-1]],
                                                     line=dict(color="black", width=1)))],
                             rows=(i // 2) + 1, cols=(i % 2) + 1)

        fig_2.update_layout(title="Decision Boundaries Of Models According To Number Of Fitted Learners",
                            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig_2.show()
        # raise NotImplementedError()
        #
        # Question 3: Decision surface of best performing ensemble
        bst_ens = np.argmin(test_errors) + 1  # number of learners in best ensemble
        bst_acc = accuracy(test_y, adaboost.partial_predict(test_X, bst_ens))
        fig_3 = go.Figure([decision_surface(adaboost.partial_predict, lims[0], lims[1], bst_ens, showscale=False),
                           go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                      marker=dict(color=test_y, symbol=symbols[test_y.astype(int)],
                                                  colorscale=[custom[0], custom[-1]],
                                                  line=dict(color="black", width=1)))])
        fig_3.update_layout(title="Decision Surface For Best Ensemble\n Size: " + str(bst_ens) +
                                  " Accuracy: " + str(bst_acc),
                            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig_3.show()

        # raise NotImplementedError()
        #
        # Question 4: Decision surface with weighted samples
        weights = adaboost.D_[n_learners - 1] / np.max(adaboost.D_[n_learners - 1]) * mult[itr]
        fig_4 = go.Figure([decision_surface(adaboost.partial_predict, lims[0], lims[1], n_learners, showscale=False),
                           go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                      marker=dict(color=train_y, size=weights,
                                                  symbol=symbols[train_y.astype(int)],
                                                  colorscale=[custom[0], custom[-1]],
                                                  line=dict(color="black", width=1)))])
        fig_4.update_layout(title="Adaboost Prediction On Train Set With Weights(Point Size)",
                            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig_4.show()
    print("running time: ", time.time() - start_time)
    # raise NotImplementedError()


def decision_surface(predict, xrange, yrange, T: int, density=120, dotted=False, colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], T)

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
                          marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False),
                          hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False,
                      opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)


if __name__ == '__main__':
    np.random.seed(0)
    noise = [0, 0.4]
    fit_and_evaluate_adaboost(noise)

    # raise NotImplementedError()
