import math

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def pre_processing(full_data: pd.DataFrame):
    """
    pre process data and return it
    """
    clean_data = full_data[(full_data["price"] > 0) | (full_data["bedrooms"] > 0) |
                           (full_data["bathrooms"] > 0) | (full_data["sqft_living"] > 0)
                           | (full_data["sqft_lot"] > 0)
                           | (full_data["floors"] > 0) | (full_data["waterfront"].isin(range(0, 2, 1)))
                           | (full_data["view"].isin(range(0, 5, 1)))
                           | (full_data.condition.isin(range(1, 6, 1)))
                           | (full_data["grade"].isin(range(1, 14, 1))) | (full_data["sqft_above"] >= 0)
                           | (full_data["sqft_basement"] >= 0) | (full_data["yr_built"] > 0)
                           | (full_data.yr_renovated > 0) | (full_data.zipcode > 0)
                           | (full_data.sqft_living15 > 0) | (full_data.sqft_lot15 > 0)]
    response = clean_data["price"]
    zip_code_dummy = pd.get_dummies(clean_data.zipcode, drop_first=True)  # give zipcode dummy values
    edited_data = clean_data.drop(["zipcode", "date", "id", "price", "yr_renovated", "yr_built",
                                   "sqft_lot15", "sqft_lot", "long", "lat", "condition"],
                                  axis=1)  # drop columns from data
    edited_data = edited_data.join(zip_code_dummy)  # add dummy values of zip codes to data
    return edited_data, response


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).drop_duplicates().dropna()
    return pre_processing(full_data)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    feature_names_corr = {}
    for col in X.columns:
        p_corr = pearson_corr(np.array(X[col]), y)
        feature_names_corr[col] = p_corr
    for key, value in feature_names_corr.items():
        fig = px.scatter(x=X[key], y=y, labels={"x": "feature value", "y": "house price"},
                         title="feature: " + str(key) + " pearson corr: " + str(value))
        fig.write_image(output_path + "/" + str(key) + ".png")


def pearson_corr(X, Y):
    """
    calculate pearson correlation between two 1D arrays
    """
    cov_X_y = np.cov(X, Y)[0][1]
    sigma_X, sigma_Y = np.std(X), np.std(Y)
    return cov_X_y / (sigma_X * sigma_Y)


def q4_create_dicts(train_X, train_y, test_X, test_y):
    avg_loss_dict, var_loss_dict = {}, {}
    train_set = train_X.join(train_y)
    for p in range(10, 101, 1):
        loss_sum = 0
        loss_vector = []
        for i in range(10):
            train_samples = train_set.sample(frac=p / 100)
            train_sample_data = train_samples.drop(columns=["price"])
            train_sample_response = train_samples["price"]
            estimator = LinearRegression()
            estimator.fit(np.array(train_sample_data), np.array(train_sample_response))
            mean_loss = estimator.loss(np.array(test_X), np.array(test_y))
            loss_sum += mean_loss
            loss_vector.append(mean_loss)
        avg_loss_dict[p] = loss_sum / 10
        var_loss_dict[p] = np.var(loss_vector)
    return avg_loss_dict, var_loss_dict


def q4_create_figure(avg_loss_dict, var_loss_dict):
    # create figure
    error_upper_bound = [avg_loss_dict[key] + 2 * math.sqrt(var_loss_dict[key]) for key in avg_loss_dict]
    error_lower_bound = [avg_loss_dict[key] - 2 * math.sqrt(var_loss_dict[key]) for key in avg_loss_dict]
    # fig = px.scatter(x=avg_loss_dict.keys(), y=avg_loss_dict.values(),
    #                  labels={"x": "% sampled from train data", "y": "average loss over 10 samples"},
    #                  title="Average loss over models fitted with different fractions of training data",
    #                  error_y_minus=error_lower_bound, error_y=error_upper_bound, height=600, width=1200)
    fig = go.Figure([go.Scatter(x=list(avg_loss_dict.keys()), y=list(avg_loss_dict.values()), name="Mean Loss"),
                     go.Scatter(x=list(avg_loss_dict.keys()), y=error_upper_bound,
                                fill=None, name="Upper Bound"),
                     go.Scatter(x=list(avg_loss_dict.keys()), y=error_lower_bound, fill="tonexty",
                                name="Lower Bound")])
    fig.update_layout(xaxis_title="% sampled from train data", yaxis_title="average loss over 10 samples",
                      title="Average loss over models fitted with different fractions of training data")

    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, prices = load_data("../datasets/house_prices.csv")
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, prices, "features_corr")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, prices)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    dict_1, dict_2 = q4_create_dicts(train_X, train_y, test_X, test_y)
    q4_create_figure(dict_1, dict_2)
