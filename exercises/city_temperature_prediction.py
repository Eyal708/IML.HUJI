import datetime

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=["Date"]).drop_duplicates().dropna()
    return pre_process_data(full_data)

    # raise NotImplementedError()


def pre_process_data(full_data):
    """
    pre process data- dump invalid values
    """
    clean_data = full_data[(full_data["Temp"] > -10)]

    return clean_data


def q2_plots(df: pd.DataFrame):
    day_israel = df["DayOfYear"]
    temp_israel = df["Temp"]
    year_israel = df["Year"].astype(str)
    fig_1 = px.scatter(x=day_israel, y=temp_israel, color=year_israel, labels={"x": "Day Of Year",
                                                                               "y": "Avg Temp", "color": "Year"},
                       title="Average temp in Israel by day of year",
                       color_discrete_sequence=px.colors.qualitative.Light24)
    fig_1.show()
    grouped_data = israel_data.groupby("Month").agg({"Temp": np.std})
    fig_2 = px.bar(x=list(range(1, 13, 1)), y=grouped_data["Temp"], labels={"x": "Month",
                                                                            "y": "STD"},
                   title="Std of temp by month(Israel)", )
    fig_2.show()


def q3_plot(df: pd.DataFrame):
    grouped_data = df.groupby(["Country", "Month"], as_index=False).agg({"Temp": ["std", "mean"]})
    fig = px.line(x=grouped_data["Month"], y=grouped_data[("Temp", "mean")], color=grouped_data["Country"],
                  labels={"x": "Month", "y": "Mean Temp", "color": "Country"},
                  error_y=grouped_data[("Temp", "std")], error_y_minus=grouped_data[("Temp", "std")],
                  title="Mean temp by month for different countries")
    fig.show()


def q4(df: pd.DataFrame):
    deg_loss_dict = {}
    features = df["DayOfYear"]
    response = df["Temp"]
    train_X, train_y, test_X, test_y = split_train_test(features, response)
    for k in range(1, 11, 1):
        estimator = PolynomialFitting(k)
        estimator.fit(np.array(train_X).flatten(), np.array(train_y))
        deg_loss_dict[k] = np.round(estimator.loss(np.array(test_X).flatten(), np.array(test_y)), 2)
    print("Error recorded for each value of k is:")
    print(deg_loss_dict)
    fig = px.bar(x=list(deg_loss_dict.keys()), y=list(deg_loss_dict.values()),
                 labels={"x": "Degree", "y": "Loss"},
                 title="MSE of model fitted over different polynomial degrees")
    fig.show()


def q5(full_data: pd.DataFrame, isr_data: pd.DataFrame, k: int):
    train_X = isr_data["DayOfYear"]
    train_y = isr_data["Temp"]
    estimator = PolynomialFitting(k)
    estimator.fit(np.array(train_X).flatten(), np.array(train_y))
    country_dict = {"Jordan": 0, "The Netherlands": 0, "South Africa": 0}
    for country in list(country_dict.keys()):
        data_set = full_data[(full_data["Country"] == country)]
        features = data_set["DayOfYear"]
        response = data_set["Temp"]
        country_dict[country] = estimator.loss(np.array(features).flatten(), response)
    fig = px.bar(x=list(country_dict.keys()), y=list(country_dict.values()), color=list(country_dict.keys()),
                 labels={"x": "Country", "y": "Loss", "color": "Country"},
                 title="Loss over model fitted with k=" + str(k))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("C:/Users/eyal7/OneDrive/GitHub/IML.HUJI/datasets/City_Temperature.csv")
    dates = data["Date"]
    day_of_year = [date.timetuple().tm_yday for date in dates]
    data["DayOfYear"] = day_of_year
    israel_data = data[(data["Country"] == "Israel")]
    # Question 2 - Exploring data for specific country
    q2_plots(israel_data)
    # raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    q3_plot(data)
    # raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    q4(israel_data)

    # raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    chosen_k = 5
    q5(data, israel_data, chosen_k)

    # raise NotImplementedError()
