# imports
import numpy
import pandas as pd
import seaborn
import statsmodels.api
from matplotlib import pyplot as plt
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# plotting functions

# CATEGORICAL RESPONSE - CONTINUOUS PREDICTOR


def cat_resp_cont_predictor(feature_name, x):
    print(type(x))

    fig_1 = ff.create_distplot([x], feature_name, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.show()

    """fig = px.histogram(x, feature_name)
    fig.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig.show()"""

    n = 200
    fig_2 = go.Figure()
    for curr_hist, curr_group in zip([x], feature_name):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, n),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_2.show()
    return


# CATEGORICAL RESPONSE - CATEGORICAL PREDICTOR
def cat_resp_cat_predictor(x, y):

    x_2 = [1 if abs(x_) > 1.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 1.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()
    return


# CONTINUOUS RESPONSE - CONTINUOUS PREDICTOR
def cont_resp_cont_predictor(X, y, feature_name):
    fig = px.scatter(x=X, y=y, trendline="ols", labels=feature_name)
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title=feature_name,
        yaxis_title="Response",
    )
    fig.show()
    return


# CONTINUOUS RESPONSE - CATEGORICAL PREDICTOR
def cont_resp_cat_predictor(y, feature_name):

    fig_1 = ff.create_distplot([y], [feature_name], bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.show()

    n = 200
    fig_2 = go.Figure()
    for curr_hist, curr_group in zip([y], [feature_name]):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, n),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_2.show()

    return


# Calculating the different ranking algos

# p-value & t-score (continuous predictors only)
# Regression: Continuous response


def PTCont(x, y, feature_name):

    predictor = statsmodels.api.add_constant(x)
    linear_regression_model = statsmodels.api.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Variable: {feature_name}")
    print(linear_regression_model_fitted.summary())

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {feature_name}",
        yaxis_title="y",
    )
    fig.show()

    print(t_value, p_value)


# Logistic Regression: Boolean response


def PTCat(x, y, feature_name):

    predictor = statsmodels.api.add_constant(x)
    log_regression_model = statsmodels.api.Logit(y, predictor)
    log_regression_model_fitted = log_regression_model.fit()
    print(f"Variable: {feature_name}")
    print(log_regression_model_fitted.summary())

    # Get the stats
    t_value = round(log_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(log_regression_model_fitted.pvalues[1])

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {feature_name}",
        yaxis_title="y",
    )
    fig.show()

    print(t_value, p_value)


# DIFFERENT DATAFRAMES

# SEABORN
def SeabornDataframe():
    df = seaborn.load_dataset("titanic").dropna().reset_index()
    print(df.columns)
    predictors = ["pclass", "sex", "age", "sibsp", "embarked", "class"]
    target = "survived"
    X = df[predictors]
    y = df[target]
    cal(X, y, predictors)


# SCKITLEARN
def SklearnDataframe():
    data = datasets.load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = df["target"]
    # print('y column', y)
    # y = pd.DataFrame(y)
    predictors = data.feature_names
    cal(X, y, predictors)


def cal(X, y, predictors):

    target_type = None
    feature_type = None
    cat_features = []
    cont_features = []

    # print(y)
    # print(y.dtype)

    if (
        len(y.unique()) <= 2 or y.dtype == "bool"
    ):  # since the values can be discrete as 0 and 1
        target_type = "Categorical"
        print("y type - Categorical")
    else:
        target_type = "Continuous"
        print("y type - Continuous")

    for i in X.columns:

        if (X[i].dtype == "int64" and len(X[i].unique()) > 5) or (
            X[i].dtype == "float64" and len(X[i].unique()) > 5
        ):
            cont_features.append(i)
            # print(i, X[i].values)
            feature_type = "Continuous"
        else:
            cat_features.append(i)
            # print(i, X[i].values)
            feature_type = "Categorical"

        # Plots for continuous and categorical columns

        if feature_type == "Continuous" and target_type == "Categorical":
            cat_resp_cont_predictor([i], X[i])

        if feature_type == "Categorical" and target_type == "Categorical":
            cat_resp_cat_predictor(X[i], y)

        if feature_type == "Categorical" and target_type == "Continuous":
            cont_resp_cat_predictor(y, i)

        if feature_type == "Continuous" and target_type == "Continuous":
            cont_resp_cont_predictor(X[i], y, i)

    if target_type == "Continuous":
        for column in cont_features:
            x = X[column].values
            PTCont(x, y, column)

    else:
        for column in cont_features:
            x = X[column].values
            PTCat(x, y, column)

    # Random forest variable importance ranking

    X_train, X_test, y_train, y_test = train_test_split(
        X[cont_features], y, test_size=0.25, random_state=12
    )
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)

    print(rf.feature_importances_)
    plt.barh(cont_features, rf.feature_importances_)
    plt.show()


if __name__ == "__main__":
    # SeabornDataframe()
    SklearnDataframe()
