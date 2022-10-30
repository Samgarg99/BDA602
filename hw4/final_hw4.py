# imports
import webbrowser

import numpy
import numpy as np
import pandas as pd
import seaborn
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    fig_1.write_html(f"{feature_name}.html")
    url = f"{feature_name}.html"

    """n = 200
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
    )"""

    return url


# CATEGORICAL RESPONSE - CATEGORICAL PREDICTOR
def cat_resp_cat_predictor(x, y, i):
    fig = px.density_heatmap(x=x, y=y)
    fig.update_layout(
        title=f"Categorical Predictor by Categorical Response (with relationship)- {i}",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )

    fig.write_html(f"{i}.html")
    url = f"{i}.html"

    return url


# CONTINUOUS RESPONSE - CONTINUOUS PREDICTOR
def cont_resp_cont_predictor(X, y, feature_name):
    fig = px.scatter(x=X, y=y, trendline="ols", labels=feature_name)
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title=feature_name,
        yaxis_title="Response",
    )
    fig.write_html(f"{feature_name}.html")
    url = f"{feature_name}.html"

    return url


# CONTINUOUS RESPONSE - CATEGORICAL PREDICTOR
def cont_resp_cat_predictor(y, feature_name):

    fig_1 = ff.create_distplot([y], [feature_name], bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.write_html(f"{feature_name}.html")
    url_1 = f"{feature_name}.html"

    """n = 200
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

    fig_2.write_html(f'{feature_name}.html')"""

    return url_1


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
    fig.write_html(f"{feature_name}_ptcont.html")
    url = f"{feature_name}_ptcont.html"

    return p_value, t_value, url


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
    fig.write_html(f"{feature_name}_ptcat.html")
    url = f"{feature_name}_ptcat.html"

    return p_value, t_value, url


# Difference with mean of response


def MeanOfResponse(X, y, col, predtype="Continuous", bin_size=10):

    if predtype == "Categorical":
        labelencoder = LabelEncoder()
        X[col] = labelencoder.fit_transform(X[col])

    temp_df = pd.DataFrame()

    y_avg = y.mean()
    count = len(y)

    temp_df[col] = X[col]
    temp_df["response"] = y
    temp_df["bins"] = pd.cut(X[col], bins=bin_size, right=False, labels=False)

    # Calculating unweighted score
    temp = temp_df.groupby("bins").mean().reset_index()
    msd = np.square(temp["response"] - y_avg)
    uw_score = sum(msd) / bin_size

    # Calculating weighted score
    temp2 = temp_df.groupby("bins").count().reset_index()
    # avg_wt = (temp2['response']/count)
    msdw = (temp2["response"] / count) * (np.square(temp["response"] - y_avg))
    w_score = sum(msdw) / bin_size

    """fig = px.histogram(temp_df[col], x=temp_df['bins'], title=col)
    fig.update_layout(bargap=0.2)"""

    counts, bins = numpy.histogram(X[col], bins=bin_size)
    bins = 0.5 * (bins[:-1] + bins[1:])

    fig = px.bar(
        x=bins,
        y=counts,
        labels={"x": "BinCenter", "y": "count"},
        title=f"Binned difference with Mean of response vs bin - {col}",
    )
    fig.update_layout(xaxis_title=col, yaxis_title="count")

    fig.write_html(f"{col}_mor.html")
    url = f"{col}_mor.html"

    return uw_score, w_score, url


# DIFFERENT DATAFRAMES

# SEABORN
def SeabornDataframe():
    # df = pd.read_csv('')
    df = seaborn.load_dataset("titanic").dropna().reset_index()
    predictors = ["pclass", "sex", "age", "sibsp", "embarked", "class"]
    target = "alive"
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
    predictors = data.feature_names
    cal(X, y, predictors)


def make_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'


def cal(X, y, predictors):

    target_type = None
    feature_type = None
    cat_features = []
    cont_features = []

    df_data = pd.DataFrame(
        columns=[
            "type_of_predictor",
            "plot",
            "t-score",
            "p-value",
            "tp-plot",
            "rf_imp",
            "unweighted_score",
            "weighted_score",
            "mean_of_response_plot",
        ],
        index=predictors,
    )

    if (
        len(y.unique()) <= 2 or y.dtype == "bool"
    ):  # since the values can be discrete as 0 and 1
        target_type = "Categorical"
        labelencoder = LabelEncoder()  # to encode categories to discrete values
        y = labelencoder.fit_transform(y)

    else:
        target_type = "Continuous"

    for i in X.columns:

        if (X[i].dtype == "int64" and len(X[i].unique()) > 5) or (
            X[i].dtype == "float64" and len(X[i].unique()) > 5
        ):
            cont_features.append(i)
            feature_type = "Continuous"
            df_data["type_of_predictor"][i] = feature_type
        else:
            cat_features.append(i)
            feature_type = "Categorical"
            df_data["type_of_predictor"][i] = feature_type

        # Plots for continuous and categorical columns

        if feature_type == "Continuous" and target_type == "Categorical":
            link = cat_resp_cont_predictor([i], X[i])
            df_data["plot"][i] = link

        if feature_type == "Categorical" and target_type == "Categorical":
            link = cat_resp_cat_predictor(X[i], y, i)
            df_data["plot"][i] = link

        if feature_type == "Categorical" and target_type == "Continuous":
            link = cont_resp_cat_predictor(y, i)
            df_data["plot"][i] = link

        if feature_type == "Continuous" and target_type == "Continuous":
            link = cont_resp_cont_predictor(X[i], y, i)
            df_data["plot"][i] = link

    # calculate tscore and pscore along with their plots

    if target_type == "Continuous":
        for column in cont_features:
            x = X[column].values
            p_val, t_score, link = PTCont(x, y, column)
            df_data["p-value"][column] = p_val
            df_data["t-score"][column] = t_score
            df_data["tp-plot"][column] = link

    else:
        for column in cont_features:
            x = X[column].values
            p_val, t_score, link = PTCat(x, y, column)
            df_data["p-value"][column] = p_val
            df_data["t-score"][column] = t_score
            df_data["tp-plot"][column] = link

    # Difference with mean of response

    for col in cont_features:
        uw_val, w_val, link = MeanOfResponse(X, y, col)
        df_data["unweighted_score"][col] = uw_val
        df_data["weighted_score"][col] = w_val
        df_data["mean_of_response_plot"][col] = link

    for col in cat_features:
        uw_val, w_val, link = MeanOfResponse(
            X, y, col, "Categorical", len(X[col].unique())
        )
        df_data["unweighted_score"][col] = uw_val
        df_data["weighted_score"][col] = w_val
        df_data["mean_of_response_plot"][col] = link

    # Random forest variable importance ranking

    X_train, X_test, y_train, y_test = train_test_split(
        X[cont_features], y, test_size=0.25, random_state=12
    )
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    df_data["rf_imp"][cont_features] = rf.feature_importances_

    df_data["plot"] = df_data["plot"].apply(make_clickable)
    df_data["tp-plot"] = df_data["tp-plot"].apply(make_clickable)
    df_data["mean_of_response_plot"] = df_data["mean_of_response_plot"].apply(
        make_clickable
    )
    df_data.to_csv("data_df.csv")
    df_data.to_html("ranking.html", render_links=True, escape=False)
    webbrowser.open("ranking.html")


if __name__ == "__main__":
    # SeabornDataframe()
    SklearnDataframe()
