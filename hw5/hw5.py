import sys
import warnings

import numpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline
import seaborn
import statsmodels.api
from plotly import express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import pearsonr
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine

# --------------------------------------  DATA LOADING  ---------------------------------------------------------------


# SEABORN
def SeabornDataframe(name):
    # df = pd.read_csv('')

    df = seaborn.load_dataset(name=name).dropna().reset_index()
    if name == "titanic":
        predictors = ["pclass", "sex", "age", "sibsp", "embarked", "class"]
        target = "survived"

    elif name == "tips":
        predictors = [
            "total_bill",
            "sex",
            "smoker",
            "day",
            "time",
            "size",
        ]
        target = "tip"

    elif name == "mpg":
        predictors = [
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "origin",
        ]
        target = "mpg"
    else:
        return "invalid name"

    return df, predictors, target


# SCKITLEARN
def SklearnDataframe():
    data = datasets.load_diabetes()
    # data = datasets.load_boston()
    # data = datasets.load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    predictors = data.feature_names
    target = "target"

    return df, predictors, target


# BASEBALL
def Baseball():
    db_user = "root"
    db_pass = ""
    db_host = "localhost"
    db_database = "baseball"
    db_port = 3306

    db_connection_str = (
        f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_database}"
    )
    db_connection = create_engine(db_connection_str)

    sql1 = """select *
                from baseball.combined4
               """

    df1 = pd.read_sql(sql1, db_connection)

    predictors = [
        "RollingSum_Home_Pitcher_Strikeout",
        "RollingSum_Home_Pitcher_walk",
        "RollingSum_Home_Pitcher_Hit_By_Pitch",
        "RollingSum_Home_Pitcher_Home_run",
        "RollingSum_Home_Pitcher_hit",
        "RollingSum_away_Pitcher_Strikeout",
        "RollingSum_away_Pitcher_walk",
        "RollingSum_away_Pitcher_Hit_By_Pitch",
        "RollingSum_away_Pitcher_home_run",
        "RollingSum_away_Pitcher_hit",
        "home_RollingAverage",
        "home_atBatsPerHomerun",
        "home_WalkStrikeoutRatio",
        "home_BABIP",
        "home_HomerunPerHit",
        "home_RollingSum_Innings_Pitched",
        "home_RollingSum_DICE",
        "home_RollingSum_Bases_On_Ball",
        "home_RollingSum_batter_count",
        "home_RollingSum_Hit_p9i",
        "away_RollingAverage",
        "away_atBatsPerHomerun",
        "away_WalkStrikeoutRatio",
        "away_BABIP",
        "away_HomerunPerHit",
        "away_RollingSum_Innings_Pitched",
        "away_RollingSum_DICE",
        "away_RollingSum_Bases_On_Ball",
        "away_RollingSum_batter_count",
        "away_RollingSum_Hit_p9i",
    ]

    target = "winner_home_or_away"

    return df1, predictors, target


# --------------------------------------------- PLOTS ----------------------------------------------------------------


# CATEGORICAL RESPONSE - CONTINUOUS PREDICTOR
def cat_resp_cont_predictor(feature_name, x, y, response="target"):
    y_orig = y
    y, labels = y.factorize()
    y = pd.DataFrame(y, columns=[response])

    hist_data = pd.concat([x, y], axis=1)
    hist_data.columns = [feature_name, response]

    fig = px.histogram(hist_data, x=feature_name, color=y_orig, marginal="rug")
    fig.update_layout(
        title=f"Relationship Between {response} and {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="count",
    )
    # fig.show()

    fig.write_html(f"{feature_name} and {response}.html")
    url = f"{feature_name} and {response}.html"
    return url


# CATEGORICAL RESPONSE - CATEGORICAL PREDICTOR
def cat_resp_cat_predictor(x, y, i, response="target"):
    fig = px.density_heatmap(x=x, y=y)
    fig.update_layout(
        title=f"Categorical Predictor by Categorical Response (with relationship)- {i} and {response}",
        xaxis_title=f"Predictor {i}",
        yaxis_title=f"Response {response}",
    )

    # fig.show()
    fig.write_html(f"{i} and {response}.html")
    url = f"{i} and {response}.html"

    return url


# CONTINUOUS RESPONSE - CONTINUOUS PREDICTOR
def cont_resp_cont_predictor(X, y, feature_name, response="Response"):
    fig = px.scatter(x=X, y=y, trendline="ols", labels=feature_name)
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title=feature_name,
        yaxis_title=response,
    )
    # fig.show()
    fig.write_html(f"{feature_name} and {response}.html")
    url = f"{feature_name} and {response}.html"
    return url


# CONTINUOUS RESPONSE - CATEGORICAL PREDICTOR
def cont_resp_cat_predictor(x, feature_name, y, response="response"):
    x_orig = x
    x, labels = x.factorize()
    x = pd.DataFrame(x, columns=[feature_name])

    hist_data = pd.concat([x, y], axis=1)
    hist_data.columns = [feature_name, "target"]
    fig = px.histogram(hist_data, x="target", color=x_orig, marginal="rug")
    fig.update_layout(
        title=f"Relationship Between response and {feature_name}",
        xaxis_title="target",
        yaxis_title="count",
    )
    # fig.show()

    fig.write_html(f"{feature_name} and {response}.html")
    url = f"{feature_name} and {response}.html"

    return url


# -------------------------------------  CORRELATION METRICS   ----------------------------------------------------


def corr_heatmap(df, pred_type, res_type):
    try:
        fig = px.density_heatmap(
            x=df["pred1"],
            y=df["pred2"],
            z=df["corr"],
            color_continuous_scale="Viridis",
            text_auto=True,
        )

        fig.layout["coloraxis"]["colorbar"]["title"] = "Correlation"

        fig.update_layout(title=f"{pred_type} Predictor by {res_type} Response")

        # fig.show()
        # fig.write_html(f'{pred_type}_{res_type}.html')
        # link = f'{pred_type}_{res_type}.html'

        link = plotly.offline.plot(fig, include_plotlyjs=False, output_type="div")
    except Exception as e:
        link = ""
        print(e)
    return link


def cat_cont_correlation_ratio(categories, values):
    f_cat, _ = pd.factorize(categories)
    cat_num = numpy.max(f_cat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[numpy.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)
    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(n_array)
    numerator = numpy.sum(
        numpy.multiply(
            n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)
        )
    )
    denominator = numpy.sum(numpy.power(numpy.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)
    return eta


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    corr_coeff = numpy.nan
    try:

        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = numpy.sqrt(
                    phi2_corrected / numpy.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = numpy.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = numpy.sqrt(phi2 / numpy.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = numpy.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


# ------------------------------ MEAN OF RESPONSE CORRELATION BETWEEN PREDICTORS -------------------------------------
# heatmaps for mean of response
def MeanOfResponseCorr(means_new, col1, col2):
    trace = go.Heatmap(
        x=means_new.columns, y=means_new.index, z=means_new.values, colorscale="Viridis"
    )

    data = [trace]
    fig = go.Figure(data=data)
    fig.update_layout(title=f"{col1} Predictor by {col2} Predictor")

    # Set x-axis title
    fig.update_xaxes(title_text=col1)

    # Set y-axes titles
    fig.update_yaxes(title_text=col2)
    # fig.show()

    fig.write_html(f"mor {col1} and {col2}.html")
    url = f"mor {col1} and {col2}.html"
    return url


# continuous-continuous predictor


def MeanOfResponse(X, X2, y, col1, col2, target, restype="Continuous"):
    if restype == "Categorical":
        y, _ = y.factorize()
        y = pd.DataFrame(y, columns=[target])
        y = y.squeeze()

    y_avg = y.mean()
    ncut = 10
    temp_df = pd.concat([X, X2, y], axis=1)
    temp_df.columns = [col1, col2, target]

    cuts = pd.DataFrame(
        {
            str(col1) + " Bin": pd.cut(temp_df[col1], ncut),
            str(col2) + " Bin": pd.cut(temp_df[col2], ncut),
        }
    )

    # means = temp_df.join(cuts).groupby(list(cuts)).mean()

    means = temp_df.join(cuts).groupby(list(cuts)).agg({target: ["mean"]})
    means = means.unstack(level=0)  # Use level 0 to put col1 Bin as columns.

    means_new = means[target] - y_avg

    means_new.index = [col.left for col in means_new.index.values]

    means_new.columns = [col[1].left for col in means_new.columns.values]

    """plt.clf()
    sns.heatmap(means[target])
    plt.title('Means of target vs Features 0 and 1')
    plt.tight_layout()
    plt.show()"""

    return MeanOfResponseCorr(means_new, col1, col2)


# continuous-categorical predictor
def ContMeanOfResponse(
    X, X2, y, col1, col2, target, restype="Continuous", ncut1=10, ncut2=3
):
    label = list(set(X2.unique()))

    if X2.dtype != "float64" and X2.dtype != "int64":
        X2, _ = X2.factorize()
        X2 = pd.DataFrame(X2, columns=[col2])
        X2 = X2.squeeze()

    if restype == "Categorical":
        y, _ = y.factorize()
        y = pd.DataFrame(y, columns=[target])
        y = y.squeeze()

    y_avg = y.mean()

    temp_df = pd.concat([X, X2, y], axis=1)
    temp_df.columns = [col1, col2, target]

    cuts = pd.DataFrame(
        {
            str(col1) + " Bin": pd.cut(temp_df[col1], ncut1),
            str(col2) + " Bin": pd.cut(temp_df[col2], ncut2, labels=label),
        }
    )

    means = temp_df.join(cuts).groupby(list(cuts)).agg({target: ["mean"]})

    means = means.unstack(level=0)  # Use level 0 to put col1 Bin as columns.

    means_new = means[target] - y_avg

    means_new.columns = [col[1].left for col in means_new.columns.values]

    return MeanOfResponseCorr(means_new, col1, col2)


# categorical-categorical predictor
def CatMeanOfResponse(
    X, X2, y, col1, col2, target, restype="Continuous", ncut1=3, ncut2=3
):
    label = list(set(X.unique()))
    label2 = list(set(X2.unique()))

    if X.dtype != "float64" and X.dtype != "int64":
        X, _ = X.factorize()
        X = pd.DataFrame(X, columns=[col1])
        X = X.squeeze()

    if X2.dtype != "float64" and X2.dtype != "int64":
        X2, _ = X2.factorize()
        X2 = pd.DataFrame(X2, columns=[col2])
        X2 = X2.squeeze()

    if restype == "Categorical":
        y, _ = y.factorize()
        y = pd.DataFrame(y, columns=[target])
        y = y.squeeze()

    y_avg = y.mean()

    temp_df = pd.concat([X, X2, y], axis=1)
    temp_df.columns = [col1, col2, target]

    cuts = pd.DataFrame(
        {
            str(col1) + " Bin": pd.cut(temp_df[col1], ncut1, labels=label),
            str(col2) + " Bin": pd.cut(temp_df[col2], ncut2, labels=label2),
        }
    )

    means = temp_df.join(cuts).groupby(list(cuts)).mean()

    means = means.unstack(level=0)  # Use level 0 to put col1 Bin as columns.

    means_new = means[target] - y_avg

    return MeanOfResponseCorr(means_new, col1, col2)


# --------------------- DIFFERENT RANKING ALGOS BETWEEN PREDICTORS AND RESPONSE  -------------------------------------


def PTScore(x, y, res_type, feature_name):
    predictor = statsmodels.api.add_constant(x)
    if res_type == "Categorical":
        y, _ = y.factorize()
        regression_model = statsmodels.api.Logit(y, predictor, missing="drop")
    else:
        regression_model = statsmodels.api.OLS(y, predictor)
    regression_model_fitted = regression_model.fit()
    print(f"Variable: {feature_name}")
    print(regression_model_fitted.summary())

    # Get the stats
    t_value = round(regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(regression_model_fitted.pvalues[1])
    print(t_value, p_value)

    # Plot the relationship
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {feature_name}",
        yaxis_title="y",
    )
    fig.write_html(f"{feature_name}_ptscore.html")
    url = f"{feature_name}_ptscore.html"

    return p_value, t_value, url


def MeanOfResponseCont(X, y, col, target, restype="Continuous", bin_size=10):
    if restype == "Categorical":
        y, _ = y.factorize()
        y = pd.DataFrame(y)
        y = y.squeeze()

    y_avg = y.mean()
    count = len(y)

    temp_df = pd.concat([y, X], axis=1)
    temp_df.columns = [target, col]

    # Dividing the data into n bins
    temp_df["bins"] = pd.cut(temp_df[col], bins=bin_size, right=False, labels=False)

    # Grouping the data
    bin_df = temp_df.groupby("bins").agg({target: ["mean", "count"], col: "mean"})
    bin_df.columns = [f"{target} mean", "count", f"{col} mean"]

    # Calculating unweighted score
    bin_df["mean_sq_diff"] = np.square(bin_df[f"{target} mean"] - y_avg)
    uw_score = sum(bin_df["mean_sq_diff"]) / bin_size

    # Calculating weighted score
    bin_df["mean_sq_diff_weighted"] = bin_df["mean_sq_diff"] * (bin_df["count"] / count)
    w_score = sum(bin_df["mean_sq_diff_weighted"]) / bin_size

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Bar(x=bin_df[f"{col} mean"], y=bin_df["count"], name=f"Predictor {col}"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=bin_df[f"{col} mean"], y=bin_df[f"{target} mean"], name="Relationship"
        ),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(title_text="Mean Of Response")

    # Set x-axis title
    fig.update_xaxes(title_text=f"{col}")

    # Set y-axes titles
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Response Mean", secondary_y=True)

    # fig.show()
    fig.write_html(f"{col}_mor.html")
    url = f"{col}_mor.html"

    return uw_score, w_score, url


def MeanOfResponseCat(X, y, col, target, restype="Continuous"):
    bin_size = len(X.unique())

    if restype == "Categorical":
        y, _ = y.factorize()
        y = pd.DataFrame(y)
        y = y.squeeze()

    y_avg = y.mean()
    count = len(y)

    temp_df = pd.concat([y, X], axis=1)
    temp_df.columns = [target, col]

    # Diving the data according to the categories
    bin_df = (
        temp_df.groupby(temp_df[col]).agg({target: ["mean", "count"]}).reset_index()
    )

    bin_df.columns = [f"{col}", f"{target} mean", "count"]

    # Calculating unweighted score
    bin_df["mean_sq_diff"] = np.square(bin_df[f"{target} mean"] - y_avg)
    uw_score = sum(bin_df["mean_sq_diff"]) / bin_size

    # Calculating weighted score
    bin_df["mean_sq_diff_weighted"] = bin_df["mean_sq_diff"] * (bin_df["count"] / count)
    w_score = sum(bin_df["mean_sq_diff_weighted"]) / bin_size

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Bar(x=bin_df[f"{col}"], y=bin_df["count"], name=f"Predictor {col}"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=bin_df[f"{col}"], y=bin_df[f"{target} mean"], name="Relationship"),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(title_text="Mean Of Response")

    # Set x-axis title
    fig.update_xaxes(title_text=f"{col}")

    # Set y-axes titles
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Response Mean", secondary_y=True)

    # fig.show()
    fig.write_html(f"{col}_mor.html")
    url = f"{col}_mor.html"
    return uw_score, w_score, url


def RFScore(X, y, cont_features, df_data):
    if y.dtype != "float64" and y.dtype != "int64":
        y, _ = y.factorize()

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X[cont_features], y)
    df_data["rf_imp"][cont_features] = rf.feature_importances_

    return df_data


# --------------------------------  DATA PROCESSING AND GROUPING  ----------------------------------------------------


def DataProcessing(df, predictors, target):
    X = df[predictors]
    y = df[target]

    cat_features = []
    cont_features = []

    if (
        len(y.unique()) <= 2 or y.dtype == "bool"
    ):  # since the values can be discrete as 0 and 1
        target_type = "Categorical"
    else:
        target_type = "Continuous"

    for i in X.columns:
        if (X[i].dtype == "int64" and len(X[i].unique()) > 8) or (
            X[i].dtype == "float64" and len(X[i].unique()) > 8
        ):
            cont_features.append(i)
        else:
            cat_features.append(i)

    return X, y, target_type, cat_features, cont_features


# ----------------------------- CONVERTING PATHS TO HTML LINKS -----------------------------------------------------


def html(df0, df1, df2, df3, df4, df5, df6, link1, link2, link3):
    pd.set_option("colheader_justify", "center")  # FOR TABLE <th>

    html_string = """
        <html>
          <head><title>HTML Pandas Dataframe with CSS</title></head>

          <body align="center">
            <h1> <b> PREDICTOR-RESPONSE RELATIONSHIP </b></h1>
                <center>{table1}</center>

            <h1> <b> CONTINUOUS-CONTINUOUS RELATIONSHIP </b></h1>
                <center>{table2}</center>

            <h1> <b> CORRELATION MATRIX </b></h1>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            {pic1}

            <h1> <b> CONTINUOUS-CONTINUOUS BRUTE FORCE </b></h1>
                <center>{table3}</center>

            <h1> <b> CONTINUOUS-CATEGORICAL RELATIONSHIP </b></h1>
                <center>{table4}</center>

            <h1> <b> CORRELATION MATRIX </b></h1>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            {pic2}

            <h1> <b> CONTINUOUS-CATEGORICAL BRUTE FORCE </b></h1>
                <center>{table5}</center>

            <h1> <b> CATEGORICAL-CATEGORICAL RELATIONSHIP </b></h1>
            <center> {table6} </center>

            <h1> <b> CORRELATION MATRIX </b></h1>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            {pic3}

            <h1> <b> CATEGORICAL-CATEGORICAL BRUTE FORCE </b></h1>
            <center> {table7} </center>
          </body>
        </html>
        """

    # OUTPUT AN HTML FILE
    with open("myhtml.html", "w") as f:
        f.write(
            html_string.format(
                table1=df0.to_html(render_links=True, escape=False),
                table2=df1.to_html(render_links=True, escape=False),
                table3=df4.to_html(render_links=True, escape=False),
                table4=df2.to_html(render_links=True, escape=False),
                table5=df5.to_html(render_links=True, escape=False),
                table6=df3.to_html(render_links=True, escape=False),
                table7=df6.to_html(render_links=True, escape=False),
                pic1=link1,
                pic2=link2,
                pic3=link3,
            )
        )


def make_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'


# ----------------------------- PREDICTORS-RESPONSE AND PREDICTORS-PREDICTORS ----------------------------------------
def pred_res(df, predictors, target):
    # df_data = DataFrame(predictors)

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

    # Calling function for data processing and categorising
    X, y, target_type, cat_features, cont_features = DataProcessing(
        df, predictors, target
    )

    for i in cat_features:
        df_data["type_of_predictor"][i] = "Categorical"

    for i in cont_features:
        df_data["type_of_predictor"][i] = "Continuous"

    # Calling functions for different plots
    if target_type == "Continuous":
        for i in cat_features:
            url = cont_resp_cat_predictor(X[i], i, y, target)
            df_data["plot"][i] = url
        for j in cont_features:
            url = cont_resp_cont_predictor(X[j], y, j, target)
            df_data["plot"][j] = url
    else:
        for i in cat_features:
            url = cat_resp_cat_predictor(X[i], y, i, target)
            df_data["plot"][i] = url
        for j in cont_features:
            url = cat_resp_cont_predictor(j, X[j], y, target)
            df_data["plot"][j] = url

    # Ranking algos

    # p-value and t-value

    for i in cont_features:
        p_val, t_val, url = PTScore(X[i], y, target_type, i)
        df_data["p-value"][i] = p_val
        df_data["t-score"][i] = t_val
        df_data["tp-plot"][i] = url

    # Mean of Response

    for col in cont_features:
        uw_score, w_score, url = MeanOfResponseCont(X[col], y, col, target, target_type)

        df_data["unweighted_score"][col] = uw_score
        df_data["weighted_score"][col] = w_score
        df_data["mean_of_response_plot"][col] = url

    for col in cat_features:
        uw_score, w_score, url = MeanOfResponseCat(X[col], y, col, target, target_type)

        df_data["unweighted_score"][col] = uw_score
        df_data["weighted_score"][col] = w_score
        df_data["mean_of_response_plot"][col] = url

    df_data = RFScore(X, y, cont_features, df_data)

    df_data["plot"] = df_data["plot"].apply(make_clickable)
    df_data["tp-plot"] = df_data["tp-plot"].apply(make_clickable)
    df_data["mean_of_response_plot"] = df_data["mean_of_response_plot"].apply(
        make_clickable
    )
    # df_data.to_csv('df_data.csv')

    # df_data.to_html('ranking.html', render_links=True, escape=False)
    # webbrowser.open('ranking.html')

    df_data = df_data.sort_values(by="weighted_score", ascending=False)
    return df_data


def pred_pred(df, predictors, target):
    # Calling function for data processing and categorising
    X, y, target_type, cat_features, cont_features = DataProcessing(
        df, predictors, target
    )

    # Correlation Metric

    # Continuous-Continuous features

    df_cont_cont = pd.DataFrame(columns=["predictors", "pearsons r", "plots"])
    df_cont = pd.DataFrame(columns=["pred1", "pred2", "corr"])
    s = []
    for i in cont_features:
        for j in cont_features:
            if (i, j) not in s and (j, i) not in s:
                corr, _ = pearsonr(X[i], X[j])
                url = cont_resp_cont_predictor(X[i], X[j], i, j)
                lst = [[f"{i} and {j}", corr, url]]

                df = pd.DataFrame(lst, columns=["predictors", "pearsons r", "plots"])
                df_cont_cont = pd.concat([df_cont_cont, df], ignore_index=True)

                s.append((i, j))

                df_cont = pd.concat(
                    [
                        df_cont,
                        pd.DataFrame(
                            [[i, j, corr]], columns=["pred1", "pred2", "corr"]
                        ),
                    ],
                    ignore_index=True,
                )
                if (j, i) not in s:
                    df_cont = pd.concat(
                        [
                            df_cont,
                            pd.DataFrame(
                                [[j, i, corr]], columns=["pred1", "pred2", "corr"]
                            ),
                        ],
                        ignore_index=True,
                    )

    link1 = corr_heatmap(df_cont, "Continuous", "Continuous")

    # Continuous-Categorical Features

    df_cont_cat = pd.DataFrame(columns=["predictors", "correlation ratio", "plots"])
    df_corr = pd.DataFrame(columns=["pred1", "pred2", "corr"])

    for i in cont_features:
        for j in cat_features:
            corr = cat_cont_correlation_ratio(X[j], X[i])
            url = cat_resp_cont_predictor(i, X[i], X[j], response=j)
            lst = [[f"{i} and {j}", corr, url]]
            df = pd.DataFrame(lst, columns=["predictors", "correlation ratio", "plots"])
            df_cont_cat = pd.concat([df_cont_cat, df], ignore_index=True)

            df_corr = pd.concat(
                [
                    df_corr,
                    pd.DataFrame([[i, j, corr]], columns=["pred1", "pred2", "corr"]),
                ],
                ignore_index=True,
            )
    link2 = corr_heatmap(df_corr, "Continuous", "Categorical")

    # Categorical-Categorical Features

    df_cat = pd.DataFrame(columns=["pred1", "pred2", "corr"])
    df_cat_cat = pd.DataFrame(columns=["predictors", "cramers v", "plots"])
    cat = []

    for i in cat_features:
        for j in cat_features:
            if (i, j) not in cat and (j, i) not in cat:
                val = cat_correlation(X[i], X[j])
                url = cat_resp_cat_predictor(X[i], X[j], i, j)
                lst = [[f"{i} and {j}", val, url]]
                df = pd.DataFrame(lst, columns=["predictors", "cramers v", "plots"])
                df_cat_cat = pd.concat([df_cat_cat, df], ignore_index=True)

                cat.append((i, j))

                # df_cat[i][j] = val
                df_cat = pd.concat(
                    [
                        df_cat,
                        pd.DataFrame([[i, j, val]], columns=["pred1", "pred2", "corr"]),
                    ],
                    ignore_index=True,
                )
                if (j, i) not in cat:
                    df_cat = pd.concat(
                        [
                            df_cat,
                            pd.DataFrame(
                                [[j, i, val]], columns=["pred1", "pred2", "corr"]
                            ),
                        ],
                        ignore_index=True,
                    )

    link3 = corr_heatmap(df_cat, "Categorical", "Categorical")

    df_cont_cont = df_cont_cont.sort_values(by="pearsons r", ascending=False)
    df_cont_cat = df_cont_cat.sort_values(by="correlation ratio", ascending=False)
    df_cat_cat = df_cat_cat.sort_values(by="cramers v", ascending=False)

    # Mean of Response - Brute force
    df1 = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Residual Plot",
        ]
    )
    df2 = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Residual Plot",
        ]
    )
    df3 = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Residual Plot",
        ]
    )
    cont_list = []
    for i in cont_features:
        for j in cont_features:
            if i != j and (i, j) not in cont_list and (j, i) not in cont_list:
                uw_score1, w_score1, _ = MeanOfResponseCont(
                    X[i], y, i, target, target_type
                )
                uw_score2, w_score2, _ = MeanOfResponseCont(
                    X[j], y, j, target, target_type
                )
                url = MeanOfResponse(X[i], X[j], y, i, j, target, target_type)
                cont_list.append((i, j))
                lst = [i, j, (uw_score1 - uw_score2), (w_score1 - w_score2), url]
                df1 = pd.concat(
                    [
                        df1,
                        pd.DataFrame(
                            [lst],
                            columns=[
                                "Predictor 1",
                                "Predictor 2",
                                "Difference of Mean Response",
                                "Weighted Difference of Mean Response",
                                "Residual Plot",
                            ],
                        ),
                    ],
                    ignore_index=True,
                )

    for i in cont_features:
        for j in cat_features:
            if i != j:
                uw_score1, w_score1, _ = MeanOfResponseCont(
                    X[i], y, i, target, target_type
                )
                uw_score2, w_score2, _ = MeanOfResponseCat(
                    X[j], y, j, target, target_type
                )
                url = ContMeanOfResponse(
                    X[i], X[j], y, i, j, target, target_type, ncut2=len(X[j].unique())
                )
                lst = [i, j, (uw_score1 - uw_score2), (w_score1 - w_score2), url]
                df2 = pd.concat(
                    [
                        df2,
                        pd.DataFrame(
                            [lst],
                            columns=[
                                "Predictor 1",
                                "Predictor 2",
                                "Difference of Mean Response",
                                "Weighted Difference of Mean Response",
                                "Residual Plot",
                            ],
                        ),
                    ],
                    ignore_index=True,
                )

    cat_list = []
    for i in cat_features:
        for j in cat_features:
            if i != j and (i, j) not in cat_list and (j, i) not in cat_list:
                uw_score1, w_score1, _ = MeanOfResponseCat(
                    X[i], y, i, target, target_type
                )
                uw_score2, w_score2, _ = MeanOfResponseCat(
                    X[j], y, j, target, target_type
                )
                url = CatMeanOfResponse(
                    X[i],
                    X[j],
                    y,
                    i,
                    j,
                    target,
                    target_type,
                    ncut1=len(X[i].unique()),
                    ncut2=len(X[j].unique()),
                )
                cat_list.append((i, j))

                lst = [i, j, (uw_score1 - uw_score2), (w_score1 - w_score2), url]
                df3 = pd.concat(
                    [
                        df3,
                        pd.DataFrame(
                            [lst],
                            columns=[
                                "Predictor 1",
                                "Predictor 2",
                                "Difference of Mean Response",
                                "Weighted Difference of Mean Response",
                                "Residual Plot",
                            ],
                        ),
                    ],
                    ignore_index=True,
                )

    df1 = df1.sort_values(by="Weighted Difference of Mean Response", ascending=False)
    df2 = df2.sort_values(by="Weighted Difference of Mean Response", ascending=False)
    df3 = df3.sort_values(by="Weighted Difference of Mean Response", ascending=False)

    # HTML FILE

    df_cont_cont["plots"] = df_cont_cont["plots"].apply(make_clickable)
    df_cont_cat["plots"] = df_cont_cat["plots"].apply(make_clickable)
    df_cat_cat["plots"] = df_cat_cat["plots"].apply(make_clickable)

    df1["Residual Plot"] = df1["Residual Plot"].apply(make_clickable)
    df2["Residual Plot"] = df2["Residual Plot"].apply(make_clickable)
    df3["Residual Plot"] = df3["Residual Plot"].apply(make_clickable)

    return df_cont_cont, df_cont_cat, df_cat_cat, df1, df2, df3, link1, link2, link3


# -----------------------------------------------------------------------------------------------------------------


def main():
    # Loading a dataframe
    # df, predictors, target = SeabornDataframe('mpg')
    # df, predictors, target = SklearnDataframe()
    df, predictors, target = Baseball()

    # Correlations and plots between predictors and response
    df0 = pred_res(df, predictors, target)

    # Correlations and plots between two predictors
    df1, df2, df3, df4, df5, df6, link1, link2, link3 = pred_pred(
        df, predictors, target
    )

    html(df0, df1, df2, df3, df4, df5, df6, link1, link2, link3)


# -----------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())
