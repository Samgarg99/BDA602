import numpy as np
import pandas as pd
import plotly.express as pt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris_data = pd.read_csv("iris.data")
iris_data.columns = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
    "class",
]

print(iris_data.columns)
print(iris_data.head())

iris_data2 = iris_data[["sepal length", "sepal width", "petal length", "petal width"]]


# Statistical Analysis

print(iris_data.describe())

print("Mean- ", np.mean(iris_data, axis=0))
print("Min", np.min(iris_data, axis=0))
print("Max", np.max(iris_data, axis=0))
print("quantile 25", np.quantile(iris_data2, q=0.25, axis=0))
print("quantile 50", np.quantile(iris_data2, q=0.50, axis=0))
print("quantile 75", np.quantile(iris_data2, q=0.75, axis=0))

# Plots

fig = pt.histogram(iris_data, x="sepal width", y="sepal length", color="class")
fig.show()
fig.write_html(file="fig2.html", include_plotlyjs="cdn")

fig2 = pt.scatter_matrix(iris_data, color="class")
fig2.show()
fig2.write_html(file="fig2.html", include_plotlyjs="cdn")


fig3 = pt.scatter(
    iris_data,
    x="sepal width",
    y="sepal length",
    color="class",
    marginal_y="violin",
    marginal_x="box",
    trendline="ols",
    template="simple_white",
)
fig3.show()
# fig3.write_html(file="fig2.html", include_plotlyjs="cdn")

fig4 = pt.box(iris_data, x="sepal width", y="sepal length", color="class")
# fig4.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig4.show()
# fig4.write_html(file="fig2.html", include_plotlyjs="cdn")

# Line plots

fig5 = pt.line(
    iris_data, y="petal length", color="class", title="petal length vs class"
)
fig5.show()
# fig5.write_html(file="fig2.html", include_plotlyjs="cdn")

fig5_1 = pt.line(
    iris_data, y="petal width", color="class", title="petal width vs class"
)
fig5_1.show()
# fig5_1.write_html(file="fig2.html", include_plotlyjs="cdn")

fig5_2 = pt.line(
    iris_data, y="sepal length", color="class", title="sepal length vs class"
)
fig5_2.show()
# fig5_2.write_html(file="fig2.html", include_plotlyjs="cdn")


fig6 = pt.parallel_coordinates(
    iris_data,
    labels={
        "class": "Species",
        "sepal width": "Sepal Width",
        "sepal length": "Sepal Length",
        "petal width": "Petal Width",
        "petal length": "Petal Length",
    },
    color_continuous_scale=pt.colors.diverging.Tealrose,
    color_continuous_midpoint=2,
)
fig6.show()
# fig6.write_html(file="fig2.html", include_plotlyjs="cdn")

# Heatmaps

fig7 = pt.density_heatmap(
    iris_data,
    x="sepal width",
    y="sepal length",
    marginal_x="rug",
    marginal_y="histogram",
)
fig7.show()
# fig7.write_html(file="fig2.html", include_plotlyjs="cdn")

fig8 = pt.density_heatmap(
    iris_data,
    x="petal width",
    y="petal length",
    marginal_x="rug",
    marginal_y="histogram",
)
fig8.show()
# fig8.write_html(file="fig2.html", include_plotlyjs="cdn")

fig9 = pt.density_heatmap(
    iris_data,
    x="sepal width",
    y="petal length",
    marginal_x="rug",
    marginal_y="histogram",
)
fig9.show()
# fig9.write_html(file="fig2.html", include_plotlyjs="cdn")


fig10 = pt.density_heatmap(
    iris_data, x="class", y="petal length", marginal_x="rug", marginal_y="histogram"
)
fig10.show()
# fig10.write_html(file="fig2.html", include_plotlyjs="cdn")

# Building Models

scaler = StandardScaler()

X = iris_data[["sepal length", "sepal width", "petal length", "petal width"]]
y = iris_data["class"]
scaler.fit(X)

# print(scaler.fit(X))
# print(scaler.mean_)

X = scaler.transform(X)

classifiers = [
    RandomForestClassifier(max_depth=2, random_state=0),
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1),
]

for clf in classifiers:
    clf.fit(X, y)
    # print(clf.predict(X))
    print(clf.score(X, y))

# Wrapping the steps into a pipeline

X = iris_data[["sepal length", "sepal width", "petal length", "petal width"]]
y = iris_data["class"]

pipe1 = Pipeline([("scaler", StandardScaler()), ("svc", SVC(gamma=2, C=1))])
pipe1.fit(X, y)
print(pipe1.score(X, y))

pipe2 = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("randomforest", RandomForestClassifier(max_depth=2, random_state=0)),
    ]
)
pipe2.fit(X, y)
print(pipe2.score(X, y))
