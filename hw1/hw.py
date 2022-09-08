import numpy as np
import pandas as pd
import plotly.express as pt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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

# Statistical Analysis

print(iris_data.describe())

print("Mean- ", np.mean(iris_data, axis=0))
print("Min", np.min(iris_data, axis=0))
print("Max", np.max(iris_data, axis=0))


# Plots

fig = pt.histogram(iris_data, x="sepal width", color="class")
# fig.show()


fig2 = pt.scatter_matrix(iris_data, color="class")


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


# Building Models

scaler = StandardScaler()

X = iris_data[["sepal length", "sepal width", "petal length", "petal width"]]
y = iris_data["class"]
scaler.fit(X)
print(scaler.mean_)
X = scaler.transform(X)


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

print(clf.predict(X))


print(clf.score(X, y))
