from turtle import color
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from warnings import simplefilter
import seaborn as sns
from matplotlib import pyplot as plt
import os
from pathlib import WindowsPath
import time
from collections import defaultdict

simplefilter(action="ignore", category=FutureWarning)

NUM_EVENT_PER_POS = 100
MAP_SET_PADS_TO_TRIANGLE = defaultdict(
    int,
    {
        frozenset([6, 5, 4]): 1,
        frozenset([4, 5, 3]): 2,
        frozenset([4, 3, 2]): 3,
        frozenset([3, 2, 1]): 4,
        frozenset([6, 8, 5]): 5,
        frozenset([5, 3, 13]): 6,
        frozenset([3, 13, 1]): 7,
        frozenset([8, 5, 10]): 8,
        frozenset([5, 10, 13]): 9,
        frozenset([13, 1, 14]): 10,
        frozenset([8, 9, 10]): 11,
        frozenset([10, 11, 13]): 12,
        frozenset([11, 13, 14]): 13,
        frozenset([9, 10, 11]): 14,
    },
)
max_num_triangle: int = 14
COL_PADS = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14])


class DumbClassifier:
    def __init__(self) -> None:
        self.min_x: int = 0
        self.max_x: int = 0
        self.min_y: int = 0
        self.max_y: int = 0

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.min_x = y["x"].min()
        self.max_x = y["x"].max()
        self.min_y = y["y"].min()
        self.max_y = y["y"].max()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        res = np.ones([X.shape[0], 2])
        res[:, 0] *= (self.min_x + self.max_x) / 2
        res[:, 1] *= (self.min_y + self.min_y) / 2
        return res


def main():
    dev_df, eval_df = get_sampled_data()
    analyse_data(dev_df, "DEVELOPMENT")
    feature_extraction(dev_df, eval_df)
    # analyse_data(dev_df, "EVALUATION")
    regression(dev_df, eval_df)


def get_sampled_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = WindowsPath(f"{os.curdir}\\data\\development_sampled.csv")
    if path.exists():
        dev_df = pd.read_csv(
            "./data/development_sampled.csv", header=0, index_col=False
        )
    else:
        dev_df = pd.read_csv("./data/development.csv", header=0, index_col=False)
        dev_df.sort_values(["x", "y"], ascending=[True, True], inplace=True)
        dev_df = sample_dataframe(dev_df, 0.1)
        print(dev_df.index)
        dev_df.to_csv("./data/development_sampled.csv", index=False)

    eval_df = pd.read_csv("./data/evaluation.csv", header=0, index_col=False)

    return dev_df, eval_df


def sample_dataframe(data: pd.DataFrame, ratio_keep: float) -> pd.DataFrame:
    step = int(1 / ratio_keep)
    if NUM_EVENT_PER_POS % step != 0:
        raise ValueError(
            f"int(1/ratio_keep) has to be a submultiple of a {NUM_EVENT_PER_POS}"
        )
    return data.iloc[::step, :]


def analyse_data(data: pd.DataFrame, df_name: str) -> None:
    print(f"DATAFRAME: {df_name}")
    data.info()
    print(data.describe())

    check_num_event_per_cell(data)

    type_of_columns = ["negpmax", "pmax", "area", "tmax", "rms"]
    for type_col in type_of_columns:
        save_heatmaps(data, type_col)
        save_distributions(data, type_col)


def check_num_event_per_cell(data: pd.DataFrame) -> None:
    df_plot = data[["x", "y", "pmax[1]"]]
    data_heatmap = (
        df_plot.groupby(["x", "y"])
        .count()
        .reset_index()
        .sort_values(by=["x", "y"], ascending=[True, False])
        .pivot(index="y", columns="x", values="pmax[1]")
        .sort_index(ascending=False)
    )
    data_heatmap_np = data_heatmap.values
    print("Distinct num of occurances per (x, y):")
    print(np.unique(data_heatmap_np))
    # sns.heatmap(data_heatmap)
    # plt.show()


def save_heatmaps(data: pd.DataFrame, prefix: str, rewrite: bool = False) -> None:
    path = WindowsPath(f"{os.curdir}\\images\\{prefix}\\heatmap")
    if path.exists() is False:
        path.mkdir()
    elif rewrite is False:
        # if the folder already exists, then the heatmaps are already present
        return

    print(f"Saving heatmaps of {prefix}")
    print(f"\rState: {0}/18", end="")
    for i in range(18):
        name_col = f"{prefix}[{i}]"
        df_plot = data[["x", "y", name_col]]
        data_heatmap = (
            df_plot.groupby(["x", "y"])
            .mean()
            .reset_index()
            .sort_values(by=["x", "y"], ascending=[True, False])
            .pivot(index="y", columns="x", values=name_col)
            .sort_index(ascending=False)
        )
        plot = sns.heatmap(data_heatmap)
        fig = plot.get_figure()
        fig.savefig(f"{path.absolute()}\\{prefix}[{i}]_heatmap.png")
        fig.savefig(f"{path.absolute()}\\{prefix}[{i}]_heatmap.pdf")
        plt.clf()
        print(f"\rState: {i+1}/18", end="")
    print()


def save_distributions(data: pd.DataFrame, prefix: str, rewrite: bool = False) -> None:
    path = WindowsPath(f"{os.curdir}/images/{prefix}/")
    if path.exists() is False:
        path.mkdir()
    elif rewrite is False:
        # if the folder already exists, then the distributions are already present
        return

    print()
    print(f"Saving distributions of {prefix}")
    print(f"\rState: {0}%", end="")
    name_cols: list[str] = []
    for i in range(18):
        if prefix == "negpmax":
            break
        plot = sns.histplot(data, x=f"{prefix}[{i}]", kde=True, stat="density")
        name_cols.append(f"{prefix}[{i}]")
        fig = plot.get_figure()
        fig.savefig(f"{path.absolute()}\\{prefix}[{i}]_distr.png")
        fig.savefig(f"{path.absolute()}\\{prefix}[{i}]_distr.pdf")
        plot.clear()
        print(f"\rState: {round(100*(i+1)/18, 0)}%", end="")

    dfm = data.melt(
        id_vars=["x", "y"],
        var_name=f"{prefix}",
        value_name="values",
        value_vars=name_cols,
    )
    plot = sns.boxplot(dfm, y=f"{prefix}", x="values", color=sns.color_palette()[0])
    fig = plot.get_figure()
    fig.savefig(f"{path.absolute()}\\{prefix}_boxplot.png")
    fig.savefig(f"{path.absolute()}\\{prefix}_boxplot.pdf")
    plot.clear()

    print()


def feature_extraction(dev_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    col_pads_pmax = [f"pmax[{pad}]" for pad in COL_PADS]

    cols = ["x", "y"] + col_pads_pmax
    sorted_index = (-1 * (dev_df[col_pads_pmax].values)).argsort(axis=1)
    # only the first 3 are relevant
    sorted_index = sorted_index[:, :3]
    sorted_index_pd = pd.DataFrame(sorted_index)

    def combine_indexes(row: pd.Series):
        global max_num_triangle
        key = frozenset(COL_PADS[row.values])
        if key not in MAP_SET_PADS_TO_TRIANGLE:
            MAP_SET_PADS_TO_TRIANGLE[key] = max_num_triangle + 1
            max_num_triangle = max_num_triangle + 1

        return MAP_SET_PADS_TO_TRIANGLE[key]

    dev_df["triangle"] = sorted_index_pd.apply(combine_indexes, axis=1)
    sns.scatterplot(dev_df, x="x", y="y", hue="triangle", palette="Set1", alpha=0.1)
    # plt.show()
    sns.scatterplot(
        dev_df[["triangle", "x", "y"]]
        .groupby(
            "triangle",
            axis=0,
        )
        .mean(),
        x="x",
        y="y",
        hue="triangle",
        palette="Set1",
    )
    # plt.show()
    triangle_to_xy = (
        dev_df[["triangle", "x", "y"]]
        .groupby(
            "triangle",
            axis=0,
        )
        .mean()
    )

    def get_x_triangle(row: pd.Series):
        return triangle_to_xy.loc[int(row["triangle"]), :]["x"]

    def get_y_triangle(row: pd.Series):
        return triangle_to_xy.loc[int(row["triangle"]), :]["x"]

    dev_df["x_triag"] = dev_df.apply(get_x_triangle, axis=1)
    dev_df["y_triag"] = dev_df.apply(get_y_triangle, axis=1)
    print(dev_df.head())
    print(sorted_index)


def regression(dev: pd.DataFrame, eval: pd.DataFrame) -> None:
    X_train, X_val, y_train, y_val = train_test_split(
        dev.drop(["x", "y"], axis=1), dev[["x", "y"]], train_size=0.8, shuffle=True
    )

    table = PrettyTable()
    table.field_names = ["Model", "Average Euclidean distance", "Execution time[s]"]
    regressors = [
        DumbClassifier(),
        RandomForestRegressor(random_state=42, max_features="sqrt"),
        LinearRegression(),
        GridSearchCV(
            DecisionTreeRegressor(),
            {"splitter": ["best", "random"]},
        ),
        # GridSearchCV(
        #     RandomForestRegressor(random_state=42),
        #     {
        #         "n_estimators": [100, 250, 500],
        #         "criterion": ["squared_error", "absolute_error"],
        #         "max_features": [1.0, "sqrt", "log2"],
        #         "random_state": [42],
        #         "max_depth": [6, None],
        #         "n_jobs": [-1],
        #     },
        #     scoring="neg_mean_squared_error",
        #     n_jobs=-1,
        # ),
        make_pipeline(StandardScaler(), Ridge(random_state=42)),
        Lasso(random_state=42),
        Ridge(random_state=42),
        GridSearchCV(
            Ridge(random_state=42),
            {"positive": [True, False], "solver": ["auto", "svd", "cholesky"]},
        ),
    ]
    names = [
        "Dumb classifier",
        "Random forest",
        "Linear regression",
        "Decision Tree Regressor",
        # "",
        "Scaler + Ridge",
        "Lasso",
        "Ridge",
        "GridSearchCV on Ridge",
    ]

    for name_m, regr in zip(names, regressors):
        start_time = time.time()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_val)
        med = (
            np.sqrt(np.sum(np.power(y_val - y_pred, 2), axis=1)).sum() / y_pred.shape[0]
        )
        end_time = time.time()
        row = [name_m, med, end_time - start_time]
        table.add_row(row)
        print(row)

    print(table)

    index_random_forest = names.index("Random forest")
    analyse_feature_importante(regressors[index_random_forest], list(dev.columns)[2:])
    # regr = regressors[index_model_choosen]
    # print(regr.best_params_)

    # y_test.to_csv(
    #     path_or_buf="./data/prediction.csv",
    #     header=["Predicted"],
    #     index_label="Id",
    #     index=True,
    # )


def analyse_feature_importante(
    forest: RandomForestRegressor, features_names: list[str]
):
    # TODO: Check the function
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    plt.clf()
    sns.barplot(y=features_names, x=importances)
    plt.show()
    print("Features names:")
    print(features_names)
    print("Feature Importance:")
    print(importances)
    print("Feature importance Variance:")
    print(std)


main()
