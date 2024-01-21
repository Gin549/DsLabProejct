from re import DEBUG
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
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
from pathlib import WindowsPath
import time
from collections import defaultdict
import re
from collections import Counter

simplefilter(action="ignore", category=FutureWarning)

DEBUGGING: bool = False
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
COL_PADS = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14], dtype=np.int16)
UNIT_OF_MEASURE_COL = {
    "pmax": "mV",
    "negpmax": "mV",
    "tmax": "ns",
    "area": "mV $\\cdot$ ns",
    "rms": "mV",
}


class DumbRegressor:
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
    ratio_keep = 1.0
    # force to reload of the datasets and the processing
    force_reload = True
    exit_after_analysis = True

    path: str
    if ratio_keep < 1.0:
        path = "./data/development_sampled_processed.csv"
    else:
        path = "./data/development_processed.csv"

    w_path = WindowsPath(path)
    if force_reload is True or w_path.exists() is False:
        #
        dev_df, eval_df = get_sampled_data(
            ratio_keep=ratio_keep, force_reload=force_reload, save_to_file=False
        )
        analyse_data(dev_df, "DEVELOPMENT")
        if exit_after_analysis:
            return
        dev_df = feature_selection(dev_df)
        eval_df = feature_selection(eval_df, is_dev=False)
        """triangle_to_xy: dict[int, list[float]] = generate_triangles_position_and_apply(
            dev_df
        )
        print(triangle_to_xy.keys)
        feature_extraction(eval_df, triangle_to_xy)"""
        dev_df.to_csv(path, index=False)
        eval_df.to_csv("./data/evaluation_processed.csv", index=False)
    else:
        # load already processed data
        dev_df = pd.read_csv(path, header=0, index_col=False)
        eval_df = pd.read_csv(
            "./data/evaluation_processed.csv", header=0, index_col=False
        )

    # TODO: think about filtering negpmax

    # analyse_data(dev_df, "EVALUATION")
    regression(dev_df, eval_df)


def get_sampled_data(
    ratio_keep: float = 0.1, force_reload: bool = False, save_to_file: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = WindowsPath(f"{os.curdir}\\data\\development_sampled.csv")

    if path.exists() and force_reload is False:
        dev_df = pd.read_csv(
            "./data/development_sampled.csv", header=0, index_col=False
        )
    else:
        dev_df = pd.read_csv("./data/development.csv", header=0, index_col=False)
        dev_df.sort_values(["x", "y"], ascending=[True, True], inplace=True)
        dev_df = sample_dataframe(dev_df, ratio_keep)
        dev_df.insert(2, "indiciPerOut", dev_df.index, True)
        dev_df = addPmax(dev_df)
        if save_to_file:
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

    check_num_event_per_cell(data, rewrite=True)

    type_of_columns = ["negpmax", "pmax", "area", "tmax", "rms"]
    for type_col in type_of_columns:
        save_heatmaps(data, type_col, rewrite=False)
        save_distributions(data, type_col, rewrite=True)


def check_num_event_per_cell(data: pd.DataFrame, rewrite: bool = False) -> None:
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
    fig, ax = plt.subplots(1, 1)
    plot = sns.heatmap(
        data_heatmap,
        ax=ax,
        square=True,
        cbar_kws={"location": "right"},
    )
    personalize_heatmap(
        ax,
        fig,
        data_heatmap,
        title="Number of events per cell",
        title_colorbar="Number of events per cell",
        xlabel="X [$\\mu m$]",
        ylabel="Y [$\\mu m$]",
    )

    if rewrite:
        fig.savefig(".\\heatmap_num_ev_per_cell.pdf")
        plt.clf()
        plt.close()


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
        fig, ax = plt.subplots(1, 1)
        plot = sns.heatmap(
            data_heatmap,
            ax=ax,
            square=True,
            cbar_kws={"location": "right"},
        )
        personalize_heatmap(
            ax,
            fig,
            data_heatmap,
            title=f"Mean {prefix}[{i}] per (x, y)",
            title_colorbar=f"{prefix}[{i}] mean [{UNIT_OF_MEASURE_COL[prefix]}]",
            xlabel="X [$\\mu m$]",
            ylabel="Y [$\\mu m$]",
        )
        fig.savefig(f"{path.absolute()}\\{prefix}[{i}]_heatmap.png")
        fig.savefig(f"{path.absolute()}\\{prefix}[{i}]_heatmap.pdf")
        plt.clf()
        plt.close()
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
        name_cols.append(f"{prefix}[{i}]")
        if prefix == "negpmax":
            continue

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        plot = sns.histplot(
            data,
            x=f"{prefix}[{i}]",
            kde=True,
            stat="density",
            color=sns.color_palette()[0],
        )
        personalize_histplot(
            ax,
            fig,
            data,
            f"Distribution {prefix}[{i}]",
            f"{prefix}[{i}] [{UNIT_OF_MEASURE_COL[prefix]}]",
            "Density",
        )
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
    plt.clf()
    fig, ax = plt.subplots(1, 1)
    plot = sns.boxplot(dfm, y=f"{prefix}", x="values", color=sns.color_palette()[0])
    personalize_boxplot(
        ax,
        fig,
        dfm,
        f"{prefix.capitalize()} distributions",
        f"{prefix} [{UNIT_OF_MEASURE_COL[prefix]}]",
        "Feature",
        13,
        9,
    )
    fig = plot.get_figure()
    fig.savefig(f"{path.absolute()}\\{prefix}_boxplot.png")
    fig.savefig(f"{path.absolute()}\\{prefix}_boxplot.pdf")
    plot.clear()

    print()


def feature_selection(data: pd.DataFrame, is_dev: bool = True) -> pd.DataFrame:
    cols = []
    if is_dev:
        cols += ["x", "y"]
        cols += ["triangle"]
        cols += ["x_triag"]
        cols += ["y_triag"]
        cols += ["indiciPerOut"]
    pads_to_keep = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
    # TODO: check if keeping tmax the results improve
    type_of_columns = ["negpmax", "pmax", "area"]
    cols += [f"{type}[{i}]" for i in pads_to_keep for type in type_of_columns]
    cols += ["pmax[15]"]
    return data[cols]


def generate_triangles_position_and_apply(
    dev_df: pd.DataFrame,
) -> dict[int, list[float]]:
    col_pads_pmax = [f"pmax[{pad}]" for pad in COL_PADS]

    cols = ["x", "y"] + col_pads_pmax

    dev_df["triangle"] = dev_df[cols].apply(define_triangle, axis=1)

    triangle_to_xy = (
        dev_df[["triangle", "x", "y"]]
        .groupby(
            "triangle",
            axis=0,
        )
        .mean()
    )

    triangle_to_xy_dict = {
        triangle: [
            triangle_to_xy.loc[triangle, :]["x"],
            triangle_to_xy.loc[triangle, :]["y"],
        ]
        for triangle in list(triangle_to_xy.index)
    }

    dev_df["x_triag"] = dev_df.apply(
        lambda x: get_x_triangle(x, triangle_to_xy_dict), axis=1
    )
    dev_df["y_triag"] = dev_df.apply(
        lambda x: get_y_triangle(x, triangle_to_xy_dict), axis=1
    )
    return triangle_to_xy_dict


def feature_extraction(
    data: pd.DataFrame, triangle_to_xy: dict[int, list[float]]
) -> None:
    cols = [f"pmax[{pad}]" for pad in COL_PADS]
    data["triangle"] = data[cols].apply(define_triangle, axis=1)
    print("Unique triangles eval:")
    print(np.unique(data["triangle"]))
    data["x_triag"] = data.apply(lambda x: get_x_triangle(x, triangle_to_xy), axis=1)
    data["y_triag"] = data.apply(lambda x: get_y_triangle(x, triangle_to_xy), axis=1)


def define_triangle(row: pd.Series):
    max_val = 0
    key_max = frozenset([6, 5, 4])
    for key in MAP_SET_PADS_TO_TRIANGLE:
        cols_to_select = [f"pmax[{pas}]" for pas in key]
        val = row[cols_to_select].mean()
        if val > max_val:
            max_val = val
            key_max = key

    return MAP_SET_PADS_TO_TRIANGLE[key_max]


def get_x_triangle(row: pd.Series, triangle_to_xy_dict: dict[int, list[float]]):
    return triangle_to_xy_dict[int(row["triangle"])][0]


def get_y_triangle(row: pd.Series, triangle_to_xy_dict: dict[int, list[float]]):
    return triangle_to_xy_dict[int(row["triangle"])][1]


def feature_extraction_old(dev_df: pd.DataFrame) -> None:
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
    if DEBUGGING:
        print((dev_df["triangle"] == pd.NA).sum())
        print(dev_df["triangle"])
        print(dev_df[col_pads_pmax].head())
        print(dev_df.tail(20))
        print(dev_df.shape)
        print(sorted_index.shape)
        print(dev_df["triangle"].unique())
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        dev_df, x="x", y="y", hue="triangle", palette="Set1", alpha=0.1, ax=ax
    )
    plt.show()
    fig, ax = plt.subplots(1, 1)
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
        ax=ax,
    )
    plt.show()
    triangle_to_xy = (
        dev_df[["triangle", "x", "y"]]
        .groupby(
            "triangle",
            axis=0,
        )
        .mean()
    )
    print("Percentage not starting triangles:")
    print((dev_df["triangle"] > 14).sum() / dev_df["triangle"].size * 100)
    print(sorted_index)

    sorted_values = -1 * (np.sort(-1 * (dev_df[col_pads_pmax].values), axis=1))
    sorted_values = sorted_values[:, :3]
    print(np.mean(sorted_values, axis=1))
    print(sorted_values)
    sns.histplot(
        x=np.mean(sorted_values, axis=1)[dev_df["triangle"] > 14],
        kde=True,
        stat="density",
        color=sns.color_palette()[0],
    )
    sns.histplot(
        x=np.mean(sorted_values, axis=1)[dev_df["triangle"] <= 14],
        kde=True,
        stat="density",
        color=sns.color_palette()[0],
    )
    plt.show()

    def get_x_triangle(row: pd.Series):
        return triangle_to_xy.loc[int(row["triangle"]), :]["x"]

    def get_y_triangle(row: pd.Series):
        return triangle_to_xy.loc[int(row["triangle"]), :]["y"]

    dev_df["x_triag"] = dev_df.apply(get_x_triangle, axis=1)
    dev_df["y_triag"] = dev_df.apply(get_y_triangle, axis=1)
    print(dev_df.head())
    print(sorted_index)


def feature_extraction_old_2(dev_df: pd.DataFrame) -> None:
    col_pads_pmax = [f"pmax[{pad}]" for pad in COL_PADS]

    cols = ["x", "y"] + col_pads_pmax

    def define_triangle(row: pd.Series):
        max_val = 0
        key_max = frozenset([6, 5, 4])
        for key in MAP_SET_PADS_TO_TRIANGLE:
            cols_to_select = [f"pmax[{pas}]" for pas in key]
            val = row[cols_to_select].mean()
            if val > max_val:
                max_val = val
                key_max = key

        return MAP_SET_PADS_TO_TRIANGLE[key_max]

    dev_df["triangle"] = dev_df[cols].apply(define_triangle, axis=1)

    triangle_to_xy = (
        dev_df[["triangle", "x", "y"]]
        .groupby(
            "triangle",
            axis=0,
        )
        .mean()
    )
    if DEBUG:
        print((dev_df["triangle"] == pd.NA).sum())
        print(dev_df["triangle"])
        print(dev_df[col_pads_pmax].head())
        print(dev_df.tail(20))
        print(dev_df.shape)
        print(dev_df["triangle"].unique())
        fig, ax = plt.subplots(1, 1)
        sns.scatterplot(
            dev_df, x="x", y="y", hue="triangle", palette="Set1", alpha=0.1, ax=ax
        )
        plt.show()
        fig, ax = plt.subplots(1, 1)
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
            ax=ax,
        )
        plt.show()
        print("Percentage not starting triangles:")
        print((dev_df["triangle"] > 14).sum() / dev_df["triangle"].size * 100)

        sorted_values = -1 * (np.sort(-1 * (dev_df[col_pads_pmax].values), axis=1))
        sorted_values = sorted_values[:, :3]
        print(np.mean(sorted_values, axis=1))
        print(sorted_values)
        sns.histplot(
            x=np.mean(sorted_values, axis=1)[dev_df["triangle"] > 14],
            kde=True,
            stat="density",
            color=sns.color_palette()[0],
        )
        sns.histplot(
            x=np.mean(sorted_values, axis=1)[dev_df["triangle"] <= 14],
            kde=True,
            stat="density",
            color=sns.color_palette()[0],
        )
        plt.show()

    def get_x_triangle(row: pd.Series):
        return triangle_to_xy.loc[int(row["triangle"]), :]["x"]

    def get_y_triangle(row: pd.Series):
        return triangle_to_xy.loc[int(row["triangle"]), :]["y"]

    dev_df["x_triag"] = dev_df.apply(get_x_triangle, axis=1)
    dev_df["y_triag"] = dev_df.apply(get_y_triangle, axis=1)
    if DEBUG:
        print(dev_df.head())


def regression(dev: pd.DataFrame, eval: pd.DataFrame) -> None:
    dev = addPmax(dev)

    # NEW SPLIT
    maskTraining = boolMaskTrSet(dev)
    maskTest = [not b for b in maskTraining]
    trainSet = dev.iloc[maskTraining, :]
    TestSet = dev.iloc[maskTest, :]
    outInd = indOutliers()
    trainSet = trainSet[~trainSet["indiciPerOut"].isin(outInd)]
    y_train = trainSet[["x", "y"]]
    X_train = trainSet.drop(["x", "y", "indiciPerOut"], axis=1)
    y_test = TestSet[["x", "y"]]
    X_test = TestSet.drop(["x", "y", "indiciPerOut"], axis=1)
    print(len(X_test))
    print(len(X_train))
    print(Counter(maskTest))
    print(Counter(maskTraining))
    print(dev.columns)
    randomForestGridSearch(X_train, y_train, X_test, y_test)
    return

    """X_train, X_val, y_train, y_val = train_test_split(
        dev.drop(["x", "y"], axis=1), dev[["x", "y"]], train_size=0.8, shuffle=True
    )"""

    table = PrettyTable()
    table.field_names = ["Model", "Average Euclidean distance", "Execution time[s]"]
    regressors = [
        DumbRegressor(),
        RandomForestRegressor(random_state=42, max_features="sqrt"),
        # LinearRegression(),
        # GridSearchCV(
        #     DecisionTreeRegressor(),
        #     {"splitter": ["best", "random"]},
        # ),
        # # GridSearchCV(
        # #     RandomForestRegressor(random_state=42),
        # #     {
        # #         "n_estimators": [100, 250, 500],
        # #         "criterion": ["squared_error", "absolute_error"],
        # #         "max_features": [1.0, "sqrt", "log2"],
        # #         "random_state": [42],
        # #         "max_depth": [6, None],
        # #         "n_jobs": [-1],
        # #     },
        # #     scoring="neg_mean_squared_error",
        # #     n_jobs=-1,
        # # ),
        # make_pipeline(StandardScaler(), Ridge(random_state=42)),
        # Lasso(random_state=42),
        # Ridge(random_state=42),
        # GridSearchCV(
        #     Ridge(random_state=42),
        #     {"positive": [True, False], "solver": ["auto", "svd", "cholesky"]},
        # ),
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
    analyse_feature_importante(
        regressors[index_random_forest], list(dev.drop(["x", "y"], axis=1).columns)
    )
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
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    feature_import_df = pd.DataFrame(
        {
            "feature": features_names,
            "importance": importances,
        }
    )
    feature_import_df = feature_import_df.sort_values("importance", ascending=False)
    print(feature_import_df)

    # groupying by type of feature
    feature_cat_import_df = pd.DataFrame(
        {
            "feature": [re.sub(r"\[\d+\]", "", feat) for feat in features_names],
            "importance": importances,
        }
    )
    feature_cat_import_df = (
        feature_cat_import_df.groupby("feature", as_index=False)
        .sum()
        .sort_values("importance", ascending=False)
    )
    print(feature_cat_import_df)
    feature_import_df["std"] = std
    plt.clf()
    fig, ax = plt.subplots(1, 1)
    plt.grid()
    plot = sns.barplot(
        feature_cat_import_df,
        y="feature",
        x="importance",
        color=sns.color_palette()[0],
        ax=ax,
    )
    personalize_barplot(
        ax,
        fig,
        feature_cat_import_df,
        "Feature importance by category",
        "Importance",
        "Feature category",
    )
    plt.show()
    plt.clf()
    fig, ax = plt.subplots(1, 1)
    plt.grid()
    plot = sns.barplot(
        feature_import_df,
        y="feature",
        x="importance",
        color=sns.color_palette()[0],
        ax=ax,
    )
    personalize_barplot(
        ax,
        fig,
        feature_import_df,
        "Feature importance",
        "Importance",
        "Feature",
        12,
        18,
    )
    plt.show()
    plt.clf()


def boolMaskTrSet(dev_df: pd.DataFrame) -> list:
    testIndex = [True] * dev_df.shape[0]
    for i in range(int(dev_df.shape[0] / 20)):
        n = 0
        while n != 4:
            k = np.random.randint(i * 20, i * 20 + 20)
            if testIndex[k] != False:
                testIndex[k] = False
                n += 1
    return testIndex


def indOutliers() -> list:
    with open("./data/indiciOutlierPadAttivi.txt", "r") as fp:
        ind = [int(index.strip()) for index in fp.readlines()]
    return ind


def addPmax(df):
    dim = df.shape
    colPmax = np.zeros(dim[0], dtype=float)
    rowPmax = np.zeros((18), dtype=float)
    names = df.columns
    z = 0
    for k, row in df.iterrows():
        for i in range(18):
            element = f"pmax[{i}]"
            if element in names:
                rowPmax[i] = row[element]
        colPmax[z] = np.max(rowPmax)
        z += 1
    df.insert(2, "maxPmax", colPmax, True)
    return df


def randomForestGridSearch(X_train, y_train, X_test, y_test):
    param_grid = {
        "n_estimators": [60, 80, 100, 250, 500],
        "criterion": ["mse"],
        "max_features": ["sqrt"],
    }
    minDist = 10
    listReg = []
    with open("risultatiTestEstimators.txt", "a") as fp:
        fp.write("Test con sample 0.2 dati con outlier e senza pmax")
        for n_est in param_grid["n_estimators"]:
            for mf in param_grid["max_features"]:
                regressorRF = RandomForestRegressor(
                    n_estimators=n_est, max_features=mf, random_state=42, n_jobs=-1
                )
                regressorRF.fit(X_train, y_train)
                y_pred = regressorRF.predict(X_test)
                med = (
                    np.sqrt(np.sum(np.power(y_test - y_pred, 2), axis=1)).sum()
                    / y_pred.shape[0]
                )

                print(f"n_estimators:{n_est} - maxFeatures:{mf} - dist{med}")
                fp.write(f"n_estimators:{n_est} - maxFeatures:{mf} - dist{med}\n")


def personalize_heatmap(
    ax,
    fig,
    data_heatmap: pd.DataFrame,
    title: str,
    title_colorbar: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig.set_size_inches(10, 8)

    color_bar = ax.collections[0].colorbar
    color_bar.ax.set_title(title_colorbar, fontsize=15)
    color_bar.ax.tick_params(axis="both", which="major", labelsize=13)
    color_bar.ax.tick_params(axis="both", which="minor", labelsize=13)
    ax.set_title(
        title,
        fontdict={
            "fontsize": 24,
            "horizontalalignment": "center",
        },
        pad=20,
    )

    ax.set_xlabel(xlabel, fontsize=18, labelpad=8.0)
    ax.set_ylabel(ylabel, rotation=0, fontsize=18, labelpad=40.0)
    col_labels = [int(col) for col in data_heatmap.columns]
    row_labels = [int(col) for col in data_heatmap.index]
    step = 10
    ax.set_xticks(
        np.arange(start=0.5, stop=data_heatmap.index.size, step=step),
        labels=col_labels[::step],
        fontsize=15,
    )
    ax.set_yticks(
        np.arange(start=0.5, stop=data_heatmap.index.size, step=step),
        labels=row_labels[::step],
        fontsize=15,
    )
    ax.tick_params(axis="x", labelrotation=0)

    plt.tight_layout()


def personalize_histplot(
    ax,
    fig,
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    width_inches: int = 12,
    height_inches: int = 8,
) -> None:
    fig.set_size_inches(width_inches, height_inches)

    ax.set_title(
        title,
        fontdict={
            "fontsize": 24,
            "horizontalalignment": "center",
        },
        pad=20,
    )

    ax.set_xlabel(xlabel, fontsize=18, labelpad=8.0, position="right")
    ax.set_ylabel(ylabel, rotation=0, fontsize=18, labelpad=70.0, position="top")
    ax.tick_params(
        axis="x",
        labelsize=15,
    )
    ax.tick_params(
        axis="y",
        labelsize=15,
    )
    ax.lines[0].set_color(sns.color_palette()[1])
    plt.tight_layout()


def personalize_boxplot(
    ax,
    fig,
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    width_inches: int = 12,
    height_inches: int = 8,
) -> None:
    fig.set_size_inches(width_inches, height_inches)

    ax.set_title(
        title,
        fontdict={
            "fontsize": 24,
            "horizontalalignment": "center",
        },
        pad=20,
    )

    ax.set_xlabel(xlabel, fontsize=18, labelpad=8.0, position="right")
    ax.set_ylabel(ylabel, rotation=0, fontsize=18, labelpad=70.0, position="top")
    ax.tick_params(
        axis="x",
        labelsize=15,
    )
    ax.tick_params(
        axis="y",
        labelsize=15,
    )
    plt.tight_layout()


def personalize_barplot(
    ax,
    fig,
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    width_inches: int = 12,
    height_inches: int = 8,
) -> None:
    fig.set_size_inches(width_inches, height_inches)

    ax.set_title(
        title,
        fontdict={
            "fontsize": 24,
            "horizontalalignment": "center",
        },
        pad=20,
    )

    ax.set_xlabel(xlabel, fontsize=18, labelpad=8.0, position="right")
    ax.set_ylabel(ylabel, rotation=0, fontsize=18, labelpad=70.0, position="top")
    ax.tick_params(
        axis="x",
        labelsize=15,
    )
    ax.tick_params(
        axis="y",
        labelsize=15,
    )
    plt.tight_layout()


main()
