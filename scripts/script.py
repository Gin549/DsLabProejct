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

simplefilter(action="ignore", category=FutureWarning)


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
    dev_df, eval_df = get_data()
    analyse_data(dev_df, "DEVELOPMENT")
    analyse_data(dev_df, "EVALUATION")
    regression(dev_df, eval_df)


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev_df = pd.read_csv("./data/development.csv", header=0, index_col=False)
    eval_df = pd.read_csv("./data/evaluation.csv", header=0, index_col=False)
    return dev_df, eval_df


def analyse_data(data: pd.DataFrame, df_name: str):
    print(f"DATAFRAME: {df_name}")
    data.info()
    print(data.describe())


def regression(dev: pd.DataFrame, eval: pd.DataFrame) -> None:
    X_train, X_val, y_train, y_val = train_test_split(
        dev.drop(["x", "y"], axis=1), dev[["x", "y"]], train_size=0.8, shuffle=True
    )

    table = PrettyTable()
    table.field_names = ["model", "Average Euclidean distance"]
    regressors = [
        DumbClassifier(),
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
        # "Linear regression",
        # "Decision Tree Regressor",
        # "Scaler + Ridge",
        # "Lasso",
        # "Ridge",
        # "GridSearchCV on Ridge",
    ]

    for name_m, regr in zip(names, regressors):
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_val)
        med = (
            np.sqrt(np.sum(np.power(y_val - y_pred, 2), axis=1)).sum() / y_pred.shape[0]
        )
        row = [name_m, med]
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
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    sns.barplot(y=features_names, x=importances)
    plt.show()


main()
