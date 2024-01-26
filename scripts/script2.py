import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from warnings import simplefilter
import seaborn as sns
from matplotlib import pyplot as plt
import os
from pathlib import WindowsPath
import time
from collections import defaultdict
import re
from collections import Counter
import random


np.random.seed(42)
random.seed(42)

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
    

Bestparam_RF = {
        "n_estimators": 130,
        "criterion": "squared_error",
        "max_features": "sqrt",
        "max_depth":None
    }
Bestparam_ET = {
        "n_estimators": 130,
        "criterion": "squared_error",
        "max_features": 1.0,
        "max_depth":None
    }


def main():
    ratio_keep = 1.0 #this value represents the ratio of the development sets that will be used for the training
    # force to reload of the datasets and the processing
    force_reload = True 
    exit_after_analysis = False
    do_analysis = False

    path: str
    if ratio_keep < 1.0:
        path = "./data/development_sampled_processed.csv"
    else:
        path = "./data/development_processed.csv"

    w_path = WindowsPath(path)
    if force_reload is True or w_path.exists() is False:
        dev_df, eval_df = get_sampled_data(
            ratio_keep=ratio_keep, force_reload=force_reload, save_to_file=False
        )
        if do_analysis:
            analyse_data(dev_df, "DEVELOPMENT")
            if exit_after_analysis:
                return
        dev_df = feature_selection(dev_df)
        eval_df = feature_selection(eval_df, is_dev=False)
        dev_df.to_csv(path, index=False)
        eval_df.to_csv("./data/evaluation_processed.csv", index=False)
    else:
        # load already processed data
        dev_df = pd.read_csv(path, header=0, index_col=False)
        eval_df = pd.read_csv(
            "./data/evaluation_processed.csv", header=0, index_col=False
        )

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
        save_heatmaps(data, type_col, rewrite=True)
        save_distributions(data, type_col, rewrite=False)


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
        xlabel="X ($\\mu m$)",
        ylabel="Y ($\\mu m$)",  
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
            title_colorbar=f"{prefix}[{i}] mean ({UNIT_OF_MEASURE_COL[prefix]})",
            xlabel="X ($\\mu m$)",
            ylabel="Y ($\\mu m$)",
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
        if i in COL_PADS:
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
            f"{prefix}[{i}] ({UNIT_OF_MEASURE_COL[prefix]})",
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
        f"{prefix} ({UNIT_OF_MEASURE_COL[prefix]})",
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
        cols += ["indiciPerOut"]
    pads_to_keep = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
    type_of_columns = ["negpmax", "pmax", "area"]
    cols += [f"{type}[{i}]" for i in pads_to_keep for type in type_of_columns]
    cols += ["pmax[15]"]
    return data[cols]




def regression(dev: pd.DataFrame, eval: pd.DataFrame) -> None:
    make_GS = False
    make_submission = True



    outInd = ind_outliers(dev)
    maskTraining = boolMaskTrSet(dev)
    maskTest = [not b for b in maskTraining]
    trainSet = dev.iloc[maskTraining, :]
    TestSet = dev.iloc[maskTest, :]

    trainSet = trainSet[~trainSet["indiciPerOut"].isin(outInd)]
    

    y_train = trainSet[["x", "y"]]
    X_train = trainSet.drop(["x", "y", "indiciPerOut"], axis=1)
    y_test = TestSet[["x", "y"]]
    X_test = TestSet.drop(["x", "y", "indiciPerOut"], axis=1)



    
    regressorRF = RandomForestRegressor(n_estimators=Bestparam_RF["n_estimators"],criterion=Bestparam_RF["criterion"],max_features=Bestparam_RF["max_features"],max_depth=Bestparam_RF["max_depth"],random_state=42,n_jobs=-1)
    regressorET = ExtraTreesRegressor(n_estimators=Bestparam_ET["n_estimators"],criterion=Bestparam_ET["criterion"],max_features=Bestparam_ET["max_features"],max_depth=Bestparam_ET["max_depth"],random_state=42,n_jobs=-1)
    


    regressorRF.fit(X_train,y_train)
    regressorET.fit(X_train,y_train)
    print("Test 80/20 split")
    test_regressor(X_test,y_test,regressorRF,regressorET)

    



    if(make_GS):
        custom_GridSearch(X_train, y_train, X_test, y_test) 
    

    if(make_submission):
        TestSet = dev.iloc[maskTest, :]
        TestSet = TestSet[~TestSet["indiciPerOut"].isin(outInd)]
        y_test = TestSet[["x", "y"]]
        X_test = TestSet.drop(["x", "y", "indiciPerOut"], axis=1)
        submission(pd.concat([X_train, X_test], axis=0, ignore_index=True),pd.concat([y_train, y_test], axis=0, ignore_index=True),regressorRF,regressorET,eval)    



    

def submission(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        regressorRF: RandomForestRegressor,
        regressorET: ExtraTreesRegressor,
        eval_df:pd.DataFrame,
):  
    


    


    reg = DumbRegressor()
    reg.fit(X_train,y_train)
    y_pred = reg.predict(eval_df)


    regressorRF.fit(X_train,y_train)
    regressorET.fit(X_train,y_train)
    

    y_pred = regressorRF.predict(eval_df)
    print("DoneRF")
    y_pred2 = regressorET.predict(eval_df)
    print("DoneET")
    y_pred_comb = (y_pred+y_pred2)/2


    i=0
    with open("submissionCombinedDef.csv","w") as fp:
        fp.write("Id,Predicted")
        for x_y in y_pred_comb:
            x,y = x_y
            fp.write(f"\n{i},{x}|{y}")
            i+=1
    i=0
    with open("submissionRFDEf.csv","w") as fp:
        fp.write("Id,Predicted")
        for x_y in y_pred:
            x,y = x_y
            fp.write(f"\n{i},{x}|{y}")
            i+=1        
    i=0
    with open("submissionETDef.csv","w") as fp:
        fp.write("Id,Predicted")
        for x_y in y_pred2:
            x,y = x_y
            fp.write(f"\n{i},{x}|{y}")
            i+=1 


    dubm = DumbRegressor()
    dubm.fit(X_train,y_train)
    ydubm = dubm.predict(eval_df)

    i=0
    with open("submissionNaive.csv","w") as fp:
        fp.write("Id,Predicted")
        for x_y in ydubm:
            x,y = x_y
            fp.write(f"\n{i},{x}|{y}")
            i+=1

def feat_extraction(df):
    dim = df.shape
    colPmax = np.zeros(dim[0],dtype=float)
    rowPmax = np.zeros((18),dtype=float)
    names = df.columns
    relColumns = []
    pmaxColumns = []
    z =0
    for k, row in df.iterrows():
        for i in range(18):
            element = f"pmax[{i}]"
            if(element in names):
                rowPmax[i] = row[element]
                if(z==0):
                    newCol = f"relPmax[{i}]"
                    relColumns.append(newCol)
                    pmaxColumns.append(element)
                    df[newCol] = 0
        colPmax[z] = np.max(rowPmax)
        for nc in range(len(relColumns)):
            df.loc[k,relColumns[nc]] = row[pmaxColumns[nc]]/colPmax[z]
        z+=1
    df.insert(0,"maxPmax",colPmax,True)
    return df

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


def ind_outliers(
        dev: pd.DataFrame
) -> list:
    path = "./data/indiciOutlierPadAttivi.txt"
    w_path = WindowsPath(path)
    if w_path.exists() is False:
        minPNeg = activePad_outliers(dev)
        ind = extrimeOutIndex(minPNeg,'ap')
        with open("./data/indiciOutlierPadAttivi.txt", "w") as fp:
            for elem in ind:
                fp.write(f"{elem}\n")
                
    else:
        with open("./data/indiciOutlierPadAttivi.txt", "r") as fp:
            ind = [int(index.strip()) for index in fp.readlines()]
    return ind


def extrimeOutIndex(col,case):
    q1 = np.percentile(col,25)
    q3 = np.percentile(col,75)
    IQR = q3-q1
    if(case != "ap"):
        extMinOut = q1-IQR*3
        extMaxOut = q3+IQR*3
    else:
        extMinOut = q1-IQR*1.5
        extMaxOut = q3+IQR*1.5
    maskExt = [i for i in range(len(col)) if(col[i]<extMinOut or col[i]>extMaxOut)]

    return maskExt

def activePad_outliers(df):
    minPNeg = np.empty((df.shape[0]),dtype=int)
    rowPNeg = np.zeros((18),dtype=int)
    names = df.columns
    for k, row in df.iterrows():
        for i in range(18):
            element = f"negpmax[{i}]"
            if(element in names):#usefull becouse we don't know which column we dropped
                rowPNeg[i] = row[element]
        minPNeg[k] = np.min(rowPNeg)
    return minPNeg




def test_regressor(
        X_test: pd.DataFrame(),
        y_test: pd.DataFrame(),
        RF_regressor: pd.DataFrame(),
        ET_regressor: pd.DataFrame()
):
    y_pred_RF = RF_regressor.predict(X_test)
    y_pred_ET = ET_regressor.predict(X_test)
    y_predict_VR = (y_pred_ET+y_pred_RF)/2



    med = (
            np.sqrt(np.sum(np.power(y_test - y_predict_VR, 2), axis=1)).sum() / y_test.shape[0]
        )
    med2 = (
            np.sqrt(np.sum(np.power(y_test - y_pred_RF, 2), axis=1)).sum() / y_test.shape[0]
        )
    med3 = (
            np.sqrt(np.sum(np.power(y_test - y_pred_ET, 2), axis=1)).sum() / y_test.shape[0]
        )
    print(f"Distance on local test combined : {med}")
    print(f"Distance on local test randomF : {med2}")
    print(f"Distance on local test ET : {med3}")



def custom_GridSearch(X_train,y_train,X_test,y_test):
    param_grid = {
    "n_estimators": [100],
    "criterion": ["squared_error"],
    "max_features": ["sqrt",1.0],
    "max_depth_RF":[None,30,22],
    "max_depth_ET":[None,33,25]
    }

    print(X_train.columns)
    print(X_test.columns)
    print(len(X_train))

    indeces_Kfold = training_split(X_train)
    with open("risultatiTestEstimatorsNew.txt",'a') as fp:
        fp.write("\nTest 80/20 split\n")
        for n_est in param_grid["n_estimators"]:
            for md in range(len(param_grid["max_depth_RF"])):
                for mf in param_grid["max_features"]:
                    for crt in param_grid["criterion"]:
                        med = 0
                        medET = 0
                        for lisInd in indeces_Kfold:
                            print(f"\nNew Test {med}")
                            maskK = np.array(X_train.index.isin(lisInd))                      
                            regressorRF = RandomForestRegressor(n_estimators=n_est,criterion=crt,max_features=mf,max_depth=param_grid["max_depth_RF"][md],random_state=42,n_jobs=-1)
                            regressorRF.fit(X_train.iloc[~maskK,:],y_train.iloc[~maskK,:])
                            regressorET = ExtraTreesRegressor(n_estimators=n_est,criterion=crt,max_features=mf,max_depth=param_grid["max_depth_ET"][md],random_state=42,n_jobs=-1) 
                            regressorET.fit(X_train.iloc[~maskK,:],y_train.iloc[~maskK,:])
                            y_predRF = regressorRF.predict(X_train.iloc[maskK,:])
                            y_predET = regressorET.predict(X_train.iloc[maskK,:])
                            med += (
                            np.sqrt(np.sum(np.power(y_train.iloc[maskK,:] - y_predRF, 2), axis=1)).sum() / y_predRF.shape[0]
                            )
                            medET+= (
                            np.sqrt(np.sum(np.power(y_train.iloc[maskK,:] - y_predET, 2), axis=1)).sum() / y_predRF.shape[0]
                            )
                        med = med/len(indeces_Kfold)
                        medET = medET/len(indeces_Kfold)
                        print(f"n_estimators:{n_est}- criterion:{crt} - maxFeatures:{mf}- maxDepth:{param_grid['max_depth_RF'][md]} - dist Random Forest:{med}")
                        fp.write(f"n_estimators:{n_est}- criterion:{crt} - maxFeatures:{mf}- maxDepth:{param_grid['max_depth_RF'][md]} - dist Random Forest:{med}\n")
                        print(f"n_estimators:{n_est}- criterion:{crt} - maxFeatures:{mf}- maxDepth:{param_grid['max_depth_ET'][md]} - dist Extra treee:{medET}")
                        fp.write(f"n_estimators:{n_est}- criterion:{crt} - maxFeatures:{mf}- maxDepth:{param_grid['max_depth_ET'][md]} - dist Extra tree:{medET}\n")


def training_split(
        training_set: pd.DataFrame
):  
    list_ind_split = []
    training_set_ind =  training_set.index.tolist()
    train_test_split = random.sample(training_set_ind, len(training_set_ind))
    num_ind_split = int(len(train_test_split)/4)
    for i in range(3):
        list_ind_split.append(train_test_split[(i*num_ind_split):(i*num_ind_split+num_ind_split)])
    list_ind_split.append(train_test_split[(i)*num_ind_split+num_ind_split:]) 
    return list_ind_split

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
