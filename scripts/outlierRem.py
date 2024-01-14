import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def outlierDetection(col):
    q1 = np.percentile(col,25)
    q3 = np.percentile(col,75)
    IQR = q3-q1
    minOut = q1-IQR*1.5
    maxOut = q3+IQR*1.5
    extMinOut = q1-IQR*3
    extMaxOut = q3+IQR*3
    dictOut = {"Outliers":0,"Extrime_Outliers":0}
    for val in col:
        if(val<extMinOut or val>extMaxOut):
            dictOut["Extrime_Outliers"]+=1
        elif(val<minOut or val>maxOut):
            dictOut["Outliers"]+=1
    return dictOut

def extrimeOutIndex(col,nm):
    q1 = np.percentile(col,25)
    q3 = np.percentile(col,75)
    IQR = q3-q1
    extMinOut = q1-IQR*3
    extMaxOut = q3+IQR*3
    maskExt = list(map(lambda val: val < extMaxOut and val > extMinOut, col))
    return maskExt