import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
FULL_TRAIN_FILE = "TrainOnMe-2.csv"
EVALUATE_FILE = "EvaluateOnMe-2.csv"

def readData():
    df = pd.read_csv(FULL_TRAIN_FILE, index_col=0)
    # print(df.info())
    # print(df["x6"].describe())
    # print(df["x6"].values)
    df = df[pd.to_numeric(df["x6"], errors="coerce").notnull()]
    df["x6"] = pd.to_numeric(df["x6"])
    df = df.dropna(axis=0)
    df = df.drop_duplicates()
    df["x12"] = df["x12"].astype("bool")
    # print(df.describe(include="all"))
    print(df.info())
    return df

def cleanOutliers(data, scale):
    newData = data.copy()
    # outliers in x4, x11
    Q1 = newData.quantile(0.25)
    Q3 = newData.quantile(0.75)
    IQR = Q3 - Q1
    isInlier = ~((newData < (Q1 - scale * IQR / 2)) | (newData > (Q3 + scale * IQR / 2))).any(axis = 1)
    newData = newData[isInlier]
    return newData



def pipelinePreprocessX(x_train):
    label_encoder = LabelEncoder()
    num_features = x_train.select_dtypes(include=["float64"]).columns
    cat_features = x_train.select_dtypes(include=["object", "bool"]).columns
    preprocessor = Pipeline()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = readData()
    # df.describe()
    roughly_cleaned_data = cleanOutliers(df, scale=8.0)
    roughly_cleaned_data.describe()


    X = roughly_cleaned_data.drop('y', axis = 1)
    Y = roughly_cleaned_data["y"]


