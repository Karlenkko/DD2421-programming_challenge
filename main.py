import pandas as pd
import numpy as np
import sklearn.utils
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, RandomizedSearchCV, cross_val_predict
# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

FULL_TRAIN_FILE = "TrainOnMe-2.csv"
EVALUATE_FILE = "EvaluateOnMe-2.csv"

random = 100

def readTrainData():
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
    # print(df.info())
    return df

def readEvaluationData():
    df = pd.read_csv(EVALUATE_FILE, index_col=0)
    # print(df.info())
    df["x12"] = df["x12"].astype("bool")
    # print(df.info())
    # print(df.describe(include="all"))
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
    num_features = x_train.drop("x1", axis=1).select_dtypes(include=["float64"]).columns
    # num_features = x_train.select_dtypes(include=["float64"]).columns
    # print(len(num_features))
    cat_features = x_train.drop("x7", axis=1).select_dtypes(include=["object", "bool"]).columns
    # cat_features = x_train.select_dtypes(include=["object", "bool"]).columns

    num_transform = Pipeline([("Standard Scale", StandardScaler())])
    # num_transform = Pipeline([("PCA", PCA(n_components=len(num_features) - 1))])
    # cat_transform = Pipeline([("Onehot Encode", OneHotEncoder())])
    cat_transform = Pipeline([("Label Encode", OrdinalEncoder())])

    preprocessor = ColumnTransformer([("num", num_transform, num_features),
                                      ("cat", cat_transform, cat_features)])
    return preprocessor

def testWeakClassifiers(x_train, y_train, preprocess):
    classifiers = {
        # "K Nearest Neighbors" : KNeighborsClassifier(5),
        # "Decision Tree" : DecisionTreeClassifier(),
        # "SVM linear kernel" : SVC(kernel="linear"),
        # "SVM poly kernel" : SVC(kernel="poly", degree=6),
        "SVM Radial Basis Function" : SVC(kernel="rbf"),
        # "Ridge" : RidgeClassifierCV(),
        # "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest" : RandomForestClassifier(n_estimators=200, random_state=random),
        "Best RF1" : RandomForestClassifier(random_state=random, n_estimators=500, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=80, bootstrap=True),
        "Best RF2" : RandomForestClassifier(random_state=random, n_estimators=100, min_samples_split=10, min_samples_leaf=1, max_features="sqrt", max_depth=50, bootstrap=True),
        "Best RF3" : RandomForestClassifier(random_state=random, n_estimators=500, min_samples_split=10, min_samples_leaf=4, max_features="auto", max_depth=70, bootstrap=True)
        # "Bagging" : BaggingClassifier(random_state=random),
        # "Adaboost Decision Tree" : AdaBoostClassifier(random_state=random),
        # "Adaboost Naive Bayes" : AdaBoostClassifier(base_estimator=GaussianNB(), random_state=random),
        # "Stochastic Gradient Descent" : SGDClassifier(random_state=random),
        # "Multi Layer Perceptron" : MLPClassifier(max_iter=2000, hidden_layer_sizes=(x_train.shape[1], 5),random_state=random),
    }
    best = None
    best_score = 0
    best_fold = None
    best_classifier = None
    KF = KFold(shuffle=True, random_state=random, n_splits=20)
    strat_KF = StratifiedKFold(shuffle=True, random_state=random, n_splits=20)
    for name, classifier in classifiers.items():
        pipeline = Pipeline([("preprocessor", preprocess),
                             ("estimator", classifier)])
        cv_score = np.average(cross_val_score(pipeline, x_train, y_train, cv=KF))
        print(f"{name} scored {cv_score} with K-Fold")
        if cv_score > best_score:
            best_score = cv_score
            best = name
            best_fold = "KF"
            best_classifier = classifier
    for name, classifier in classifiers.items():
        pipeline = Pipeline([("preprocessor", preprocess),
                             ("estimator", classifier)])
        cv_score = np.average(cross_val_score(pipeline, x_train, y_train, cv=strat_KF))
        print(f"{name} scored {cv_score} with Stratified K-Fold")
        if cv_score > best_score:
            best_score = cv_score
            best = name
            best_fold = "Strat KF"
            best_classifier = classifier

    print(f"{best} is the best classifier with score = {best_score}, best fold is {best_fold}")
    return best_classifier

def testRandomForestParameters(x_train, y_train, preprocess):
    cross_validation = StratifiedKFold(shuffle=True, random_state=random, n_splits=20)
    params = {
        'randomforest__bootstrap': [True, False],
        'randomforest__max_depth': [40, 50, 60, 70, 80, 90, 100, None],
        'randomforest__max_features': ['auto', 'sqrt'],
        'randomforest__min_samples_leaf': [1, 2, 4],
        'randomforest__min_samples_split': [2, 5, 10],
        'randomforest__n_estimators': [100, 200, 500, 1000]
    }
    pipeline = Pipeline([("preprocessor", preprocess),
                         ("randomforest" , RandomForestClassifier(random_state=random))])
    forest_search = RandomizedSearchCV(pipeline, param_distributions=params, n_iter=120, verbose=1, n_jobs=-1, cv=cross_validation)
    forest_search.fit(x_train, y_train)
    print(f"{forest_search.best_score_} is highest score")
    print(forest_search.best_params_)
    return forest_search.best_params_


def modelEvaluation(X, y, pre):
    cross_validation = StratifiedKFold(shuffle=True, random_state=(random + 20), n_splits=20)
    model = RandomForestClassifier(random_state=(random + 20), n_estimators=500, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=80, bootstrap=True)
    y_pred = cross_val_predict(Pipeline([("pre", pre), ("evl", model)]), X, y, cv=cross_validation, n_jobs=-1)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))


    cross_validation = StratifiedKFold(shuffle=True, random_state=(random + 20), n_splits=20)
    model = RandomForestClassifier(random_state=(random + 20), n_estimators=100, min_samples_split=10, min_samples_leaf=1, max_features="sqrt", max_depth=50, bootstrap=True)
    y_pred = cross_val_predict(Pipeline([("pre", pre), ("evl", model)]), X, y, cv=cross_validation, n_jobs=-1)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

    cross_validation = StratifiedKFold(shuffle=True, random_state=(random + 28), n_splits=20)
    model = RandomForestClassifier(random_state=(random + 40), n_estimators=500, min_samples_split=10,
                                   min_samples_leaf=4, max_features="auto", max_depth=70, bootstrap=True)
    y_pred = cross_val_predict(Pipeline([("pre", pre), ("evl", model)]), X, y, cv=cross_validation, n_jobs=-1)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))


def fitAndOutput(x_train, y_train, preprocess, best_model):
    complete_pipeline = Pipeline([("preprocessor", preprocess),
                                  ("estimator", best_model)])
    complete_pipeline.fit(x_train, y_train)
    x_eval = readEvaluationData()
    res = complete_pipeline.predict(x_eval)
    OUTPUT = "estimations.txt"
    with open(OUTPUT, 'w') as f:
        for prediction in res:
            f.write(prediction + "\n")

if __name__ == '__main__':
    df = readTrainData()
    # df.describe()
    roughly_cleaned_data = cleanOutliers(df, scale=8.0)
    # roughly_cleaned_data.describe()
    X_train = roughly_cleaned_data.drop("y", axis = 1)
    # X_train = X_train.drop("x1", axis=1)
    # print(X_train.describe())
    # arr1 = X_train.x7.unique()

    Y_train = roughly_cleaned_data["y"]
    # arr2 = roughly_cleaned_data.y.unique()
    # for prof in arr2:
    #     print(prof)
    #     for singer in arr1:
    #         print(singer)
    #         print(roughly_cleaned_data[roughly_cleaned_data["y"] == prof][roughly_cleaned_data["x7"] == singer].shape[0])
    X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train, random_state=random)
    preprocessor = pipelinePreprocessX(X_train)
    # best_model = testWeakClassifiers(X_train, Y_train, preprocessor)
    # testRandomForestParameters(X_train, Y_train, preprocessor)
    # modelEvaluation(X_train, Y_train, preprocessor)
    best_model = RandomForestClassifier(random_state=random + 1, n_estimators=500, min_samples_split=10, min_samples_leaf=4, max_features="auto", max_depth=70, bootstrap=True)
    fitAndOutput(X_train, Y_train, preprocessor, best_model)


