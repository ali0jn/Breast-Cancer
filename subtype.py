from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC


dataFrame = pd.read_excel('COH plasma data.xlsx', sheet_name='Sheet1')
dataFrame.dropna(how='all')
X = np.array(dataFrame.iloc[16:, 8:]).transpose()
y = np.column_stack((dataFrame.iloc[1:4, 8:].T, dataFrame.iloc[15, 8:].T))
pd.DataFrame(X).fillna(0, inplace=True)
pd.DataFrame(y).fillna(0, inplace=True)
y = y.astype('int')
sc = StandardScaler()
X_std = sc.fit_transform(X)


def getScore(clf, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    scores = []
    clf = clf
    for train_index, test_index in skf.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    return round(sum(scores) * 10.0, 2)


# Approach2: Create two classifiers: The first one will predict whether a given individual is healthy or breast cancer,
# and the second one will predict the subtype of the breast cancer for those individuals predicted as breast cancer by
# the first individual. Then, you will create ensembl of these two classifiers (you may need to study the corresponding
# chapter of the book, or another tutorial for this).  Please note that the first classifier will utilize the whole
# dataset, while the second classifier will only be using the patients for both training and testing.

X_train, X_test, y_train, y_test = train_test_split(X_std, y[:, 3])
clf = SVC()
clf.fit(X_train, y_train)
yPred = clf.predict(X_test)
wrongEstimations = (yPred != y_test).sum()

