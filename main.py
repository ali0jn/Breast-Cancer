import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def loadData():
    dataFrame = pd.read_excel('COH plasma data.xlsx', sheet_name='Sheet1')
    dataFrame.dropna()
    metabolicsArr = np.array(dataFrame.iloc[16:, 8:].T)
    patientsArr = np.array(dataFrame.iloc[15, 8:])
    X_train = metabolicsArr[:70]
    X_test = metabolicsArr[70:]
    DataFrame(X_train).fillna(0, inplace=True)
    DataFrame(X_test).fillna(0, inplace=True)
    y_train = np.where(patientsArr[:70], 'cancer', 'healthy')
    y_test = np.where(patientsArr[70:], 'cancer', 'healthy')
    return X_train, X_test, y_train, y_test


def scaleFeatures(X_train, X_test):
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std


def logisticRegression(X_train_std, y_train, combinedX, combinedY):
    lrClf = LogisticRegression()
    lrClf.fit(X_train_std, y_train)
    prob = lrClf.predict_proba(X_test_std)
    kFold = round(sum(cross_val_score(lrClf, combinedX, combinedY, cv=10, scoring='f1_micro')) * 10.0, 2)
    return kFold, prob, lrClf


def svm(X_train_std, y_train, combinedX, combinedY):
    svmClf = SVC()
    svmClf.fit(X_train_std, y_train)
    kFold = round(sum(cross_val_score(svmClf, combinedX, combinedY, cv=10, scoring='f1_micro')) * 10.0, 2)
    return kFold


def rfClassifier(X_train_std, y_train, combinedX, combinedY):
    rfClf = RandomForestClassifier()
    rfClf.fit(X_train_std, y_train)
    kFold = round(sum(cross_val_score(rfClf, combinedX, combinedY, cv=10, scoring='f1_micro')) * 10.0, 2)
    return kFold


def dtClassifier(X_train_std, y_train, combinedX, combinedY):
    dtClf = DecisionTreeClassifier()
    dtClf.fit(X_train_std, y_train)
    kFold = round(sum(cross_val_score(dtClf, combinedX, combinedY, cv=10, scoring='f1_micro')) * 10.0, 2)
    return kFold


def plotAccuracy(acc1, acc2, acc3, acc4):
    labels = ['LR', 'SVM', 'RF', 'DT']
    index = np.arange(len(labels))
    plotter = plt.bar(index, [acc1, acc2, acc3, acc4])
    plotter[0].set_color('r'), plotter[1].set_color('b'), plotter[2].set_color('pink'), plotter[3].set_color('y')
    plt.xlabel('Classifiers', fontsize=10)
    plt.ylabel('Classifiers\'s Scores', fontsize=10)
    plt.xticks(index, labels, fontsize=10, rotation=30)
    plt.ylim((0, 100))
    plt.title('10-K Cross-Validation')
    plt.show()


X_train, X_test, y_train, y_test = loadData()
X_train_std, X_test_std = scaleFeatures(X_train, X_test)
combinedX = np.vstack((X_train_std, X_test_std))
combinedY = np.hstack((y_train, y_test))

lrAccuracy, prob , lrClf= logisticRegression(X_train_std, y_train, combinedX, combinedY)
svmAccuracy = svm(X_train_std, y_train, combinedX, combinedY)
rfAccuracy = rfClassifier(X_train_std, y_train, combinedX, combinedY)
dtAccuracy = dtClassifier(X_train_std, y_train, combinedX, combinedY)

# plotAccuracy(lrAccuracy, svmAccuracy, rfAccuracy, dtAccuracy)

