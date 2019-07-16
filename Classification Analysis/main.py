import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
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
    y_train = np.where(patientsArr[:70], 1, 0)
    y_test = np.where(patientsArr[70:], 1, 0)
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
    y_pred = lrClf.predict(X_test_std)
    misClassified = (y_test != y_pred).sum()
    return kFold, prob, misClassified, y_pred, lrClf


def svm(X_train_std, y_train, combinedX, combinedY):
    svmClf = SVC(probability=True)
    svmClf.fit(X_train_std, y_train)
    kFold = round(sum(cross_val_score(svmClf, combinedX, combinedY, cv=10, scoring='f1_micro')) * 10.0, 2)
    return kFold, svmClf


def rfClassifier(X_train_std, y_train, combinedX, combinedY):
    rfClf = RandomForestClassifier()
    rfClf.fit(X_train_std, y_train)
    kFold = round(sum(cross_val_score(rfClf, combinedX, combinedY, cv=10, scoring='f1_micro')) * 10.0, 2)
    return kFold, rfClf


def dtClassifier(X_train_std, y_train, combinedX, combinedY):
    dtClf = DecisionTreeClassifier()
    dtClf.fit(X_train_std, y_train)
    kFold = round(sum(cross_val_score(dtClf, combinedX, combinedY, cv=10, scoring='f1_micro')) * 10.0, 2)
    return kFold, dtClf


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

lrAccuracy, prob, misClassified, y_pred, lrClf = logisticRegression(X_train_std, y_train, combinedX, combinedY)
svmAccuracy, svmClf = svm(X_train_std, y_train, combinedX, combinedY)
rfAccuracy, rfClf = rfClassifier(X_train_std, y_train, combinedX, combinedY)
dtAccuracy, dtClf = dtClassifier(X_train_std, y_train, combinedX, combinedY)


y_test = np.where(y_test == 1, 1, 0)
errors = np.where(y_pred == y_test, True, False)

for i in range(len(y_pred)):
    if not errors[i]:
        plt.plot(prob[i,0],prob[i,1],'x',color='red',label='Wrong estimations')
    else:
        if y_test[i]:
            plt.plot(prob[i,0],prob[i,1],'x',color='yellow',label='cancer cells')
        else:
            plt.plot(prob[i,0],prob[i,1],'x',color='blue',label='healthy cells')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('Healty Score')
plt.ylabel('Cancer Score')
plt.title('Decision Boundary and Estimation by LogisticRegression')


def plot_decision_regions(X, y, classifier, resolution=0.02):
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = -0.1,1.1
    x2_min, x2_max = -0.1,1.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

cls_ = LogisticRegression()
cls_.fit(prob,y_pred)
plot_decision_regions(prob,y_pred,cls_)
plt.show()


def plotROC(classifier, X_test_std, y_test):
    probs = classifier.predict_proba(X_test_std)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    return auc, fpr, tpr, thresholds

# logisticReg = plotROC(lrClf, X_test_std, y_test)
# svm = plotROC(svmClf, X_test_std, y_test)
# rf = plotROC(rfClf, X_test_std, y_test)
# dt = plotROC(dtClf, X_test_std, y_test)

# plt.plot(logisticReg[1], logisticReg[2], marker='.', color='black', label='LR: {}'.format(round(logisticReg[0],2)))
# plt.plot(svm[1], svm[2], marker='.', color='blue', label='SVM: {}'.format(round(svm[0],2)))
# plt.plot(rf[1], rf[2], marker='.', color='red', label='RF: {}'.format(round(rf[0],2)))
# plt.plot(dt[1], dt[2], marker='.', color='green', label='DT: {}'.format(round(dt[0],2)))
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel('False Positive')
# plt.ylabel('True Positive')
# plt.title('ROC Curves')
# plt.legend()
# plt.show()
