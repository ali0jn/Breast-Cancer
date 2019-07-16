from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from scipy import interp
from itertools import cycle
from inspect import signature

dataFrame = pd.read_excel('COH plasma data (edited1).xlsx', sheet_name='Sheet1')
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
    for train_index, test_index in skf.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test) * 10.0)
    return round(sum(scores), 2)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = -0.1,1.1
    x2_min, x2_max = -0.1,1.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

# Approach 1.1: Create a multi-class classifier (again, you may use different algorithms)
# that will (this time) predict if a given individual is healthy, ER, PR, or HER-2 neu
def getnewY(y):
    newY = np.zeros((111,))
    for row_num in range(len(y)):
        if 1 not in y[row_num]:
            newY[row_num, ] = 3
        else:
            for index in range(len(y[row_num])):
                if y[row_num][index] == 1:
                    newY[row_num, ] = index
                    break
    newY = newY.astype('int')
    return newY

def plotmultiEst(X, y):
    y = getnewY(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=0)    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    errors = np.where(y_pred == y_test, True, False)
    prob = clf.predict_proba(X_test)
    for i in range(len(y_pred)):
        if not errors[i]:
            plt.plot(prob[i,0],prob[i,1],'x',color='red',label='Wrong estimations')
        else:
            if y_test[i]:
                plt.plot(prob[i,0],prob[i,1],'x',color='green',label='cancer cells')
            else:
                plt.plot(prob[i,0],prob[i,1],'x',color='blue',label='healthy cells')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('Healty Score')
    plt.ylabel('Cancer Score')
    plt.title('Estimations and Decision Boundary by Logistic Reg.')
    clf.fit(prob, y_pred)
    plot_decision_regions(prob,y_pred,clf)
    plt.show()

# plotmultiEst(X_std[:, :2], y)

def multiKFold(X, y):
    y = getnewY(y)
    lrClf = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=0)
    svmClf = SVC()
    dtClf = DecisionTreeClassifier()
    knnClf = KNeighborsClassifier()
    accScores = [getScore(svmClf, X, y), getScore(lrClf, X, y), getScore(dtClf, X, y), getScore(knnClf, X, y)]
    labels = ['SVM', 'LR', 'DT', 'KNN']
    index = np.arange(len(labels))
    plotter = plt.bar(index, accScores)
    plotter[0].set_color('r'), plotter[1].set_color('b'), plotter[2].set_color('g'), plotter[3].set_color('y')
    plt.xlabel('Classifiers', fontsize=10)
    plt.ylabel('Classifiers\'s Scores', fontsize=10)
    plt.xticks(index, labels, fontsize=10, rotation=30)
    plt.ylim((0, 100))
    plt.title('Stratified 10-K Cross-Validation Scores on Multi-Class Classifier')
    plt.legend((plotter[0], plotter[1], plotter[2], plotter[3]), (accScores[0], accScores[1], accScores[2], accScores[3]))
    plt.show()

# multiKFold(X_std, y)

def multiClassROC(classifier, X_test, y_test):
    probs = classifier.predict_proba(X_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    return auc, fpr, tpr, thresholds

def plotmultiROC(X, y):
    classesName = ['ER', 'PR', 'HER-2 neu', 'Healthy/Patient']
    y = getnewY(y)
    y = label_binarize(y, classes=[0, 1, 2, 3])
    random_state = np.random.RandomState(0)
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # mean_tpr /= n_classes
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # plt.figure()
    # colors = cycle(['red', 'blue', 'green', 'orange'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2, marker='.',
    #             label='ROC curve of class {0} (area = {1:0.2f})'
    #             ''.format(classesName[i], roc_auc[i]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve of Multi-Class Classification by SVM')
    # plt.legend(loc="lower right")
    # plt.show()


    average_precision = average_precision_score(y_test, y_score)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))

    colors = cycle(['orange', 'green', 'red', 'blue', 'magenta'])
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                    ''.format(classesName[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve for Multi-Class Classification by SVM')
    plt.legend(lines, labels, loc=(0, -.38))
    plt.show()

plotmultiROC(X_std, y)



# Approach 1.2: For the same task above, you will create a different binary classifier for each label, and evaluate them separately.

def plotAccuracy(my_list):
    labels = ['SVM', 'Logistic Reg.', 'Decision Tree', 'K-Nearest Neigh.']
    titles = ['ER Accuracy Scores', 'PR Accuracy Scores', 'HER-2 neu Accuracy Scores', 'Healthy/Patient Accuracy Scores']
    plt.figure(figsize=(12, 8))
    index = np.arange(len(labels))
    colors = ('green', 'blue', 'orange', 'red')
    for i in range(1, len(labels)+1):
        ax1 = plt.subplot(2, 2, i)
        plotter = ax1.bar(index, my_list[i-1], color=colors)
        ax1.title.set_text(titles[i-1])
        ax1.legend((plotter[0], plotter[1], plotter[2], plotter[3]), my_list[i-1], loc=4)
        ax1.set_ylim((0, 100))
        ax1.set_xlabel('Classifiers', fontsize=10)
        ax1.set_ylabel('Classifiers\'s Scores', fontsize=10)
        ax1.set_xticks(index)
        ax1.set_xticklabels(labels, fontsize=7, rotation=30)
    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    plt.show()


# classifiers = [SVC(), LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=10)]
# ER_acc = [round(getScore(clf, X_std, y[:, 0]), 2) for clf in classifiers]
# PR_acc = [round(getScore(clf, X_std, y[:, 1]), 2) for clf in classifiers]
# HER2_neu_acc = [round(getScore(clf, X_std, y[:, 2]), 2) for clf in classifiers]
# healthy_patient_acc = [round(getScore(clf, X_std, y[:, 3]), 2) for clf in classifiers]
# my_list = [ER_acc, PR_acc, HER2_neu_acc, healthy_patient_acc]
# plotAccuracy(my_list)

# decision boundaries

def plotEstimations():
    plt.figure(figsize=(12, 8))
    titles = ['ER Est. by Logistic Reg.', 'PR Est. by Logistic Reg.', 'HER-2 neu Est. by Logistic Reg.', 'Healthy/Patient Est. by Logistic Reg.']
    for j in range(1, 5):
        clf = LogisticRegression()
        ax1 = plt.subplot(2, 2, j)
        X_train, X_test, y_train, y_test = train_test_split(X_std, y[:, j-1], test_size=0.3)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        errors = np.where(y_pred == y_test, True, False)
        prob = clf.predict_proba(X_test)
        for i in range(len(y_pred)):
            if not errors[i]:
                ax1.plot(prob[i,0],prob[i,1],'x',color='red',label='Wrong estimations')
            else:
                if y_test[i]:
                    ax1.plot(prob[i,0],prob[i,1],'x',color='green',label='cancer cells')
                else:
                    ax1.plot(prob[i,0],prob[i,1],'x',color='blue',label='healthy cells')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
        ax1.set_xlabel('Healty Score')
        ax1.set_ylabel('Cancer Score')
        ax1.title.set_text(titles[j-1])
        clf.fit(prob,y_pred)
        plot_decision_regions(prob,y_pred,clf)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.show()

# plotEstimations()


def plotROC(classifier, X_test_std, y_test):
    probs = classifier.predict_proba(X_test_std)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    return auc, fpr, tpr, thresholds

def ROC():
    titles = ['ER - ROC Curve', 'PR - ROC Curve', 'HER-2 Neu - ROC Curve', 'Healthy/Patient - ROC Curve']
    plt.figure(figsize=(12, 8))
    for i in range(1, 5):
        ax1 = plt.subplot(2, 2, i)
        X_train, X_test, y_train, y_test = train_test_split(X_std, y[:, i-1], test_size=0.3)
        lrClf = LogisticRegression()
        lrClf.fit(X_train, y_train)
        svmClf = SVC(probability=True)
        svmClf.fit(X_train, y_train)
        dtClf = DecisionTreeClassifier()
        dtClf.fit(X_train, y_train)
        knnClf = KNeighborsClassifier(n_neighbors=10)
        knnClf.fit(X_train, y_train)
        logisticReg = plotROC(lrClf, X_test, y_test)
        svm = plotROC(svmClf, X_test, y_test)
        dt = plotROC(dtClf, X_test, y_test)
        knn = plotROC(knnClf, X_test, y_test)
        ax1.plot(logisticReg[1], logisticReg[2], marker='.', color='black', label='LR: {}'.format(round(logisticReg[0],2)))
        ax1.plot(svm[1], svm[2], marker='.', color='blue', label='SVM: {}'.format(round(svm[0],2)))
        ax1.plot(knn[1], knn[2], marker='.', color='red', label='KNN: {}'.format(round(knn[0],2)))
        ax1.plot(dt[1], dt[2], marker='.', color='green', label='DT: {}'.format(round(dt[0],2)))
        ax1.plot([0, 1], [0, 1], linestyle='--')
        ax1.set_xlabel('False Positive')
        ax1.set_ylabel('True Positive')
        ax1.title.set_text(titles[i-1])
        ax1.legend()
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.show()

# ROC()

def precision_recall():
    titles = ['ER Precision Recall Curve by SVM', 'PR Precision Recall Curve by SVM', 'HER-2 neu Precision Recall Curve by SVM', 'Healthy / Patient Precision Recall Curve by SVM']
    plt.figure(figsize=(12, 8))
    for i in range(1, 5):
        ax1 = plt.subplot(2, 2, i)
        X_train, X_test, y_train, y_test = train_test_split(X_std, y[:, i-1], test_size=0.3)
        classifier = SVC()
        classifier.fit(X_train, y_train)
        y_score = classifier.decision_function(X_test)
        average_precision = average_precision_score(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        ax1.step(recall, precision, color='b', alpha=0.2, where='post')
        ax1.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlim([0.0, 1.0])
        ax1.title.set_text(titles[i-1] + ': AP={0:0.2f}'.format(average_precision))
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.show()

# precision_recall()



# Approach2: Create two classifiers: The first one will predict whether a given individual is healthy or breast cancer,
# and the second one will predict the subtype of the breast cancer for those individuals predicted as breast cancer by
# the first individual. Then, you will create ensembl of these two classifiers (you may need to study the corresponding
# chapter of the book, or another tutorial for this).  Please note that the first classifier will utilize the whole
# dataset, while the second classifier will only be using the patients for both training and testing.

# X_train, X_test, y_train, y_test = train_test_split(X_std, y[:, 3])
# clf = SVC()
# clf.fit(X_train, y_train)
# yPred = clf.predict(X_test)
# wrongEstimations = (yPred != y_test).sum()

