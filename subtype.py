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
ER_X = np.array(dataFrame.iloc[16:, 8:]).transpose()
ER_y = np.array([dataFrame.iloc[1, 8:], np.zeros(171, )]).transpose()


ER_X_train, ER_X_test, ER_y_train, ER_y_test = train_test_split(ER_X, ER_y, test_size=0.3)

sc = StandardScaler()
sc.fit(ER_X_train)
ER_X_train_std = sc.transform(ER_X_train)
ER_X_test_std = sc.transform(ER_X_test)

pd.DataFrame(ER_X_train_std).fillna(0, inplace=True)
pd.DataFrame(ER_X_test_std).fillna(0, inplace=True)

lrClf = LogisticRegression()
print(ER_X_train_std.shape, ER_X_test_std.shape)
lrClf.fit(ER_X_train_std, ER_y_train)

# combinedX = np.vstack((X_train_std, X_test_std))
# combinedY = np.vstack((y_train, y_test))


# lrCrossVal = round(sum(cross_val_score(lrClf, combinedX, combinedY, cv=10, scoring='f1_micro')) * 10.0, 2)






