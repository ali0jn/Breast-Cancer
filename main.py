import numpy as np, pandas as pd

dataFrame = pd.read_excel('COH plasma data.xlsx', sheet_name='Sheet1')

metabolicsArr = np.array(dataFrame.iloc[16:, 8:].T)
patientsArr = np.array(dataFrame.iloc[15, 8:])

print(metabolicsArr.shape, patientsArr.shape)
