import statsmodels.stats.oneway as sm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
xl = pd.ExcelFile("WAIS_t_expanded.xlsx")
subtest = 'ANOVA'
dataset = xl.parse(subtest)
print(subtest)

# Split-out validation dataset
array = dataset.values
X = array[:, :-1]
y = array[:, -1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
X_valid64 = X_validation.astype('float64')
X_validDf = pd.DataFrame(X_valid64, columns=['Vanus', 'VCI', 'POI', 'PSI', 'WMI', 'PIQ', 'VIQ', 'FSIQ', 'GAI'])
X_train64 = X_train.astype('float64')
X_trainDf = pd.DataFrame(X_train64, columns=['Vanus', 'VCI', 'POI', 'PSI', 'WMI', 'PIQ', 'VIQ', 'FSIQ', 'GAI'])
Y_train64 = Y_train.astype('float64')
Y_trainDf = pd.DataFrame(Y_train64, columns=['CPI'])
Y_valid64 = Y_validation.astype('float64')
Y_validDf = pd.DataFrame(Y_valid64, columns=['CPI'])
Xlist = ['Vanus', 'VCI', 'POI', 'PSI', 'WMI', 'PIQ', 'VIQ', 'FSIQ', 'GAI']

i = 0
for x in Xlist:
    print(x)
    print(sm.anova_oneway((X_train[:, i], X_validation[:, i]), use_var='equal', welch_correction=False))
    print('')
    i += 1

print('CPI')
print(sm.anova_oneway((Y_train, Y_validation), use_var='equal', welch_correction=False))
print('')
print('WMI')
WMI_train = X_train[:, 4][np.logical_not(np.isnan(X_train[:, 4]))]
WMI_val = X_validation[:, 4][np.logical_not(np.isnan(X_validation[:, 4]))]
print(sm.anova_oneway((WMI_train, WMI_val), use_var='equal', welch_correction=False))
print('\ntrain means and std')

for x in Xlist:
    print('{}: {:.2f} ({:.2f})'.format(x, X_trainDf[x].mean(), X_trainDf[x].std()))
print('CPI: {:.2f} ({:.2f})'.format(Y_trainDf['CPI'].mean(), Y_trainDf['CPI'].std()))
print('\nvalidation means and std')

for x in Xlist:
    print('{name} {mean:.2f} ({std:.2f})'.format(name=x, mean=X_validDf[x].mean(), std=X_validDf[x].std()))
print('CPI: {:.2f} ({:.2f})'.format(Y_validDf['CPI'].mean(), Y_validDf['CPI'].std()))
