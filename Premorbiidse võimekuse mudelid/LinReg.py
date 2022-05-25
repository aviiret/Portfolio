import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats

# Load data
xl = pd.ExcelFile("LR_selected_models.xlsx")
print(xl.sheet_names)
for subtest in xl.sheet_names:
    dataset = xl.parse(subtest)
    print(subtest)

    # Split-out validation dataset
    array = dataset.values
    X = array[:, :-1]
    y = array[:, -1]
    X = sm.add_constant(X)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # LR
    X_train64 = X_train.astype('float64')
    X_df = pd.DataFrame(X_train64)
    Y_train64 = Y_train.astype('float64')
    Y_df = pd.DataFrame(Y_train64)
    Y_df_z = Y_df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore).fillna(1.0)
    X_df_z = X_df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore).fillna(1.0)
    model_z = sm.OLS(Y_df_z, X_df_z)
    model = sm.OLS(y, X)
    results_z = model_z.fit()
    results = model.fit()
    print(results.summary())
    print('std')
    print(results_z.summary())
    print('r2: {:.3f} adj r2: {:.3f}'.format(results.rsquared, results.rsquared_adj))
