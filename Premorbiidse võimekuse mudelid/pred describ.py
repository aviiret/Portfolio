import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
import numpy as np
from scipy.stats import pearsonr


def pearsonr_pval(x, y):
    return pearsonr(x, y)[1]


# Load data
xl = pd.ExcelFile("LR_selected_models.xlsx")
print(xl.sheet_names)
r2 = []
SEElist = []
wSEElist = []
wci = []
corr = []
pval = []
acc = []

for subtest in xl.sheet_names:
    validClasses = pd.DataFrame()
    predClasses = pd.DataFrame()
    dataset = xl.parse(subtest)

    # Split-out validation dataset
    array = dataset.values
    X = array[:, :-1]
    y = array[:, -1]
    X = sm.add_constant(X)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # Assign classes to Y_validation
    Y_valid64 = Y_validation.astype('float64')
    Y_validDf = pd.DataFrame(Y_valid64, columns=['FSIQ'])
    conditions = [
        (Y_validDf['FSIQ'] < 70),
        (Y_validDf['FSIQ'] >= 70) & (Y_validDf['FSIQ'] < 80),
        (Y_validDf['FSIQ'] >= 80) & (Y_validDf['FSIQ'] < 90),
        (Y_validDf['FSIQ'] >= 90) & (Y_validDf['FSIQ'] < 110),
        (Y_validDf['FSIQ'] >= 110) & (Y_validDf['FSIQ'] < 120),
        (Y_validDf['FSIQ'] >= 120) & (Y_validDf['FSIQ'] < 130),
        (Y_validDf['FSIQ'] >= 130)
    ]
    values = [1, 2, 3, 4, 5, 6, 7]
    validClasses['class'] = np.select(conditions, values)

    # Fit model
    model = sm.OLS(Y_train, X_train)
    regression_results = model.fit()

    # Get predictions and evaluations
    predictions = regression_results.predict(X_validation)
    predictions2 = regression_results.get_prediction(X_validation)
    frame = predictions2.summary_frame(alpha=0.05)
    score = r2_score(Y_validation, predictions)
    r2.append(score)

    # Descriptives
    predDf = pd.DataFrame(predictions, columns=['FSIQ'])
    print(subtest + ': mean: {:.2f} ({:.2f}) min: {:.2f} max: {:.2f}'.format(predDf['FSIQ'].mean(), predDf['FSIQ'].std(),
                                                                             predDf['FSIQ'].min(), predDf['FSIQ'].max()))

print('Actual mean: {:.2f} ({:.2f})'.format(Y_validDf['FSIQ'].mean(), Y_validDf['FSIQ'].std()))
