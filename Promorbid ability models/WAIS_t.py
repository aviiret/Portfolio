# Load libraries
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load data
xl = pd.ExcelFile("AGE_4ST.xlsx")
print(xl.sheet_names)

for subtest in xl.sheet_names:
    dataset = xl.parse(subtest)
    print(subtest)

    # Split-out validation dataset
    array = dataset.values
    X = array[:, :-1]
    y = array[:, -1]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # Spot Check Algorithms
    models = [('LR', LinearRegression()),
              ('GBR', GradientBoostingRegressor(min_samples_leaf=1, min_samples_split=8, n_estimators=150)),
              ('KNR', KNeighborsRegressor(n_neighbors=8)), ('SVR', SVR(C=3, coef0=10, kernel='poly')),
              ('L1', Lasso(alpha=0.1)), ('MLPR', MLPRegressor(max_iter=2000, hidden_layer_sizes=(50, 100, 50)))]

    # Evaluate each model in turn
    print("Models evaluation:")
    results = []
    names = []
    resultList = []
    tableList = []
    kfold = KFold(n_splits=5, random_state=1, shuffle=True)

    for name, model in models:
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold)
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        resultList = name, cv_results.mean(), cv_results.std()
        tableList.append(resultList)

    # Compare algorithms
    pyplot.boxplot(results, labels=names)
    pyplot.title(subtest + ' Algoritmide võrdlus')
    pyplot.show()
