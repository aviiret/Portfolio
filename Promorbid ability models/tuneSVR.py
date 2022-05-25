import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Load data
xl = pd.ExcelFile("AGE_4ST.xlsx")
print(xl.sheet_names)
BP = []

for subtest in xl.sheet_names:
    dataset = xl.parse(subtest)
    print(subtest)

    # Split-out validation dataset
    array = dataset.values
    X = array[:, :-1]
    y = array[:, -1]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    
    # Train model
    model = SVR()
    model.fit(X_train, Y_train)
    
    # Print prediction results
    predictions = model.predict(X_validation)
    print((r2_score(Y_validation, predictions)))
    
    # Defining parameter range
    param_grid = {'kernel' : ('poly','sigmoid'),
                  'C' : [0.1,0.3,1,3],
                  'coef0' : [0.01,10,0.5,0,5]}

    grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 0,n_jobs=-1)
    
    # fitting the model for grid search
    grid.fit(X_train, Y_train)
    
    # Print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_validation)
    BP.append(grid.best_params_)
    print(r2_score(Y_validation, grid_predictions))

df = pd.DataFrame(BP)
print('')
print(df.mode())
