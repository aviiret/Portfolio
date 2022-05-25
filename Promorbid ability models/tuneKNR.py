import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Load data
xl = pd.ExcelFile("WAIS_t.xlsx")
print(xl.sheet_names)

for subtest in xl.sheet_names:
    dataset = xl.parse(subtest)
    print(subtest)

    # Split-out validation dataset
    array = dataset.values
    X = array[:, :-1]
    y = array[:, -1]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    
    # Train model
    model = KNeighborsRegressor()
    model.fit(X_train, Y_train)

    # Print prediction results
    predictions = model.predict(X_validation)
    print((r2_score(Y_validation, predictions)))
    
    # Defining parameter range
    param_grid = {'n_neighbors': [2, 4, 6, 8, 10, 12, 14]}
    grid = GridSearchCV(KNeighborsRegressor(), param_grid, refit=True, verbose=1, n_jobs=-1)
    
    # fitting the model for grid search
    grid.fit(X_train, Y_train)
    
    # Print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_validation)
    
    # Print r2
    print(r2_score(Y_validation, grid_predictions))
    
    # Print cv results table
    df = pd.DataFrame(grid.cv_results_)
    print(df.iloc[:, [4, 11, 13]])
    print('')
