import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Load data
xl = pd.ExcelFile("WAIS_t.xlsx")
print(xl.sheet_names)
dataset = xl.parse('V_PC_MR_I')
print('V_PC_MR_I')

# Split-out validation dataset
array = dataset.values
X = array[:, :-1]
y = array[:, -1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    
# Train model
model = MLPRegressor(max_iter=5000, solver='lbfgs')
model.fit(X_train, Y_train)
    
# Print prediction results
predictions = model.predict(X_validation)
print(r2_score(Y_validation, predictions))

# Defining parameter range
param_grid = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,1)],
   'activation': ['relu','tanh','logistic'],
   'alpha': [0.0001, 0.05],
   'learning_rate': ['constant','adaptive'],
   #'solver': ['adam', 'lbfgs']
   }
    
grid = GridSearchCV(MLPRegressor(max_iter=30000, solver='adam'), param_grid, refit = True, verbose = 1,n_jobs=-1)
    
# fitting the model for grid search
grid.fit(X_train, Y_train)
    
# Print best parameter after tuning
print(grid.best_params_)
grid_predictions = grid.predict(X_validation)
print(r2_score(Y_validation, grid_predictions))
