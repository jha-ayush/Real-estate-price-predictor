############### Import libraries ###############
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Ridge, ElasticNet



############### Random Forest Regressor ###############
def random_forest(X, y):
    """
    Fits a Random Forest Regressor model to the input data and returns the predicted values.

    Parameters:
    X (array-like): The input features to use for training the model.
    y (array-like): The target variable to predict.

    Returns:
    y_pred (array-like): The predicted target variable.
    """
    global rf_mean
    global rf_r2
    global rf_mae
    global rf_mse
    global rf_rmse
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize the regressor and fit it to the training data
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    
    # Use the regressor to predict the test data
    y_pred = regressor.predict(X_test)
    
    # Calculate metrics and store them in global variables
    rf_mean = mean_squared_error(y_test, y_pred)
    rf_r2 = r2_score(y_test, y_pred)
    rf_mae = mean_squared_error(y, y_pred)
    rf_mse = np.sqrt(mean_squared_error(y, y_pred)
    
    return y_pred


############### Bagging Regressor ############### 
def bagging_regressor(X, y):
    """
    Fits a Bagging Regressor model to the input data and returns the predicted values.

    Parameters:
    X (array-like): The input features to use for training the model.
    y (array-like): The target variable to predict.

    Returns:
    y_pred (array-like): The predicted target variable.
    """
    global br_mean
    global br_r2
    global br_mae
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize the base estimator and bagging regressor, and fit it to the training data
    base_estimator = DecisionTreeRegressor(max_depth=10)
    bagging_regressor = BaggingRegressor(base_estimator=base_estimator, n_estimators=100, random_state=10)
    bagging_regressor.fit(X_train, y_train)
    
    # Use the regressor to predict the test data
    y_pred = bagging_regressor.predict(X_test)
    
    # Calculate metrics and store them in global variables
    br_mean = mean_squared_error(y_test, y_pred)
    br_r2 = r2_score(y_test, y_pred)
    br_mae = mean_absolute_error(y, y_pred)
    
    return y_pred


############### Extra Trees Regressor ###############
def extra_trees_regressor(X, y):    
    """
    Fits an Extra Trees Regressor model to the input data and returns the predicted values.

    Parameters:
    X (array-like): The input features to use for training the model.
    y (array-like): The target variable to predict.

    Returns:
    y_pred (array-like): The predicted target variable.
    """
    global etr_mean
    global etr_r2
    global etr_mae
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    # Initialize the regressor and fit it to the training data
    etr_regressor = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=10)
    etr_regressor.fit(X_train, y_train)
                                       
    # Use the regressor to predict the test data   
    y_pred = etr_regressor.predict(X_test)
                                       
    # Calculate metrics and store them in global variables
    etr_mean=mean_squared_error(y_test, y_pred)
    etr_r2=r2_score(y_test, y_pred)
    etr_mae = mean_absolute_error(y, y_pred)
                                       
    return y_pred  
                                       

############### Gradient Boosting Regressor ###############
def gradient_boosting_regressor(X, y):
    """
    Fits a Gradient Boosting Regressor model to the input data and returns the predicted values.

    Parameters:
    X (array-like): The input features to use for training the model.
    y (array-like): The target variable to predict.

    Returns:
    y_pred (array-like): The predicted target variable.
    """   
    global gbr_mean
    global gbr_r2
    global gbr_mae
        
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
                                       
    # Initialize the regressor and fit it to the training data
    gbr_regressor = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=10, random_state=10)
    gbr_regressor.fit(X_train, y_train)
                                       
    # Use the regressor to predict the test data
    y_pred = gb_regressor.predict(X_test)
    
    # Calculate metrics and store them in global variables
    gbr_mean=mean_squared_error(y_test, y_pred)
    gbr_r2=r2_score(y_test, y_pred)
    gbr_mae = mean_absolute_error(y, y_pred)
                                       
    return y_pred


############### LassoCV Regressor ###############
def lasso_cv(X, y):
    """
    Fits a LassoCV Regressor model to the input data and returns the predicted values.

    Parameters:
    X (array-like): The input features to use for training the model.
    y (array-like): The target variable to predict.

    Returns:
    y_pred (array-like): The predicted target variable.
    """   
    global lasso_mean
    global lasso_r2
    global lasso_mae
        
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
                                       
    # Initialize the regressor and fit it to the training data
    lasso_regressor = LassoCV(alphas=10, cv=5, random_state=10)
    lasso_regressor.fit(X_train, y_train)
                                       
    # Use the regressor to predict the test data
    y_pred = lasso_regressor.predict(X_test)
       
    # Calculate metrics and store them in global variables                                   
    lasso_mean = mean_squared_error(y_test, y_pred)
    lasso_r2 = r2_score(y_test, y_pred)
    lasso_mae = mean_absolute_error(y, y_pred)
                                       
    return y_pred                   


############### Ridge Regressor ###############
def ridge_regressor(X, y):
    """
    Fits a Ridge Regressor model to the input data and returns the predicted values.

    Parameters:
    X (array-like): The input features to use for training the model.
    y (array-like): The target variable to predict.

    Returns:
    y_pred (array-like): The predicted target variable.
    """ 
    global ridge_mean
    global ridge_r2
    global ridge_mae
    
    # Split data into training and test sets                                   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
                                       
    # Initialize the regressor and fit it to the training data
    ridge_regressor = Ridge(alpha=1.0, random_state=10)
    ridge_regressor.fit(X_train, y_train)
    
    # Use the regressor to predict the test data                                   
    y_pred = ridge_regressor.predict(X_test)
                                       
    # Calculate metrics and store them in global variables
    ridge_mean = mean_squared_error(y_test, y_pred)
    ridge_r2 = r2_score(y_test, y_pred)
    ridge_mae = mean_absolute_error(y, y_pred)
                                       
    return y_pred


############### ElasticNet Regressor ###############
def elastic_net_regressor(X, y):
    """
    Fits a Elastic Net Regressor model to the input data and returns the predicted values.

    Parameters:
    X (array-like): The input features to use for training the model.
    y (array-like): The target variable to predict.

    Returns:
    y_pred (array-like): The predicted target variable.
    """ 
    global enet_mean
    global enet_r2
    global enet_mae
    
    # Split data into training and test sets                                   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
                                       
    # Initialize the regressor and fit it to the training data
    enet_regressor = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=10)
    enet_regressor.fit(X_train, y_train)
    
    # Use the regressor to predict the test data                                   
    y_pred = enet_regressor.predict(X_test)
                                       
    # Calculate metrics and store them in global variables
    enet_mean = mean_squared_error(y_test, y_pred)
    enet_r2 = r2_score(y_test, y_pred)
    enet_mae = mean_absolute_error(y, y_pred)
                                       
    return y_pred