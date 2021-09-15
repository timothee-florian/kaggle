from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

def split(X, y, random_state):
    y2 = y.to_numpy().reshape([-1]) # prefered by the sklearn algorithms
    X2 = X.to_numpy() # needed for the k_fold of the model selection
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X2, y2, test_size=0.2, random_state = random_state)
    return X_train_val, X_test, y_train_val, y_test

def regression(model, X_train, y_train, params):
    '''
    Example: 
    regr = regression(model= RandomForestRegressor, X_train=X_train, y_train=y_train, params={'max_depth': 15, 'random_state': 42})
    '''
    model_reg = model()
    model_reg.set_params(**params)
    model_reg.fit(X_train, y_train)
    return model_reg

def model_evaluation(model, X, y, params, metric, n_splits = 3):
    '''
    Example: 
    [metric_scores] = model_evaluation(model= RandomForestRegressor, X=X_train_val, y=y_train_val, params={'max_depth': 15}, metric = r2_score, n_splits = 5)
    '''
    model = model()
    model.set_params(**params)
    k_fold = KFold(n_splits)
    return [model.fit(X[train], y[train]).score(X[val], y[val])
         for train, val in k_fold.split(X)]