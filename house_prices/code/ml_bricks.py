from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

import numpy as np

def split_2(X, y, train_size = 0.8, random_state = None):
    y2 = y.to_numpy().reshape([-1]) # prefered by the sklearn algorithms
    X2 = X.to_numpy() # needed for the k_fold of the model selection

    X_train, X_test, y_train, y_test = train_test_split(
            X2, y2, train_size= train_size, random_state = random_state)
    
    return X_train, y_train, X_test, y_test

def split_3(X, y, splits = [0.6, 0.2, 0.2], random_state = None):
    y2 = y.to_numpy().reshape([-1]) # prefered by the sklearn algorithms
    X2 = X.to_numpy() # needed for the k_fold of the model selection

    w = sum(splits)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
            X2, y2, test_size= splits.pop()/w, random_state = random_state)

    w = sum(splits)
    X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, train_size= splits.pop(0)/w, random_state = random_state)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

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
    return [metric(y[val], model.fit(X[train], y[train]).predict(X[val]))
            for train, val in k_fold.split(X)]

def evaluate(X, y, model_eval):
    '''
    model_eval: dictionary containging the model type, its parameters, 
    the metric used to evaluate and the numer of splits for cross validation

    Add the results of the evaluation to model_eval
    '''
    results = model_evaluation(model = model_eval['model'], X = X, y = y, 
                               params = model_eval['parameters'], metric = model_eval['metric'], 
                               n_splits = model_eval['n_splits'])
    model_eval['results'] = {}
    model_eval['results']['values'] = results
    model_eval['results']['mean'] = np.mean(results)
    model_eval['results']['median'] = np.median(results)
    model_eval['results']['std'] = np.std(results)