from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def split(X, y, random_state):
    y2 = y.to_numpy().reshape([-1]) #prefered by the sklearn algorithms
    X_train, X_test, y_train, y_test = train_test_split(
        X, y2, test_size=0.2, random_state = random_state)
    return X_train, X_test, y_train, y_test

def regression(model, X_train, y_train, params):
    '''
    Example: 
    regr = regression(model= RandomForestRegressor, X_train=X_train, y_train=y_train, params={'max_depth': 15, 'random_state': 42})
    '''
    model_reg = model()
    model_reg.set_params(**params)
    model_reg.fit(X_train, y_train)
    return model_reg
