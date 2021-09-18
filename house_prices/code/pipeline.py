import json
import pickle
from data_acquisition_processing import *
from ml_bricks import *

def get_data():
    X, y = load_data(path = '../data/train.csv', y_col ='SalePrice', index_col = 'Id')
    X = cleaning(X = X.copy() , processus= [drop_na, fill_na], variables = [{'percent' : 95}, {'numeric': 'mean', 'string': 'Null'}])
    cat_cols = get_categorical_cols(X)
    X = make_categorical(X, cols = cat_cols)
    return X, y


    

if __name__ == '__main__':
    X, y = get_data()
    X_train, y_train, X_test, y_test = split_2(X, y, train_size = 0.9, random_state = 42)
    with open('models_eval.json', 'r') as f:
        models_eval = json.load(f)['models_eval']
    for model_eval in models_eval:
        evaluate(X_train, y_train, model_eval = model_eval)
    with open('models_eval.json', 'w') as f:
        json.dump(models_eval, f)
    best_model = chose_model(models_eval)

    if type(best_model['model']) == str:
        model = eval(best_model['model'])
    else:
        model = best_model['model']

    model_reg = regression(model, X_train, y_train, params= best_model['parameters'])
    filename = '{}_{}_{0:.3f}.sav'.format(best_model['model'], best_model['metric'], best_model['results']['mean'])

    pickle.dump(model, open(filename, 'wb'))

    