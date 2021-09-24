
import json
import pickle
from data_acquisition_processing import *
from ml_bricks import *


def get_data():
    X_train, y_train = load_data(path = '../data/train.csv', y_col ='SalePrice', index_col = 'Id')
    X_test = load_data(path = '../data/test.csv', y_col = None, index_col = 'Id')
    train_ids = X_train.index
    test_ids = X_test.index
    X = pd.concat([X_train, X_test])

    X = cleaning(X = X.copy() , processus= [drop_na, fill_na], variables = [{'percent' : 95}, {'numeric': 'mean', 'string': 'Null'}])
    cat_cols = get_categorical_cols(X)
    X = make_categorical(X, cols = cat_cols)
    X_train = X.loc[train_ids]
    X_test = X.loc[test_ids]
    return X_train, X_test, y_train

    

if __name__ == '__main__':
    X_train, X_test, y_train = get_data()
    X_train, y_train, X_val, y_val = split_2(X_train, y_train, train_size = 1)
    with open('models_eval.json', 'r') as f:
        models_eval = json.load(f)['models_eval']
    for model_eval in models_eval:
        evaluate(X_train, y_train, model_eval = model_eval)
    # with open('models_eval2.json', 'w') as f:
    #     json.dump(models_eval, f)
    best_model = chose_model(models_eval)

    if type(best_model['model']) == str:
        model = eval(best_model['model'])
    else:
        model = best_model['model']

    model_reg = regression(model, X_train, y_train, params= best_model['parameters'])
    filename = '{0}_{1}_{2:.0f}.sav'.format(best_model['model'], best_model['metric'], best_model['results']['mean'] * 100)

    pickle.dump(model_reg, open(filename, 'wb'))

    y_pred = model_reg.predict(X_test)
    pd.DataFrame({'id':X_test.index, 'SalePrice':y_pred}).set_index('id').to_csv('submission.csv')


    