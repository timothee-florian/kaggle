#!/usr/bin/env python3
import json
import pickle
from data_acquisition_processing import *
from ml_bricks import *



def model_grid_search():
    models_eval = []

    model = {'metric': 'mean_squared_error', 'n_splits': 5}
    models = ["RandomForestRegressor"]
    max_depth = [2, 6, 18, 54]
    n_estimators = [10, 100, 1000]
    max_features = ['auto', 'log2', 'sqrt']
    n_jobs = -1
    for i in range(len(models)):
        m = model.copy()
        m['model'] = models[i]
        m['parameters'] = {}
        for j in range(len(max_depth)):
            for k in range(len(n_estimators)):
                for l in range(len(max_features)):
                    m['parameters']['n_jobs'] = n_jobs
                    m['parameters']['max_depth'] = max_depth[j]
                    m['parameters']['n_estimators'] = n_estimators[k]
                    m['parameters']['max_features'] = max_features[l]
                    models_eval.append(m)
    return models_eval



if __name__ == '__main__':
    X_train, X_test, y_train = get_data()
    X_train, y_train, X_val, y_val = split_2(X_train, y_train, train_size = 1)
    with open('models.json', 'r') as f:
        models_eval = json.load(f)['models_eval']
    for model_eval in models_eval:
        evaluate(X_train, y_train, model_eval = model_eval)
    with open('models_evaluation.json', 'w') as f:
        json.dump(models_eval, f)
    best_model = chose_model(models_eval)
    print('Mean {0} score: {1}'.format(best_model['results']['metric'], best_model['results']['mean']))
    
    if type(best_model['model']) == str:
        model = eval(best_model['model'])
    else:
        model = best_model['model']

    model_reg = regression(model, X_train, y_train, params= best_model['parameters'])
    filename = '{0}_{1}_{2:.0f}.sav'.format(best_model['model'], best_model['metric'], best_model['results']['mean'] * 100)

    pickle.dump(model_reg, open(filename, 'wb'))

    y_pred = model_reg.predict(X_test)
    pd.DataFrame({'id':X_test.index, 'SalePrice':y_pred}).set_index('id').to_csv('submission.csv')
