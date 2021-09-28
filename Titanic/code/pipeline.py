#!/usr/bin/env python3
import json
import pickle
from data_acquisition_processing import *
from ml_bricks import *

from hyperopt import hp, fmin, tpe, STATUS_OK



def model_grid_search():
    models_eval = []

    model = {'metric': 'mean_squared_log_error', 'n_splits': 5}
    models = ["RandomForestClassifier"]
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

def pipeline_defined_models():

    # get data
    X_train, X_test, y_train = get_data()
    X_train, y_train, X_val, y_val = split_2(X_train, y_train, train_size = 1)

    # models to evaluate
    models_eval = model_grid_search()

    # models_evaluation
    for model_eval in models_eval:
        evaluate(X_train, y_train, model_eval = model_eval)

    # get best model
    best_model = chose_model(models_eval)
    print('Mean {0} score: {1}'.format(best_model['results']['metric'], best_model['results']['mean']))
    if type(best_model['model']) == str:
        model = eval(best_model['model'])
    else:
        model = best_model['model']

    # retrain best model on whole train data

    model_reg = regression(model, X_train, y_train, params= best_model['parameters'])

    # save model
    filename = '{0}_{1}_{2:.3f}.sav'.format(best_model['model'], best_model['metric'], best_model['results']['mean'] * 100)
    pickle.dump(model_reg, open(filename, 'wb'))

    # get submission file
    y_pred = model_reg.predict(X_test)
    pd.DataFrame({'PassengerId':X_test.index, 'Survived':y_pred}).set_index('PassengerId').to_csv('submission.csv')

def hyperparameter_tuning(params):
    cleanings = [
        {'processus': 'drop_na', 'variables' :{'percent' : params.pop('drop_na')}},
        {'processus': 'fill_na', 'variables' :{'numeric': params.pop('fill_na'), 'string': 'Null'}}
    ]
    X_train, X_test, y_train = get_data(cleanings)
    X_train, y_train, _, _ = split_2(X_train, y_train, train_size = 1)

    params['n_estimators'] = int(params['n_estimators'])
    metric = eval(params.pop('metric'))
    model = eval(params.pop('model'))

    model = model(**params, n_jobs = -1)
    k_fold = KFold(n_splits = 5)

    loss = np.mean([metric(y_train[val], model.fit(X_train[train], y_train[train]).predict(X_train[val]))
            for train, val in k_fold.split(X_train)])

    return {"loss": loss, "status": STATUS_OK}

def pipeline_random_search(max_evals):
    space = {
        'drop_na' : hp.quniform("drop_na", 0, 100, 1),
        'fill_na' : hp.choice("fill_na", ['mean', 'flag']),
        'criterion' : hp.choice("criterion", ['entropy', 'gini']),
        'metric' : 'accuracy_score',
        'model': 'RandomForestClassifier',
        "n_estimators": hp.quniform("n_estimators", 1, 1000, 1),
        "max_depth": hp.quniform("max_depth", 1, 64,1),
        "max_features" : hp.choice('max_features', ['auto', 'log2', 'sqrt']),
    }
    best = fmin( fn=hyperparameter_tuning, space = space, algo=tpe.suggest,
    max_evals = max_evals)

    best['max_features'] = ['auto', 'log2', 'sqrt'][best['max_features']]
    best['fill_na'] = ['mean', 'flag'][best['fill_na']]
    best['n_estimators'] = int(best['n_estimators'])

    cleanings= [
    {'processus': 'drop_na', 'variables' :{'percent' : best.pop('drop_na')}},
    {'processus': 'fill_na', 'variables' :{'numeric': best.pop('fill_na'), 'string': 'Null'}}
    ]
    X_train, X_test, y_train = get_data(cleanings)
    X_train, y_train, _, _ = split_2(X_train, y_train, train_size = 1)

    model_cla = regression(RandomForestClassifier, X_train, y_train, params= best)
    # get submission file
    y_pred = model_cla.predict(X_test)
    pd.DataFrame({'PassengerId':X_test.index, 'SalePrice':y_pred}).set_index('PassengerId').to_csv('submission.csv')

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
