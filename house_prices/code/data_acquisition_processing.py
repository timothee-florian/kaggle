#!/usr/bin/env python3

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import numpy as np
import sys

def load_data(path, y_col, index_col = None):
    data = pd.read_csv(path, index_col = index_col)
    cols =set(data.columns)
    cols.remove(y_col)
    x_cols = list(cols)
    X = data[x_cols]
    y = data[[y_col]]
    return X, y

def cleaning(X, processus, variables):
    for i in range(len(processus)):
        X = processus[i](X.copy(), variables[i])
    return X

def drop_na(X, percent):
    X = X.dropna(axis=1, thresh=int(X.shape[0]*percent['percent']/100))
    return X

def fill_na(X, rules):
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            if rules['numeric'] == 'mean':
                v= X[col].mean()
                X[col].fillna(v, inplace = True)
            elif rules['numeric'] == 'median':
                v =X[col].median()
                X[col].fillna(v, inplace = True)
            else:
                X[col].fillna(rules['numeric'], inplace = True)
        else:
            X[col].fillna(rules['string'], inplace = True)
    return X

def get_categorical_cols(df):
    cat_cols = list(filter(lambda col: not is_numeric_dtype(df[col]), df.columns))
    return cat_cols

def make_categorical(X, cols):
    X = pd.concat([X, pd.get_dummies(X[cols])], axis =1).drop(cols, axis =1)
    return X

def main():
    X, y = load_data(path = '../data/train.csv', y_col ='SalePrice', index_col = 'Id')
    X2 = cleaning(X = X.copy() , processus= [drop_na, fill_na], variables = [{'percent' : 95}, {'numeric': 'mean', 'string': 'Null'}])
if __name__ == "__main__":
    main()