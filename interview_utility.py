import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def deleteNullValues(X): 
    indexWithNull = X[X.isnull().any(axis = 1)].index
    print("The following percentage of rows is deleted since catefory values are zero "+ str(len(indexWithNull)/X.shape[0]))
    return X.drop(indexWithNull)


def strToList(list_l, splitSymbol):
    """
    @param

    list_l      =  input string in form ['4,6,6']
    splitSymbol =  is the symbol between future list elements
    """

    list_l = list_l.split(splitSymbol)
    temp = list()
    for l in list_l: 
        l = l.replace("[",'').replace("]",'').replace("'", '').replace(" ", '')
        temp.append(l)
    return temp

def infToNan(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    return X

class ColStrColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns = [], columnsToAdd= 5):
        self.columns = columns
        self.columnsToAdd = columnsToAdd
    def fit(self,X): 
        return self
    def transform(self, X): 
        X_len = X.shape[0]
        output = pd.DataFrame([])
        for row in range(0, X_len):
            a = pd.DataFrame(list(X[self.columns].loc[row].values.flatten()[0])).T
            output = output.append(a.loc[:,:self.columnsToAdd])
        return output

def dfWithZeroNullOnly(df):
    count = 0
    for col in df.columns:
        count += len(df[df[col]>1].index)
        df = df.drop(df[df[col]>1].index)

    print(str(count) + " values are deleted")
    return df

