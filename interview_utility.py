import pandas as pd
import numpy as np

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