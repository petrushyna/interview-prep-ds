import pandas as pd
import numpy as np

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
