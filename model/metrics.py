from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import numpy as np

def accuracy_true_pred(y_test, y_pred):
    '''
    @param
    y_pred and y_test should be lists
    '''
    try:
        ac = accuracy_score(y_test, y_pred)
        return ac
    except: 
        print("y_pred and y_test need to be lists or dataframes")

def precision_recall(y_pred, y_test):
    if(len(np.unique(y_pred)) == 2):
        return precision_recall_binary(y_pred, y_test)
    elif(len(np.unique(y_pred)) > 2):
        return precision_recall_multiclass(y_pred, y_test)

def precision_recall_binary(y_pred, y_test):
    precision = precision_score(y_test, y_pred)
    print(precision)
    recall = recall_score(y_test, y_pred)
    print(recall)
    return [precision, recall]

def precision_recall_multiclass(y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    recall = np.mean(np.diag(cm) / np.sum(cm, axis = 1))
    precision = np.mean(np.diag(cm) / np.sum(cm, axis = 0))
    print(precision)
    print(recall)
    return [precision, recall]

def mse(y_pred, y_test):
    return mean_squared_error(y_pred, y_test)