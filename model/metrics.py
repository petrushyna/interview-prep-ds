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
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        recall = np.diag(cm) / np.sum(cm, axis = 1)
        precision = np.diag(cm) / np.sum(cm, axis = 0)
        print(precision)
        print(recall)
        return ac
    except: 
        print("y_pred and y_test need to be lists or dataframes")

# def precision_recall(y_pred, y_test):
#     print(precision_score(y_test, y_pred))
#     print(recall_score(y_test, y_pred))

def mse(y_pred, y_test):
    return mean_squared_error(y_pred, y_test)