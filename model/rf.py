from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from model import metrics
import numpy as np

def getBestModel(train_df, test_df, train_labels, test_labels, model):
    param_grid = {}
    param_grid = { "max_depth": [3, 6]}
    #"max_features": sp_randint(1000, 2000)}
    
    r_search = RandomizedSearchCV(model, param_grid, n_iter = 20, 
                    scoring='f1_macro', cv = 3, verbose = 2)
    r_search.fit(train_df, train_labels)

    cvres = r_search.cv_results_
    for params in cvres['params']: 
        print(params)
    
    print(r_search.best_estimator_)
    model = r_search.best_estimator_
        
    labels_prediction = model.fit(train_df, train_labels)

    return model

def predict_rf(model, X_test,  y_test):
    y_pred = model.predict(X_test)
    print(metrics.accuracy_true_pred(y_pred, y_test))
    if(len(np.unique(y_test)) == 2):
        metrics.precision_recall_binary(y_pred,y_test)
    else:
        metrics.precision_recall_multiclass(y_pred,y_test)
    return y_pred

def run_rf(X_train, X_test, y_train, y_test,
            bootstrap = False, max_depth = 5,
            n_estimators = 10, class_weight = None):

    rf_clf = RandomForestClassifier(bootstrap = bootstrap, max_depth = max_depth,
                n_estimators = n_estimators)
    if(class_weight):
        rf_clf = RandomForestClassifier(bootstrap = bootstrap, max_depth = max_depth,
                n_estimators = n_estimators, class_weight = class_weight)
    rf_clf = getBestModel(X_train, X_test, y_train, y_test, rf_clf)
    
    y_pred = predict_rf(rf_clf, X_test, y_test)
    return [y_pred, rf_clf]

