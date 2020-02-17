from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

def log_reg(X_train, X_test, y_train, y_test, **kwargs): 
    if(kwargs):
        log_reg = LogisticRegression(penalty = kwargs['penalty'], 
                                solver = kwargs['solver'], 
                                l1_ratio= kwargs['l1_ratio'], 
                                max_iter = kwargs['max_iter'])
    else:
        log_reg = LogisticRegression()    
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print(precision_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))