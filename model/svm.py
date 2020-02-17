from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score

def simple_svm(X_train, X_test, y_train, y_test): 
    svm_clf = LinearSVC(C = 10, loss = "hinge")

    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    print(precision_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
