import matplotlib.pyplot as plt
from model import metrics
import numpy as np

def plot_learning_curves(model, X_train, X_test, y_train, y_test, metric = "mse"):
    train_errors = []
    test_errors = []
    train_errors_pr = []
    test_errors_pr = []
    train_errors_r = []
    test_errors_r = []
    for m in range(1000, len(X_train), 100):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test[:m])
        
        if(metric == 'mse'):
            train_errors.append(metrics.mse(y_train_predict, y_train[:m]))
            test_errors.append(metrics.mse(y_test_predict, y_test[:m]))
        elif((metric == 'precision') | (metric == 'recall')):
            [precision_train, recall_train] = metrics.precision_recall(y_train_predict, y_train[:m])
            [precision_test, recall_test] = metrics.precision_recall(y_test_predict, y_test[:m])
            train_errors_pr.append(precision_train)
            test_errors_pr.append(precision_test)
            train_errors_r.append(recall_train)
            test_errors_r.append(recall_test)
    if(metric == 'mse'):
        plt.plot(np.sqrt(train_errors), 'r+-', linewidth = 4, label = 'Train errors')
        plt.plot(np.sqrt(test_errors), 'b-', linewidth = 4, label = 'Test errors')
    elif(metric == 'precision'):
         plt.plot(train_errors_pr, 'r+-', linewidth = 4, label = 'Train errors')
         plt.plot(test_errors_pr, 'b-', linewidth = 4, label = 'Test errors')
    elif(metric == 'recall'):
         plt.plot(train_errors_r, 'r+-', linewidth = 4, label = 'Train errors')
         plt.plot(test_errors_r, 'b-', linewidth = 4, label = 'Test errors')
    plt.savefig('learning_curves.png')