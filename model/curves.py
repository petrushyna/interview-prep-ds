import matplotlib.pyplot as plt
from model import metrics
import numpy as np

def plot_learning_curves(model, X_train, y_train, X_test, y_test):
    train_errors = []
    test_errors = []
    for m in range(10, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test[:m])
        train_errors.append(metrics.mse(y_train_predict, y_train[:m]))
        test_errors.append(metrics.mse(y_test_predict, y_test[:m]))
    plt.plot(np.sqrt(train_errors), 'r+-', linewidth = 4, label = 'Train errors')
    plt.plot(np.sqrt(test_errors), 'b-', linewidth = 4, label = 'Test errors')
    plt.savefig('learning_curves.png')