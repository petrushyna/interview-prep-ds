from sklearn.model_selection import train_test_split

def simple_split(X,y,test_size):
    X_train, X_test, y_train, y_test = train_test_split( X, y , test_size = test_size , random_state = 42)
    return [X_train, X_test, y_train, y_test]

def stratified_split(X,y, columnToPredict, n_splits = 1, test_size = 0.2):
    from sklearn.model_selection import StratifiedShuffleSplit
    stat_train = []
    stat_test = []
    split = StratifiedShuffleSplit(n_splits = n_splits,test_size = test_size, random_state = 42)
    for train_index, test_index in split.split(X, y): 
        stat_train.append(X.iloc[train_index])
        stat_test.append(X.iloc[test_index])

    # X_train = stat_train.drop(columns = columnToPredict)
    # y_train = stat_train[columnToPredict]
    # X_test = stat_test.drop(columns = columnToPredict)
    # y_test = stat_test[columnToPredict]
    return [stat_train, stat_test]
