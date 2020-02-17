from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNet
from sklearn.pipeline import Pipeline

def simple_reg(X_train, y_train, X_test):
    pipeline_m = Pipeline( steps = [ ( 'LR_model', LinearRegression() ) ] )
    pipeline_m.fit( X_train, y_train )

    y_pred = pipeline_m.predict( X_test ) 
    return y_pred

def stochastic_reg(X_train, y_train, X_test, n_iteration = 10, penalty = None, eta = 0.1):
    pipeline_m = Pipeline( steps = [ ( 'sgd_Regressor', SGDRegressor(max_iter = n_iteration,
                                                             penalty = penalty, eta0 = eta ) ) ] )
    pipeline_m.fit( X_train, y_train )

    y_pred = pipeline_m.predict( X_test ) 
    return y_pred

def elastic_net(X_train, y_train, X_test, alpha = 0.1, l1_ratio = 0.5):
    pipeline_m = Pipeline( steps = [ ( 'elasticNet_Regressor', ElasticNet(alpha = alpha,
                                                                    l1_ratio= l1_ratio ) ) ] )
    pipeline_m.fit( X_train, y_train )

    y_pred = pipeline_m.predict( X_test ) 
    return y_pred