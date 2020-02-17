from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
import interview_utility as ut

import pandas as pd

class catToOneHot(BaseEstimator, TransformerMixin):
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self,X, y = 0): 
        return self
    def transform(self, X, y = 0): 
        X = ut.deleteNullValues(X)
        
        columns=X.columns
        if(self.columns != []):
            X = X[self.columns]
            columns = X.columns
            
        X_onehot = pd.get_dummies(X, columns=columns)
        return X_onehot

class MyLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self,X): 
        return self
    def transform(self, X): 
        columns = X.columns
        if(self.columns !=[]):
            X = X[self.columns]
        X = ut.deleteNullValues(X)
        lb_bin = LabelBinarizer()
        main_cat_encoder_results = lb_bin.fit_transform(list(X.values.flatten()))
        return pd.DataFrame(main_cat_encoder_results, columns = self.columns, index = X.index)