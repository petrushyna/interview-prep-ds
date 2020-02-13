import interview_utility as ut
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class avoidInf(BaseEstimator, TransformerMixin):
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self,X): 
        return self
    def transform(self, X):
        columns = self.columns
        if(self.columns == []):
            columns = X.columns
        for col in columns:
            X[col] = ut.infToNan(X[col])
        return X[columns]
class vecToDF(BaseEstimator, TransformerMixin):
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self,X): 
        return self
    def transform(self, X):
        return pd.DataFrame(X, columns = self.columns)