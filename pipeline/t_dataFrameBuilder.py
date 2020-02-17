from sklearn.base import BaseEstimator, TransformerMixin

class df(BaseEstimator, TransformerMixin):
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self,X, y = 0): 
        return self
    def transform(self, X, y = 0): 
        
        return X_onehot