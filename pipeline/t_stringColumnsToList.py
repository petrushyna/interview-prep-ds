from pipeline import strToList as sl
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

class colToList(BaseEstimator, TransformerMixin): 
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self, X):
        return self
    def transform(self, X): 
      for column in self.columns:
        X[column] = X[column].apply(lambda x : sl.strToList(x, ','))
      return X