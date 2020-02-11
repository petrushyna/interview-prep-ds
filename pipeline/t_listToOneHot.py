import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

class listElemToOneHot(BaseEstimator, TransformerMixin): 
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self, X):
        return self
    
    def transform(self, X): 
        cv = CountVectorizer(analyzer=lambda x: x)
        for column in self.columns:
            test = cv.fit_transform(X[column].to_list())
            test_columns = [x for x in cv.get_feature_names()]
            
            X_onehotencoded = pd.DataFrame(test.toarray(), columns = test_columns, index = X.index)

            X = X.join(X_onehotencoded, how = "inner")

        return X