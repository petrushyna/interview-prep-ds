from pipeline import t_stringColumnsToList
from pipeline import t_listToOneHot
from pipeline import t_catToOneHot, t_numTransform
from model import SplitTrainTest as split
from model import metrics
from model import linearRegression as lr, svm
from model import curves
from model import logistic_regression as log_reg
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


import pandas as pd
import numpy as np
import interview_utility as ut

df_kickstarter = pd.read_csv('/home/svetik/Notebooks/interview/kickstarter/ks-projects-201801.csv', parse_dates=['deadline', 'launched'])


df_kickstarter = df_kickstarter[(df_kickstarter['state'] == 'successful') | (df_kickstarter['state'] == 'failed')]

col_for_classification = ['state', 'category', 'main_category', 'backers', 'country', 'usd_pledged_real', 'usd_goal_real']
df_kickstarter = df_kickstarter[col_for_classification]
df_kickstarter_index = df_kickstarter.index
pip_cat = Pipeline([
    ('categoryToOneHot', t_catToOneHot.catToOneHot(columns = ['category', 'main_category', 'country']))
])
pip_label = Pipeline([
        ('binaryLabel', t_catToOneHot.MyLabelBinarizer(columns = ['state']))
])
pip_num = Pipeline([
    ("avoidInf", t_numTransform.avoidInf(columns = ['backers', 'usd_pledged_real', 'usd_goal_real'])),
    ('impute', SimpleImputer(strategy = "mean")), 
    ('scaling', StandardScaler())
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pip", pip_num),
    ("cat_pip", pip_cat),
    ("label_y", pip_label)
])

df_kickstarter = full_pipeline.fit_transform(df_kickstarter)
df_kickstarter = pd.DataFrame(df_kickstarter, index = df_kickstarter_index)

df_kickstarter_columns_len = len(df_kickstarter.columns)
df_kickstarter_column = []
for col_num in range(df_kickstarter_columns_len-1):
    df_kickstarter_column.append(col_num)
    
df_kickstarter_column.append('state')
df_kickstarter.columns = df_kickstarter_column

index_to_delete = df_kickstarter[df_kickstarter.isna().any(axis = 1)].index
df_kickstarter = df_kickstarter.drop(index_to_delete)


[X_train, X_test, y_train, y_test] = split.stratified_split(df_kickstarter,df_kickstarter['state'], 'state', 1, test_size = 0.2, )
print(y_train.value_counts())
print(y_test.value_counts())
svm.simple_svm(X_train, X_test, y_train, y_test)
# log_reg.log_reg(X_train, X_test, y_train, y_test, 
# penalty = 'l1', solver = 'liblinear', l1_ratio = 0.8, max_iter = 100)
