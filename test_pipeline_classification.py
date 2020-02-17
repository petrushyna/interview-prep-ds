from pipeline import t_stringColumnsToList
from pipeline import t_listToOneHot
from pipeline import t_catToOneHot, t_numTransform
from model import SplitTrainTest as split
from model import metrics
from model import linearRegression as lr
from model import curves
from model import logistic_regression as log_reg
from sklearn.linear_model import LinearRegression, SGDRegressor

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer



import pandas as pd
import numpy as np
import interview_utility as ut


def transform_stringToList(df):
    pip = Pipeline([
         ('stringsToList', t_stringColumnsToList.colToList(columns = ['ingredients']))
    ])
    transformed_df = pip.transform(df[:10])

    print("The whole df ")
    print(transformed_df)

    print('Only transformed column ')
    print(transformed_df['ingredients'])

    return transformed_df

def transform_listToOneHot(df):
    pip = Pipeline([
         ('listToOneHot',t_listToOneHot.listElemToOneHot(columns = ['ingredients']))
    ])

    transformed_df = pip.transform(df)
    print("The ingredients column with data in the list is transformed into a set of one hot encoded columns")
    print(transformed_df)

def transform_InfValues(df, columns = []): 
    pip = Pipeline([
        ("avoidInfValues", t_numTransform.avoidInf(columns = columns))
        ])

    return pip.transform(df)

def transform_numVariables(df, columns = []): 
    df = df[columns]
    pip = Pipeline([
        ("avoidInfValues", t_numTransform.avoidInf(columns = columns)), 
        ('imputer', SimpleImputer(strategy = 'median') ),
        ( 'std_scaler', StandardScaler() ) 
        ])

    return pd.DataFrame(pip.fit_transform(df), columns = columns) 

def transform_catToOneHot(df, columns = []):
    pip = Pipeline([(
                    'catToOneHot', t_catToOneHot.catToOneHot(columns = columns)
    )])
    return pip.transform(df)

def transform_catLabelBinarizer(df, columns = []): 
    
    cat_encoder = df[columns]

    pip = Pipeline([(
        "encodeCategoryWithLabelBinarizer", t_catToOneHot.labelBinarizer()
    )])
    return pip.transform(cat_encoder)



df_kickstarter = pd.read_csv('/home/svetik/Notebooks/interview/kickstarter/ks-projects-201801.csv', parse_dates=['deadline', 'launched'])


df_kickstarter = df_kickstarter[(df_kickstarter['state'] == 'successful') | (df_kickstarter['state'] == 'failed')]

col_for_classification = ['state', 'category', 'main_category', 'backers', 'country', 'usd_pledged_real', 'usd_goal_real']
df_kickstarter = df_kickstarter[col_for_classification]

pip_cat = Pipeline([
    ('categoryToOneHot', t_catToOneHot.catToOneHot(columns = ['category', 'main_category', 'country']))
])
pip_label = Pipeline([
        ('binaryLabel', t_catToOneHot.MyLabelBinarizer(columns = ['state']))
])
df_kickstarter_cat = pip_cat.transform(df_kickstarter)
df_kickstarter_cat['state'] = pip_label.transform(df_kickstarter)
#new_col = df_kickstarter_cat.columns.remove('backers', 'usd_pledged_real', 'usd_goal_real')

pip_num = Pipeline([
    ("avoidInf", t_numTransform.avoidInf(columns = ['backers', 'usd_pledged_real', 'usd_goal_real'])),
    ('impute', SimpleImputer(strategy = "mean")), 
    ('scaling', StandardScaler())
])
df_kickstarter_num = pd.DataFrame(pip_num.fit_transform(df_kickstarter), 
                                            columns = ['backers', 'usd_pledged_real', 'usd_goal_real'], 
                                            index = df_kickstarter_cat.index)

df_kickstarter = df_kickstarter_cat.join(df_kickstarter_num, rsuffix = '_')

index_to_delete = df_kickstarter[df_kickstarter.isna().any(axis = 1)].index
print(df_kickstarter.loc[331675])
df_kickstarter = df_kickstarter.drop(index_to_delete)
print(df_kickstarter.shape)
# [X_train, X_test, y_train, y_test] = split.stratified_split(df_kickstarter,df_kickstarter['state'], 'state', 1, test_size = 0.2, )
# print(y_train.value_counts())
# print(y_test.value_counts())

# log_reg.log_reg(X_train, X_test, y_train, y_test, 
# penalty = 'l1', solver = 'liblinear', l1_ratio = 0.8, max_iter = 100)
