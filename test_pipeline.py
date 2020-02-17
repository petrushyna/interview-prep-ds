from pipeline import t_stringColumnsToList
from pipeline import t_listToOneHot
from pipeline import t_catToOneHot, t_numTransform
from model import SplitTrainTest as split
from model import metrics
from model import linearRegression as lr
from model import curves

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

    pip = Pipeline([("encodeCategoryWithLabelBinarizer", t_catToOneHot.MyLabelBinarizer())])
    return pip.transform(cat_encoder)


#r_recipes = pd.read_csv('/home/svetik/Notebooks/interview-prep-ds/food-com/r_recipes.csv')
r_recipes = pd.read_csv('/home/svetik/Notebooks/helloFresh_tf_dataset.csv')
#print("Columns of the current of the dataframe")
print(r_recipes.columns)
print(r_recipes.dtypes)

categorical_pipeline = Pipeline( steps = [  
    ('categoriesToOneHot', t_catToOneHot.catToOneHot(columns = ['Unit', 'n_ingredients', 'n_names']))
])
numerical_pipeline = Pipeline( steps = [  
        ("avoidInfValues", t_numTransform.avoidInf(columns = ['price'])), 
        ('imputer', SimpleImputer(strategy = 'median') ),
        ( 'std_scaler', StandardScaler() ),
        ('vecToDF', t_numTransform.vecToDF(columns = ['price'])) 
        ])

recipes_num = numerical_pipeline.fit_transform(r_recipes)
print(recipes_num)

recipes_cat = categorical_pipeline.transform(r_recipes) 
recipes_num = pd.DataFrame(numerical_pipeline.fit_transform(r_recipes), columns = ['price'], index = recipes_cat.index)

recipes = recipes_cat.join(recipes_num, how = "left", rsuffix = '_transformed')
print(recipes.columns)
recipes_old = recipes[recipes['score_transformed'].notna()]
columns_X = ['Unit_0.25', 'Unit_0.5', 'Unit_0.75', 'Unit_1.0',
       'Unit_2.0', 'Unit_4.0', 'Unit_5.0', 'Unit_6.0', 'Unit_8.0', 'Unit_9.0',
       'Unit_9.6', 'Unit_10.0', 'Unit_11.0', 'Unit_12.0', 'Unit_13.4',
       'Unit_16.0', 'Unit_24.0', 'Unit_25.0', 'n_ingredients_4',
       'n_ingredients_5', 'n_ingredients_6', 'n_ingredients_7',
       'n_ingredients_8', 'n_ingredients_9', 'n_ingredients_10',
       'n_ingredients_11', 'n_ingredients_12', 'n_ingredients_13',
       'n_ingredients_14', 'n_names_2', 'n_names_3', 'n_names_4', 'n_names_5',
       'n_names_6', 'n_names_7', 'n_names_8', 'n_names_9', 'n_names_10',
       'n_names_11', 'n_names_12', 'n_names_13', 'price_transformed']
columns_Y = ['score']
X = recipes_old[columns_X]
y = recipes_old[columns_Y]

recipes_new = recipes[recipes['score'].isna()]

# X_train = r_recipes_old[['Unit', 'n_ingredients', 'price', 'n_names']]
# y_train = r_recipes_old['score']
# X_test = r_recipes_new[['Unit', 'n_ingredients', 'price', 'n_names']]
# y_test = r_recipes_new['score']

#full_pipeline.fit( X_train, y_train )


 

# check the usage of split strategies 
# simple strategy
# [X_train, X_test, y_train, y_test] = split.simple_split(X,y, test_size = 0.2)

# pipeline_m.fit( X_train, y_train )

# y_pred = pipeline_m.predict( X_test ) 
# print(y_pred)


# stratified in case the classes are imbalanced to keep these imbalancing
y = np.round(y)
X = X.join(y)
[X_train, X_test, y_train, y_test] = split.stratified_split(X,y, 'score', 1, test_size = 0.2, )

#y_pred = lr.simple_reg(X_train, y_train, X_test)
y_pred = lr.stochastic_reg(X_train, y_train, X_test, 100, "l1", 0.15)
#y_pred = lr.elastic_net(X_train, y_train, X_test, l1_ratio = 0.1)

# model = SGDRegressor(max_iter = 100, penalty = None, eta0 = 0.1 )
# curves.plot_learning_curves(model, X_train, y_train, X_test, y_test)

y_pred = pd.DataFrame(np.round(y_pred.flatten()), index = X_test.index)

print(metrics.accuracy_pred_test(y_pred, y_test))

#print((res[res == True].shape[0])/(res.shape[0]))
#print(res.
#print(y_test)
#print(X_test)
#y_test = y_test.apply(lambda x : int(x))
# #print(y_test)
# y_pred = pd.DataFrame(np.round(y_pred.flatten()), index = X_test.index)
# print(y_pred.values.flatten())
# print(y_pred.dtypes)
# print(y_test.dtypes)
# print(np.round(y_test).values.flatten())
# #print(np.round(y_test.values.flatten()))
# #print([np.round(y_pred), np.round(y_test.values)

#transformed_recipes = transform_stringToList(r_recipes)

#transform_listToOneHot(transformed_recipes)
#print(r_recipes.shape[0])
#print(transform_InfValues(r_recipes, columns = ['score']))
#print(r_recipes[r_recipes['minutes'] == np.nan].shape[0])

# transfomedNumVariables = transform_numVariables(r_recipes, columns = ['score', 'price'])
# for col in transfomedNumVariables.columns:
#     print(transfomedNumVariables[col].describe())

    
# print(ut.infToNan(pd.DataFrame([[5,6,4],[np.Inf,8,9]], columns = ['1', '2', '3'])))





#df_kickstarter = pd.read_csv('/home/svetik/Notebooks/interview-prep-ds/kickstarter/ks-projects-201801.csv', parse_dates=['deadline', 'launched'])


# df_kickstarter_cat = df_kickstarter.select_dtypes(include = ['object']).copy()

# df_kickstarter_categoryEncoded = transform_catToOneHot(df_kickstarter, columns = ['category'])
# print(df_kickstarter_cat.head())

# main_cat_encoder = transform_catLabelBinarizer(df_kickstarter_cat, columns = ['main_category'])
# print(main_cat_encoder.head())