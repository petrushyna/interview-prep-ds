from pipeline import strToList as sl
from pipeline import t_stringColumnsToList
from pipeline import t_listToOneHot
from sklearn.pipeline import Pipeline

import pandas as pd

# def transform_stringToList(df):
#     pip = Pipeline([
#          ('stringsToList', t_stringColumnsToList.colToList(columns = ['ingredients']))
#     ])
#     transformed_df = pip.transform(df[:10])

#     print("The whole df ")
#     print(transformed_df)

#     print('Only transformed column ')
#     print(transformed_df['ingredients'])

#     return transformed_df

# def transform_listToOneHot(df):
#     pip = Pipeline([
#          ('listToOneHot',t_listToOneHot.listElemToOneHot(columns = ['ingredients']))
#     ])

#     transformed_df = pip.transform(df)
#     print("The ingredients column with data in the list is transformed into a set of one hot encoded columns")
#     print(transformed_df)

# r_recipes = pd.read_csv('/home/svetik/Notebooks/interview-prep-ds/food-com/r_recipes.csv')
# print("Columns of the current of the dataframe")
# print(r_recipes.columns)

# print("result of strToList")
# print(sl.strToList('[4,5,6]', ','))

# transformed_recipes = transform_stringToList(r_recipes)

# transform_listToOneHot(transformed_recipes)

df_kickstarter = pd.read_csv('/home/svetik/Notebooks/interview-prep-ds/kickstarter/ks-projects-201801.csv', parse_dates=['deadline', 'launched'])
print(df_kickstarter.dtypes)
df_kickstarter_cat = df_kickstarter.select_dtypes(include = ['object']).copy()
print(df_kickstarter_cat.shape)
inexWithNull = df_kickstarter_cat[df_kickstarter_cat.isnull().any(axis = 1)].index
df_kickstarter_cat = df_kickstarter_cat.drop(inexWithNull)
print(df_kickstarter_cat.shape)

df_kickstarte_cat_onehot = pd.get_dummies(df_kickstarter_cat, columns=['category', 'main_category', 'currency', 'state', 'country'])
print(df_kickstarte_cat_onehot.columns)
print(df_kickstarter_cat['state'].value_counts())