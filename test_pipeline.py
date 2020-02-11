from pipeline import strToList as sl
from pipeline import t_stringColumnsToList
from pipeline import t_listToOneHot
from sklearn.pipeline import Pipeline

import pandas as pd

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
    print(transformed_df)

r_recipes = pd.read_csv('/home/svetik/Notebooks/interview-prep-ds/food-com/r_recipes.csv')
print(r_recipes.columns)
print(sl.strToList('[4,5,6]', ','))

transformed_recipes = transform_stringToList(r_recipes)
print(transform_listToOneHot(transformed_recipes))