from pipeline import t_catToOneHot
from model import SplitTrainTest as split
from model import metrics
from model import curves, rf
from model import SplitTrainTest as stt
from language import tf_idf

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer


import pandas as pd
import numpy as np
import interview_utility as ut
from sklearn.pipeline import Pipeline, FeatureUnion
from pipeline import t_listToOneHot

r_recipes = pd.read_csv('/home/svetik/Notebooks/interview/food-com/RAW_recipes.csv')
r_recipes['steps'] = r_recipes['steps'].apply(lambda x : x.replace("'", '').
                                                 replace('[', '').
                                                 replace(']','').
                                                 split(','))
pip = Pipeline([
    ("stepsToColumns", ut.ColStrColumns(columns = ['steps'], columnsToAdd = 20)), 
    ("mostFrequentWordOneHotEncoded", tf_idf.freqToOneHot())
])
output = pip.transform(r_recipes[:50])

#join information from other files
r_recipes = r_recipes[['id']].join(output, how = 'inner')
interaction_df = pd.read_csv('/home/svetik/Notebooks/food-com/interactions_train.csv')
interaction_rating_df = interaction_df.groupby(['recipe_id']).mean()['rating']
interaction_rating_df = pd.DataFrame(interaction_rating_df)
interaction_rating_df.index.rename('id')
r_r = r_recipes.join(interaction_rating_df, on = 'id', how = 'inner')
r_r.index = r_r.id
r_r = r_r.drop(columns = ['id'])
r_r['rating'] = r_r['rating'].apply(lambda x: round(x))
#r_r.to_pickle("r_r.pickle")
#r_r = pd.read_pickle("r_r.pickle")
r_r = r_r[r_r['rating'] > 3]

pip = Pipeline([(
        "encodeCategoryWithLabelBinarizer", t_catToOneHot.MyLabelBinarizer(columns = ['rating'])
    )])
r_r['rating'] = pip.transform(r_r)

#predict based on last 20 steps
[stat_train_list, stat_test_list] = stt.stratified_split(r_r, r_r['rating'], 'rating', test_size = 0.1, n_splits = 1)

length_nSplits = len(stat_train_list)
for split in range(0, length_nSplits):
    stat_train = stat_train_list[split]
    stat_test = stat_test_list[split]
    X_train = stat_train.drop(columns = 'rating')
    y_train = stat_train['rating']
    X_test = stat_test.drop(columns = 'rating')
    y_test = stat_test['rating']

    print(y_train.value_counts())
    print(y_test.value_counts())
    [y_pred, rf_clf] = rf.run_rf(X_train, X_test, y_train, y_test, class_weight = 'balanced')
    #curves.plot_learning_curves(rf_clf, X_train, X_test, y_train, y_test, metric = 'recall')

