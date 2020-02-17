from pipeline import t_stringColumnsToList
from pipeline import t_listToOneHot
from pipeline import t_catToOneHot, t_numTransform
from model import SplitTrainTest as split
from model import metrics
from model import linearRegression as lr
from model import curves, rf
from model import SplitTrainTest as stt
from language import tf_idf
from sklearn.linear_model import LinearRegression, SGDRegressor

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer


import pandas as pd
import numpy as np
import interview_utility as ut
from sklearn.pipeline import Pipeline, FeatureUnion
from pipeline import t_listToOneHot

# r_recipes = pd.read_csv('/home/svetik/Notebooks/interview/food-com/RAW_recipes.csv')

# r_recipes['steps'] = r_recipes['steps'].apply(lambda x : x.replace("'", '').
#                                                  replace('[', '').
#                                                  replace(']',''))
#                                                  #.
#                                                  #split(', '))

# #print(tf_idf.rank_words(r_recipes, 'steps', ngram_range = (1,2), numOfNgrams = 100))

# #separate each step to column - greedy?
# r_recipes = r_recipes[:30008]
# r_recipes_len = r_recipes.shape[0]
# output = pd.DataFrame([])
# for row in range(0, r_recipes_len):
#     a = pd.DataFrame(r_recipes.steps.apply(lambda x: x.replace("'", '').split(',')).loc[row]).T
#     output = output.append(a.loc[:,:20])
#     print(row)


# #output_limited = output.loc[:,:20]
# output_limited = output
# output_width = output_limited.shape[1]
# output_limited = output_limited.fillna('0')
# output_limited.index = range(0,output.shape[0])
# newVoc = tf_idf.define_new_voc(['abov', 'afterward', 'alon', 'alreadi', 'alway', 
#                                 'ani', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 
#                                 'becaus', 'becom', 'befor', 'besid', 'cri', 'describ', 
#                                 'dure', 'els', 'elsewher', 'empti', 'everi', 'everyon', 
#                                 'everyth', 'everywher', 'fifti', 'forti', 'henc', 'hereaft', 
#                                 'herebi', 'howev', 'hundr', 'inde', 'mani', 'meanwhil', 'minut',
#                                 'moreov', 'nobodi', 'noon', 'noth', 'nowher', 'onc', 'onli', 
#                                 'otherwis', 'ourselv', 'perhap', 'pleas', 'sever', 'sinc', 
#                                 'sincer', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 
#                                 'themselv', 'thenc', 'thereaft', 'therebi', 'therefor', 'togeth', 
#                                 'twelv', 'twenti', 'veri', 'whatev', 'whenc', 'whenev', 'wherea', 
#                                 'whereaft', 'wherebi', 'wherev', 'whi', 'yourselv', 'anywh', 'el', 
#                                 'elsewh', 'everywh', 'ind', 'otherwi', 'plea', 'somewh', 'f'])
# main_words = []
# for col in range(0,output_width):
#     print(col)
#     main_words.append(list(tf_idf.rank_words(output_limited, col, ngram_range = (1,2), numOfNgrams = 5, stop_words = newVoc, tokenizer = True).term.values))

# print(main_words)
# for col in range(0, output_width):
#     cv = CountVectorizer(analyzer=lambda x: x)
#     output_limited2 = output_limited.iloc[:][col].apply(lambda x : x.split(' '))
#     test = cv.fit_transform(output_limited2.to_list())
#     test_columns = [x for x in cv.get_feature_names()]

#     X_onehotencoded = pd.DataFrame(test.toarray(), columns = test_columns)
#     a = list(set(list(X_onehotencoded.columns)).intersection(set(main_words[col])))

#     output_limited = output_limited.join(X_onehotencoded[a], rsuffix = "_"+str(col) + "step")


# print(output_limited[a].sum())
# output_limited_columns = list(output_limited.columns)
# for i in range(0,21):
#     output_limited_columns.remove(i)

# r_recipes = r_recipes[['id']].join(output_limited[output_limited_columns])
# interaction_df = pd.read_csv('/home/svetik/Notebooks/food-com/interactions_train.csv')
# interaction_rating_df = interaction_df.groupby(['recipe_id']).mean()['rating']
# interaction_rating_df = pd.DataFrame(interaction_rating_df)
# interaction_rating_df.index.rename('id')
# r_r = r_recipes.join(interaction_rating_df, on = 'id', how = 'inner')
# r_r.index = r_r.id
# r_r = r_r.drop(columns = ['id'])
# # r_r_columns = list(r_r.columns)
# # r_r_columns.remove('id')
# # r_r = r_r[r_r_columns]
# r_r['rating'] = r_r['rating'].apply(lambda x: round(x))
# r_r.to_pickle("r_r.pickle")
r_r = pd.read_pickle("r_r.pickle")
r_r = r_r[r_r['rating'] > 3]

[stat_train_list, stat_test_list] = stt.stratified_split(r_r, r_r['rating'], 'rating', test_size = 0.1, n_splits = 15)

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
    rf.run_rf(X_train, X_test, y_train, y_test, class_weight = 'balanced')

