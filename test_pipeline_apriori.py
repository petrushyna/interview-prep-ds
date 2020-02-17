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
from sklearn.pipeline import Pipeline
from pipeline import t_listToOneHot
import apriori

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

recipesSteps = ut.dfWithZeroNullOnly(r_r.drop(columns = ['rating']))
freqItems = apriori.detect_freq_items(recipesSteps, 
                            min_percentage = 0.01, max_len = 5)

rules = apriori.get_association_rules(freqItems)

seq_ = rules[(rules.confidence > 0.5) & (rules.lift > 2) & (rules.iloc[:,2] > 0.01)][['antecedents', 'consequents']]

print(len(apriori.allSequencesToList(seq_)))

