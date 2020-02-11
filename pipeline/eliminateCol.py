import pandas as pd
import numpy as np

def avoidLessFrequentColumns(df):
            col = list(df.columns)
            
            sum_item_columns = pd.DataFrame(df[col].sum(axis = 0)/df.shape[0])
            
            sum_freq_item_columns = list(sum_item_columns[sum_item_columns[0]>0.01].index)

            if ((sum_item_columns.shape[0] - len(sum_item_columns)) == 0):
                return df

            return df[sum_freq_item_columns]