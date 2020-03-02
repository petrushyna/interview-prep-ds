
from mlxtend.frequent_patterns import apriori, association_rules
from nltk.tokenize import RegexpTokenizer

def detect_freq_items(df, min_percentage = 0.01, max_len = 1):
    freqItems = apriori(df, 
                        min_support = min_percentage, 
                        max_len = max_len, use_colnames=True, verbose=1)
    return freqItems

def allSequencesToList(seq_):
    seq_['antecedents'] = seq_['antecedents'].apply(lambda x : str(list(x)).replace('[', '').replace(']', '').replace('"', ''))
    seq_['consequents'] = seq_['consequents'].apply(lambda x : str(list(x)).replace('[', '').replace(']', '').replace('"', ''))
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+[_]*[0-9]*[a-zA-Z]+')
    seq_steps_ = []
    for i in range(0,seq_.shape[0]):
        a = tokenizer.tokenize(seq_.iloc[i]['antecedents'].lower())
        b = tokenizer.tokenize(seq_.iloc[i]['consequents'].lower())
        a.extend(b)
        seq_steps_.append(a)
    return seq_steps_

def get_association_rules(freq_items, min_confidence = 1):
    rules = association_rules(freq_items, 
    metric="confidence", min_threshold=min_confidence)
    return rules