from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag
from nltk import RegexpParser

from sklearn.base import BaseEstimator, TransformerMixin

def tokenize(text):
    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')
    stems = [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

    # try:
    #     tag = pos_tag(stems)

    #     chunkGram = r"""Chunk: {<NN.?>+}"""
    #     chunkParser = RegexpParser(chunkGram)

    #     chunked = chunkParser.parse(tag)
    #     stemsLength = len(stems) 
    #     result = []
    #     for word in range(0, stemsLength):
    #         if(tag[0][1] == 'VB'):
    #             result.append(stems[word])
    #     return result
    # except Exception as e:
    #     print(str(e))

    return stems

def define_new_voc(listOfWords):
    return text.ENGLISH_STOP_WORDS.union(listOfWords)

def rank_words(df, text_column, ngram_range = (1,2), numOfNgrams = 15, stop_words = 'english', tokenizer = False):
    # if(tokenizer): 
    #     vectorizer = TfidfVectorizer(ngram_range = ngram_range,  
    #                                 stop_words=stop_words, tokenizer = tokenize)
    # vectorizer = TfidfVectorizer(ngram_range = ngram_range,  
    #                                 stop_words=stop_words)

    vectorizer = TfidfVectorizer(ngram_range = ngram_range,  
                                    stop_words=stop_words, tokenizer = tokenize)

    tf_idf_df = vectorizer.fit_transform(df[text_column]) 
    features = (vectorizer.get_feature_names()) 
    scores = (tf_idf_df.toarray()) 
        
    # Getting top ranking features 
    sums = tf_idf_df.sum(axis = 0) 
    ranking = [] 
        
    for col, term in enumerate(features): 
        ranking.append( (term, sums[0,col] )) 

    ranking_df = pd.DataFrame(ranking, columns = ['term','rank']) 
    words = (ranking_df.sort_values('rank', ascending = False))[:numOfNgrams] 
    return words

class freqToOneHot(BaseEstimator, TransformerMixin):
        def fit(self,X): 
            return self
        def transform(self, X):
            #most frequent words in each step to one hot encoding
            X_width = X.shape[1]
            X = X.fillna('0')
            X.index = range(0,X.shape[0])
            newVoc = define_new_voc(['abov', 'afterward', 'alon', 'alreadi', 'alway', 
                                            'ani', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 
                                            'becaus', 'becom', 'befor', 'besid', 'cri', 'describ', 
                                            'dure', 'els', 'elsewher', 'empti', 'everi', 'everyon', 
                                            'everyth', 'everywher', 'fifti', 'forti', 'henc', 'hereaft', 
                                            'herebi', 'howev', 'hundr', 'inde', 'mani', 'meanwhil', 'minut',
                                            'moreov', 'nobodi', 'noon', 'noth', 'nowher', 'onc', 'onli', 
                                            'otherwis', 'ourselv', 'perhap', 'pleas', 'sever', 'sinc', 
                                            'sincer', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 
                                            'themselv', 'thenc', 'thereaft', 'therebi', 'therefor', 'togeth', 
                                            'twelv', 'twenti', 'veri', 'whatev', 'whenc', 'whenev', 'wherea', 
                                            'whereaft', 'wherebi', 'wherev', 'whi', 'yourselv', 'anywh', 'el', 
                                            'elsewh', 'everywh', 'ind', 'otherwi', 'plea', 'somewh', 'f'])
             
            main_words = []
            for col in range(0,X_width):
                print(col)
                main_words.append(list(rank_words(X, col, ngram_range = (1,2), numOfNgrams = 5, stop_words = newVoc, tokenizer = True).term.values))
            print(main_words)
            #encode words in 20 steps
            for col in range(0, X_width):
                cv = CountVectorizer(analyzer=lambda x: x)
                X2 = X.iloc[:][col].apply(lambda x : x.split(' '))
                test = cv.fit_transform(X2.to_list())
                test_columns = [x for x in cv.get_feature_names()]

                X_onehotencoded = pd.DataFrame(test.toarray(), columns = test_columns)
                a = list(set(list(X_onehotencoded.columns)).intersection(set(main_words[col])))

                X = X.join(X_onehotencoded[a], rsuffix = "_"+str(col) + "step")
            X_columns = list(X.columns)
            for i in range(0,21):
                X_columns.remove(i)

            return X[X_columns]

