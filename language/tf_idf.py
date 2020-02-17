from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag
from nltk import RegexpParser


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

