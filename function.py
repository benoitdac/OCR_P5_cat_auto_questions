import os
from pathlib import Path
import re

from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from operator import itemgetter
import collections
from collections import Counter,defaultdict


from joblib import parallel_backend, Parallel, delayed , dump , load
from joblib import wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer

from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF, PCA

import gensim
from gensim.models import TfidfModel
import gensim.corpora as corpora
from gensim import models
from gensim.utils import simple_preprocess
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.models import EnsembleLda

import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim_models 

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

def clean_word(raw_review, lem = True , Porter=True):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    
    # 2. Remove non-letters and Convert to lower case, split into individual words
    words = re.findall(r'(?u)\b[a-zA-Z][a-zA-Z+#.]*\b[+#]*', review_text.lower())
    words = [w for w in words if (len(w)>2)|(w[0]=='c')|(w[0]=='r')]

    # 3. Stem or Lem all the words   
    if lem == True : 
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in words]
    else : 
        Stem = PorterStemmer() if Porter == True else EnglishStemmer()
        words = [Stem.stem(word) for word in words]

    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
     
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    meaningful_words = set(meaningful_words)
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return " ".join(meaningful_words)  

def words_vectorizer(DF,vocabulary = None ):
    vectorizer = CountVectorizer(analyzer='word',vocabulary = vocabulary ,token_pattern = r'(?u)\b[a-zA-Z][a-zA-Z+#.]*\b[+#]*')
    data_features = vectorizer.fit_transform(DF)
    vocab = vectorizer.get_feature_names() if vocabulary is None else vocabulary
    DF_features = pd.DataFrame(data_features.toarray() , columns = vocab)  

def words_vectorizer_TF_IDF(DF,vocabulary = None ):
    vectorizer = TfidfVectorizer(analyzer='word',vocabulary = vocabulary ,token_pattern = r'(?u)\b[a-zA-Z][a-zA-Z+#.]*\b[+#]*')
    data_features = vectorizer.fit_transform(DF)
    vocab = vectorizer.get_feature_names() if vocabulary is None else vocabulary
    DF_features = pd.DataFrame(data_features.toarray() , columns = vocab)  

class SupervisedModel:

    def __init__(self):

        tf_model = "./models/words_tf_model.joblib"
        tfidf_model = "./models/tfidf_model.joblib"

        vocabulary = "./models/vocabulary.joblib"
        vocabulary_tfidf = "./models/vocabulary.joblib"
                
        pca_model = "./models/pca_model.joblib"
        pca_model_tfidf = "./models/pca_tfidf_model.joblib"
        
        svm_model = "./models/svm_model.joblib"
        svm_model_tfidf = "./models/svm_model_tfidf.joblib"

        mlb_model = "./models/mlb_model.joblib"

        self.svm_model = load(svm_model)
        self.svm_model_tfidf = load(svm_model_tfidf)
        self.mlb_model = load(mlb_model)
        self.tf_model = load(tf_model)
        self.tfidf_model = load(tfidf_model)
        self.pca_model = load(pca_model)
        self.pca_model_tfidf = load(pca_model_tfidf)
        self.vocabulary = load(vocabulary)
        self.vocabulary_tfidf = load(vocabulary_tfidf)

    def predict_tags(self, text , tf_idf = False):
        text = clean_word(text, lem = True , Porter=True)
        if tf_idf == True : 
            input_vector = self.tfidf_model.transform(text)
            input_vector = pd.DataFrame(input_vector.toarray(), columns=self.vocabulary_tfidf)
            input_vector = self.pca_model_tfidf.transform(input_vector)
            resultat = self.svm_model_tfidf.predict(input_vector)
        else : 
            input_vector = self.tf_model.transform(text)
            input_vector = pd.DataFrame(input_vector.toarray(), columns=self.vocabulary)
            input_vector = self.pca_model.transform(input_vector)
            resultat = self.svm_model.predict(input_vector)
            
        resultat = self.mlb_model.inverse_transform(resultat)
        resultat = list({tag for list_tags in resultat for tag in list_tags if (len(tag_list) != 0)})
        # resultat = [tag for tag  in resultat if tag in text]
        
        return resultat