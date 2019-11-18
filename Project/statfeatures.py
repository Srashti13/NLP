# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:06:40 2019
@author: nickn
"""

#%% Quora Insincere Questions


import os
import numpy as np
import pandas as pd
import zipfile
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
import string
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#%%
localfolder = 'kaggle/input/quora-insincere-questions-classification/'
train = pd.read_csv(localfolder + 'train.csv', nrows=10000) # , nrows=50000 remove for whole dataset

#%%

train = train.iloc[:,1:]

#%%
train.head()
X = train['question_text']
y = train['target']
#%% Majority class baseline accuracy

prediction = np.zeros(len(train),)

#%% Since the class imbalance is so high. Merely predicting the negative class
# results in an accuracy of almost 94%

accuracy = accuracy_score(y, prediction)

#%% This same method yields a 0 F1-score because precision and recall are both 0. This
# is due to the method not predicting any positive cases

f1_score(y, prediction)

#%% 

#%%
# creating a mapping for contractions to be tokenized
contractions ={
        "'s":"is",
        "'re":"are",
        "'ve":"have",
        "n't":"not"
        }

class TextPreprocess(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # fixing contractions
        X = X.apply(lambda r: [contractions[word] if word in contractions else word for word in word_tokenize(r.lower())]) 
        # removing stopwords
        stop_words = set(stopwords.words('english'))
        X = X.apply(lambda a: [t for t in (a) if t not in stop_words])
        # removing punctuation
        X = X.apply(lambda a: [t for t in a if t not in string.punctuation])
        # stemming the words
        porter_stemmer = PorterStemmer()
        X = X.apply(lambda r: [porter_stemmer.stem(word) for word in r])
        X = pd.Series([" ".join(i) for i in X])
        return X
    

#create feature vectors
class StatFeatureVectorizer(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        features = pd.Dataframe()
        
        #do stuff here to make vectors
        print(X)
        return X

# using a dense transformer so that the Gaussian NB can read in the data      
class DenseTransformer(TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.todense()
#%%

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2,
                                                                  stratify=y) 
#%%


LR_pipeline = Pipeline([
        ('PreProcess', TextPreprocess()),
        ('features', StatFeatureVectorizer()), #used to reduce dimensions pick this one or 'dense'
        # ('dense', DenseTransformer()), #comment out if svd performed
        ('clf', LogisticRegression(n_jobs=-1, solver='lbfgs'))])
    
#%% Running Logistic Regression cross validation
    
# LR_scores = cross_val_score(LR_pipeline, X_train, y_train, cv=2, scoring="f1_macro")   
# print(LR_scores)

#error analysis
print('\n-------------- TFIDF LR -----------------')
CVLR = GridSearchCV(LR_pipeline,{},scoring="f1_macro", cv=2, n_jobs=1)
CVLR.fit(X_train,y_train)
y_pred = CVLR.predict(X_valid)

printed = 0
for (i, label) in enumerate(y_pred):
    if label != y_valid.iloc[i]:
        if printed < 20:
            print(X_valid.iloc[i].encode("utf-8"))
            print('predicted: %s' % (y_pred[i]))
            print('labeled: %s' % (y_valid.iloc[i]))
            printed +=1
results = precision_recall_fscore_support(y_valid,y_pred,average ='macro')
print('\n\nprecision, accuracy, recall:')
print(results) # precision , recall, fscore


#https://www.kaggle.com/evanmiller/pipelines-gridsearch-awesome-ml-pipelines