"""
AIT726 HW 1 Due 9/26/2019
Sentiment classificaiton using Naive Bayes and Logistic Regression on a dataset of 25000 training and 25000 testing tweets.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman

"""
import os
import pandas as pd
import re
import numpy as np
from string import punctuation
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
import itertools
from sklearn.feature_extraction.text import CountVectorizer

def main():
    '''
    main function
    '''
    vocabulary, vocabulary_stemmed, trainingdocs, trainingdocs_stemmed = get_train_vocab()
    train_BOW_freq, train_BOW_binary, train_BOW_freq_stemmed, train_BOW_binary_stemmed = get_BOW(trainingdocs, trainingdocs_stemmed, vocabulary, vocabulary_stemmed)
    P_positive, P_negative  = get_class_priors()
    wordlikelihood_freq, wordlikelihood_freq_stem, wordlikelihood_binary, wordlikelihood_binary_stem = get_perword_likelihood(vocabulary, vocabulary_stemmed, train_BOW_freq, train_BOW_binary)

def get_train_vocab():
    '''
    Create your Vocabulary: Read the complete training data word by word and create the
    vocabulary V for the corpus. You must not include the test set in this process. Remove any
    markup tags, e.g., HTML tags, from the data. Lower case capitalized words (i.e., starts with a
    capital letter) but not all capital words (e.g., USA). Keep all stopwords. 

    Create 2 versions of V:
    with stemming and without stemming. You can use appropriate tools in nltk1 to stem. Stem at
    white space and also at each punctuation. In other words, “child’s” consists of two tokens “child
    and ‘s”, “home.” consists of two tokens “home” and “.”. 

    Consider emoticons in this process. You can use an emoticon tokenizer, if you so choose. If yes, specify which one.
    '''

    def tokenize(txt,stem=False):
        """
        Tokenizer that tokenizes text. Can also stem words.
        :param text:
        :return:
        """
        def lower_repl(match):
            return match.group(1).lower()

        txt = re.sub('<[^<]+?>', '', txt)# remove html tags
        txt = re.sub('([A-Z][a-z]+)',lower_repl,txt) #lowercase words that start with captial 
        tokens = TweetTokenizer().tokenize(txt) #emoticon stemming 
        if stem:
            stemmer = PorterStemmer()
            stemmed = [stemmer.stem(item) for item in tokens]
            tokens = stemmed
        
        return tokens

    #initalize variables
    trainingdocs = []
    trainingdocs_stemmed = []
    vocabulary = []
    vocabulary_stemmed = []

    #create megadocument of all training tweets stemmed and not stemmed
    for folder in os.listdir('Data/train'):
        for f in os.listdir(os.path.join('Data/train',folder)):
            tweet = open(os.path.join('Data/train',folder,f),encoding="utf8").read()
            # print(tweet)
            trainingdocs.append(tokenize(tweet,False)) #don't stem
            # trainingdocs_stemmed.append(tokenize(tweet,True))# stem

    #raw python to get unique vocabulary words from the mega training documents
    vocabulary = list(set(list(itertools.chain.from_iterable(trainingdocs))))
    # print(vocabulary[:100])
    vocabulary_stemmed = list(set(list(itertools.chain.from_iterable(trainingdocs_stemmed))))
    # print(vocabulary_stemmed[:100])

    #using sklearn to convert mega documents to feature vectors and get vocab  -- is this cheating?
    # vec = CountVectorizer(tokenizer=lambda x: x,lowercase = False)
    # count_feature_vector = vec.fit_transform(trainingdocs)
    # vocabulary = vec.get_feature_names()

    # vec = CountVectorizer(tokenizer=lambda x: x,lowercase = False)
    # count_feature_vector_stemmed = vec.fit_transform(trainingdocs_stemmed)
    # vocabulary_stemmed = vec.get_feature_names()

        
    return vocabulary, vocabulary_stemmed, trainingdocs, trainingdocs_stemmed

def get_BOW(trainingdocs, trainingdocs_stemmed, vocabulary, vocabulary_stemmed):
    '''
    Extract Features: Convert documents to vectors using Bag of Words (BoW) representation. Do
    this in two ways: keeping frequency count where each word is represented by its count in each
    document, keeping binary representation that only keeps track of presence (or not) of a word in
    a document.
    '''
    #gettings frequency based bag of words feature vectors -- currently too slow...
    train_BOW_freq = pd.DataFrame(0,index=np.arange(25000),columns=vocabulary)
    print(train_BOW_freq.head())
    i=0
    for twt in trainingdocs:
        i+=1
        for word in twt:
            if word in vocabulary:
                train_BOW_freq.loc[train_BOW_freq.index[i]][word] = train_BOW_freq.loc[train_BOW_freq.index[i]][word] + 1
                # print(i)
                # print(word)
                # print(train_BOW_freq.loc[train_BOW_freq.index[i]][word])
    # print(train_BOW_freq.head())

    #these will be similar to above...
    train_BOW_binary = pd.DataFrame()
    train_BOW_freq_stemmed = pd.DataFrame()
    train_BOW_binary_stemmed = pd.DataFrame()

    return train_BOW_freq, train_BOW_binary, train_BOW_freq_stemmed, train_BOW_binary_stemmed

def get_class_priors():
    '''
    calculate the prior for each class = number of samples of class C in training set / total number of samples in training set (25000)
    Pˆ(c) = Nc/N
    '''
    return P_positive, P_negative 


def get_perword_likelihood(vocabulary, vocabulary_stemmed, train_BOW_freq, train_BOW_binar):
    '''
    Pˆ(w | c) = count(w, c)+1 /(count(c)+ |V|)  

    depends on vocabulary being stemmed/non-stemmed and the type of vectors being used
    '''


    return wordlikelihood

if __name__ == "__main__":
    main()