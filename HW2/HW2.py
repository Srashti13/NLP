"""
AIT726 HW 2 Due 10/10/2019
Sentiment classificaiton using Naive Bayes and Logistic Regression on a dataset of 25000 training and 25000 testing tweets.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW1.py 
i. main - runs all of the functions
    ii. get_trainandtest_vocabanddocs() - converts dataset into tokens (stemmed and unstemmed), creates megatraining document and extracts vocabulary


"""
import os
import re
import time
import numpy as np
import itertools
from nltk.util import ngrams
from nltk import word_tokenize
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import random



#emoji regex
start_time = time.time()
emoticon_string = r"(:\)|:-\)|:\(|:-\(|;\);-\)|:-O|8-|:P|:D|:\||:S|:\$|:@|8o\||\+o\(|\(H\)|\(C\)|\(\?\))"
#https://www.regexpal.com/96995

def main():
    '''
    The main function 
    '''
    print("Start Program --- %s minutes ---" % (round((time.time() - start_time)/60,2)))
    docs = get_docs()
    ngrams, fakegrams = get_ngrams(docs,2)
    return

def get_docs():

    '''
    Pre-processing: Read the complete data word by word. Remove any markup tags, e.g., HTML
    tags, from the data. Lower case capitalized words (i.e., starts with a capital letter) but not all
    capital words (e.g., USA). Do not remove stopwords. Tokenize at white space and also at each
    punctuation. Consider emoticons in this process. You can use an emoticon tokenizer, if you so
    choose. If yes, specify which one. 

    '''

    def tokenize(txt):
        """
        Tokenizer that tokenizes text. Can also stem words.
        """
        txt = re.sub(r'\d+', '', txt) #remove numbers
        def lower_repl(match):
            return match.group(1).lower()

        # txt = r"This is a practice tweet :). Let's hope our-system can get it right. \U0001F923 something."
        txt = re.sub('(?:<[^>]+>)', '', txt)# remove html tags
        txt = re.sub('([A-Z][a-z]+)',lower_repl,txt) #lowercase words that start with captial
        tokens = re.split(emoticon_string,txt) #split based on emoji faces first 
        tokensfinal = []
        for i in tokens:
            if not re.match(emoticon_string,i):
                to_add = word_tokenize(i)
                tokensfinal = tokensfinal + to_add
            else:
                tokensfinal.append(i)
        # tokensfinal.insert(0, '_start_')
        # tokensfinal.append('_end_')
        return tokensfinal

    #initalize train variables
    docs = []

    #create megadocument of all training tweets
    for f in os.listdir('LanguageModelingData'):
            tweet = open(os.path.join('LanguageModelingData',f),encoding="utf8").read()
            docs.extend(tokenize(tweet)) 

    # print(docs)
    print("Train Docs Prepared --- %s seconds ---" % (round((time.time() - start_time)/60,2)))

   
    return docs 

def get_ngrams(docs, num_grams):
    '''
    Construct your n-grams: Create positive n-gram samples by collecting all pairs of adjacent
    tokens. Create 2 negative samples for each positive sample by keeping the first word the same
    as the positive sample, but randomly sampling the rest of the corpus for the second word. The
    second word can be any word in the corpus except for the first word itself. 
    '''
    grams = ngrams(docs,num_grams)
    fakegrams = []
    for element in grams:
        for _ in range(2): #get two fake grams
            random_word_in_corpus = element[0]
            while random_word_in_corpus == element[0]: #keep going until it's different from the first word
                random_word_in_corpus = random.choice(docs)
            fakegrams.append((element[0], random_word_in_corpus))

    # print(fakegrams)
    # print([b for b in grams])
    return grams, fakegrams
if __name__ == "__main__":
    main()