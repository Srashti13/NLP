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
from sklearn.model_selection import train_test_split
import random



#emoji regex
start_time = time.time()
emoticon_string = r"(:\)|:-\)|:\(|:-\(|;\);-\)|:-O|8-|:P|:D|:\||:S|:\$|:@|8o\||\+o\(|\(H\)|\(C\)|\(\?\))"
#https://www.regexpal.com/96995

def main():
    '''
    The main function 
    '''
    print("Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    docs, sentences = get_docs()
    vector, labels = get_ngrams_vector(docs,sentences,2)
    train, test = get_embedded_traintest(docs, vector, labels)
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
    sentences = []
    #create megadocument of all training tweets
    for f in os.listdir('LanguageModelingData'):
            tweet = open(os.path.join('LanguageModelingData',f),encoding="utf8").read()
            token_tweets = tokenize(tweet)
            docs.extend(token_tweets) 
            sentences.append(token_tweets)

    # print(docs)
    print("Text Extracted --- %s seconds ---" % (round((time.time() - start_time),2)))

   
    return docs, sentences 

def get_ngrams_vector(docs, sentences, num_grams):
    '''
    Construct your n-grams: Create positive n-gram samples by collecting all pairs of adjacent
    tokens. Create 2 negative samples for each positive sample by keeping the first word the same
    as the positive sample, but randomly sampling the rest of the corpus for the second word. The
    second word can be any word in the corpus except for the first word itself. 
    '''
    
    ## creating the ngrams from each sentence, so that ngrams from the end
    # of one sentence aren't combined with the beginning of another sentence
    ngram_list = []
    for sentence in sentences:
        for gram in ngrams(sentence, 2):
            ngram_list.append(gram)
            
    # dictionary with the key being the first term in the bigram and the values
    # being the second terms
    ngram_dict = defaultdict(list)
    for gram in ngram_list:
        ngram_dict[gram[0]].append(gram[1]) 
    
    # creating fake ngrams by ensuring that the value is not a duplicate of the
    # key and is also not one of the values for a correct bigram
    fakegrams = []     
    for gram in ngram_list:
        # get two fake ngrams for each correct one
        for _ in range(2): 
            random_word = random.choice(docs)
            while random_word == gram[0] or random_word in ngram_dict[gram]:
                random_word = random.choice(docs)
            fakegrams.append((gram[0], random_word))
    


    gramvec, labels = [], []
    for element in ngram_list:
        gramvec.append([element[0],element[1]])
        labels.append(1)
    for element in fakegrams:
        gramvec.append([element[0],element[1]])
        labels.append(0)

    # print(fakegrams)
    # print([b for b in grams])
    # print(vector)
    print("Grams Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return gramvec, labels

def get_embedded_traintest(docs, gramvec, labels):
    '''
    put grams into embedding format and make tst and train set for model
    '''
    # vocab = set(docs) for manual encoding...
    # word_to_ix = {word: i for i, word in enumerate(vocab)}
    # print(word_to_ix['the'])

    #load some twitter embeddings #https://github.com/RaRe-Technologies/gensim-data
    print("Loading Twitter Embeddings --- %s seconds ---" % (round((time.time() - start_time),2)))
    import gensim.downloader as api
    model = api.load("glove-twitter-25")
    print("Twitter Embeddings Loaded --- %s seconds ---" % (round((time.time() - start_time),2)))
    print("Encoding Train/Test Vectors --- %s seconds ---" % (round((time.time() - start_time),2)))
    embeddingvec = []
    print(len(gramvec))
    for element in gramvec:
        try:
            embeddingvec.append([model[element[0]],model[element[1]]])
        except KeyError:
            model.add(element[0],np.random.rand(25,1),replace=False)#https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.Word2VecKeyedVectors
            model.add(element[1],np.random.rand(25,1),replace=False)
            embeddingvec.append([model[element[0]],model[element[1]]])
    print(embeddingvec)
    print("Encoded Train/Test Vectors --- %s seconds ---" % (round((time.time() - start_time),2)))
    return train, test, word_to_ix
  

if __name__ == "__main__":
    main()