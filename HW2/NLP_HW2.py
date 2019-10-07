
# coding: utf-8

# In[2]:

from gensim.models import Word2Vec
import tensorflow
import keras


# In[21]:

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
import glob


# In[4]:

#emoji regex
#start_time = time.time()
emoticon_string = r"(:\)|:-\)|:\(|:-\(|;\);-\)|:-O|8-|:P|:D|:\||:S|:\$|:@|8o\||\+o\(|\(H\)|\(C\)|\(\?\))"


# In[28]:

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


# In[299]:

# Reading all reviews in the training folder
def get_docs():#perform_tokenization = True):

    '''
    Pre-processing: Read the complete data word by word. Remove any markup tags, e.g., HTML
    tags, from the data. Lower case capitalized words (i.e., starts with a capital letter) but not all
    capital words (e.g., USA). Do not remove stopwords. Tokenize at white space and also at each
    punctuation. Consider emoticons in this process. You can use an emoticon tokenizer, if you so
    choose. If yes, specify which one. 
    
    # added to code to add tweets as lists (currently commented out)
    '''

    #initalize train variables
    docs = []
    #create megadocument of all training tweets
    for f in glob.glob('./LanguageModelingData/*.txt'):#os.listdir('LanguageModelingData'):
        #print(f)
        tweet = open(f, encoding="utf8").read()#os.path.join('LanguageModelingData',f)
#        if perform_tokenization:
        docs.extend(tokenize(tweet)) 
#         else:
#             sent = " ".join(tokenize(tweet))
#             docs.append(sent)
#             tweet = re.sub(r'\d+', '', tweet) #remove numbers
#             def lower_repl(match):
#                 return match.group(1).lower()

#             # txt = r"This is a practice tweet :). Let's hope our-system can get it right. \U0001F923 something."
#             tweet = re.sub('(?:<[^>]+>)', '', tweet)# remove html tags
#             tweet = re.sub('([A-Z][a-z]+)',lower_repl,tweet) #lowercase words that start with captial
#             tweet = tweet.replace(".", " .")
#             docs.append(tweet)
    
    # print(docs)
    print("Text Extracted --- %s seconds ---" % (round((time.time() - start_time),2)))
   
    return docs


# In[300]:

docs = get_docs()
## NOTE::
# It throws utf08 encoding error as due to presence of .DS_Store file in mac.. that's why using glob to only take in 
# '.txt' files..


# In[301]:

# joining all words to form entire tweets one after another
tweets = " ".join(docs)


# In[257]:

#tweets = get_docs(perform_tokenization=False)


# In[302]:

tweets


# In[50]:

# Building the positive and negative n-grams
def get_ngrams_vector(docs, num_grams):
    '''
    Construct your n-grams: Create positive n-gram samples by collecting all pairs of adjacent
    tokens. Create 2 negative samples for each positive sample by keeping the first word the same
    as the positive sample, but randomly sampling the rest of the corpus for the second word. The
    second word can be any word in the corpus except for the first word itself. 
    '''
    grams = ngrams(docs,num_grams)
    neg_grams1 = []
    neg_grams2 = []
    pos_grams = []
    for element in grams:
        for i in range(1,3): #get two neg grams
            random_word_in_corpus = element[0] # taking the first word from the pos gram
            while random_word_in_corpus == element[0]: #keep going until it's different from the first word
                random_word_in_corpus = random.choice(docs)
            if i == 1:
                neg_grams1.append((element[0], random_word_in_corpus))
            elif i == 2:
                neg_grams2.append((element[0], random_word_in_corpus))
        pos_grams.append([element[0],element[1]])
    
    print("Grams Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return pos_grams, neg_grams1, neg_grams2


# In[51]:

pos_grams, neg_grams1, neg_grams2 = get_ngrams_vector(docs, 2)


# In[53]:

pos_grams[0:5]


# In[54]:

neg_grams1[0:5]


# In[55]:

neg_grams2[0:5]


# In[86]:

def train_test_sample(pos_list, neg_list1, neg_list2, perct_train = .8):
    
    # Calculating the random index numbers
    trainIndex = random.sample(range(len(pos_grams)), round(len(pos_grams)*.8))
    trainIndex.sort()

    testIndex = list(set(range(len(pos_grams))) - set(trainIndex))
    testIndex.sort()
    
    pos_train = [pos_list[i] for i in trainIndex]
    pos_test = [pos_list[i] for i in testIndex]
    
    neg1_train = [neg_list1[i] for i in trainIndex]
    neg1_test = [neg_list1[i] for i in testIndex]
    neg2_train = [neg_list2[i] for i in trainIndex]
    neg2_test = [neg_list2[i] for i in testIndex]
    
    trainvec = pos_train + neg1_train + neg2_train
    train_labels = [1]*len(pos_train) + [0]*len(neg1_train) + [0]*len(neg2_train)
    
    testvec = pos_test + neg1_test + neg2_test
    test_labels = [1]*len(pos_test) + [0]*len(neg1_test) + [0]*len(neg2_test)
        
    return trainvec, testvec, train_labels, test_labels
#    return pos_train, pos_test, neg1_train, neg1_test, neg2_train, neg2_test


# In[87]:

#pos_train, pos_test, neg1_train, neg1_test, neg2_train, neg2_test = train_test_sample(pos_grams, neg_grams1, neg_grams2, perct_train=.8)
trainvec, test_vec, train_labels, test_labels = train_test_sample(pos_grams, neg_grams1, neg_grams2, perct_train=.8)


# ** Creating input and output using keras tokenizer and text to sequence builder**

# For the purpose to running a Fed forward Neural network model, all words in the vocab need to be assigned a number and we have to represent the entire document using those assigned numbers

# In[96]:

from keras.preprocessing.text import Tokenizer


# In[315]:

tokenizer = Tokenizer(filters='')
#tokenizer.fit_on_sequences(docs)
tokenizer.fit_on_texts(docs)
#encoded = tokenizer.texts_to_sequences([docs])[0]


# In[382]:

# checking the numbers assigned to each token
tokenizer.word_index


# In[326]:

# Encoding the entire tweet wrt to the numeric labels assigned above1
encoded = tokenizer.texts_to_sequences([tweets])[0]


# ** Testing: **

# In[321]:

tokenizer.word_index['this']


# In[322]:

tweets[0:100]


# In[328]:

encoded[0:100]


# ** Recreating the positive and negative n grams using the encoding **

# In[361]:

pos_grams, neg_grams1, neg_grams2 = get_ngrams_vector(encoded, num_grams=2)


# In[365]:

trainvec, test_vec, train_labels, test_labels = train_test_sample(pos_grams, neg_grams1, neg_grams2, perct_train=.8)


# In[362]:

pos_grams[0:5]


# In[364]:

neg_grams2[0:5]


# In[338]:

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


# In[329]:

# checking the length of the encoding
len(encoded)


# In[330]:

# tests:
# sequences = list()
# for i in range(1, len(encoded)):
#     sequence = encoded[i-1:i+1]
#     sequences.append(sequence)
# print('Total Sequences: %d' % len(sequences))


# ### Creating the input and output training dataset for the Neural Network model

# In[366]:

#sequences = np.array(sequences)
sequences = np.array(trainvec)
X, y = sequences[:,0],sequences[:,1]


# In[372]:

print('Total Sequences: %d' % len(sequences))


# In[385]:

len(X)


# ** Note: ** we need to convert the integer to a one hot encoding while specifying the number of classes as the vocabulary size.

# In[369]:

y = np_utils.to_categorical(y, nb_classes= vocab_size) # converting it into a matrix


# checking.... 

# In[368]:

y[1]


# In[376]:

y[0][0:13]


# In[386]:

len(y)


# In[396]:

y.shape


# ### Building the model

# In[414]:

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM


# In[415]:

# Building a Neural Network model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
#model.add(Dense(20, activation='relu'))
# By using a dense layer instead of LSTM, the dimension requirement for the output changes to 3, but input is 2
# currently unable to remove that error, therefore sticking to LSTM.
model.add(LSTM(20))                                
model.add(Dense(vocab_size, activation='sigmoid'))
print(model.summary())


# In[384]:

from keras import optimizers


# In[388]:

SGD = optimizers.SGD(lr=0.01, nesterov=False)


# In[417]:

model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])


# In[ ]:

# Fitting the model
model.fit(X, y, nb_epoch= 100, verbose=2)


# ** Steps Left: **
# 1. Change evaluation functions
# 2. Evaluation
# 3. Accuray Calculations
# 4. Cross validation on training dataset
# 

# ** Links used as reference **
# 1. https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/
# 2. https://machinelearningmastery.com/what-are-word-embeddings/
# 3. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# 4. https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# 5. https://keras.io/optimizers/
# 6. https://keras.io/layers/embeddings/
# 7. https://keras.io/getting-started/sequential-model-guide/
# 8. https://keras.io/layers/recurrent/

# In[ ]:



