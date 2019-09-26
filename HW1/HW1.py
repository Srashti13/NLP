"""
AIT726 HW 1 Due 9/26/2019
Sentiment classificaiton using Naive Bayes and Logistic Regression on a dataset of 25000 training and 25000 testing tweets.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW1.py 
i. main - runs all of the functions
    ii. get_trainandtest_vocabanddocs() - converts dataset into tokens (stemmed and unstemmed), creates megatraining document and extracts vocabulary
    iii. get_vectors() - creates BOW and TFIDF vectors for test and train both stemmed and unstemmed
    iv. get_class_priors() - calculates the class prior likelihoods for use in Naive Bayes predictions
    v. get_perword_likelihood() - calculates dictionaries for each feature vector to be used in the Naive Bayes prediction calculation
    vi. predict_NB() - predicts the class of all of the test documents for all of the feature vectors using Naive Bayes
    vii. evaluate - returns accuracy and confusion matrix for predictions 
    viii. Logistic_Regression_L2_SGD - logistic regression model class used to create the model and form predictions on test vectors

Due to the size of the dataset, and the number of tokens we are required to keep, many of the operations when creating vectors utilize a large amount of RAM. 
This code was tested on a machine with 64GB of DDR4 RAM. Variables are deleted throughout when they are not needed to save memory. Needed data structures are saved and loaded
for later use.
"""
import os
import pandas as pd
import re
import time
import numpy as np
import itertools
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import string
import pickle

#emoji regex
start_time = time.time()
emoticon_string = r"(:\)|:-\)|:\(|:-\(|;\);-\)|:-O|8-|:P|:D|:\||:S|:\$|:@|8o\||\+o\(|\(H\)|\(C\)|\(\?\))"
#https://www.regexpal.com/96995

#pickeling code for saving objects
def save_obj(name, obj ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    '''
    The main function can be utilized to create vectors, vocabulary, and likelihoods for predictions and evaluation. Users may 
    also comment out the creation steps and merely evaluate the result on the saved data.
    '''
    print("Start Program --- %s minutes ---" % (round((time.time() - start_time)/60,2)))

    # create vocabulary, tokenize dataset, create vectors, and store perword likelihoods by class
    get_trainandtest_vocabanddocs()
    get_vectors()
    P_positive, P_negative  = get_class_priors()
    get_perword_likelihood()
    predictions = predict_NB(P_positive, P_negative)

    #evaluate results for Naive Bayes
    y_test = np.load('Stored/DocsVocab/y_test.npy')
    evaluate(predictions[0], y_test, "NB-NOSTEM-FREQ")
    evaluate(predictions[1], y_test, "NB-NOSTEM-BINARY")
    evaluate(predictions[2], y_test, "NB-NOSTEM-TFIDF")
    evaluate(predictions[3], y_test, "NB-STEM-FREQ")
    evaluate(predictions[4], y_test, "NB-STEM-BINARY")
    evaluate(predictions[5], y_test, "NB-STEM-TFIDF")
    
    #Logistic Regression
    #nostem no l2
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=1, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/trainbow_freq.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-bowfreq-noL2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/testbow_freq.npy'))
    evaluate(predictions, y_test, "LOGISTIC_FREQ_NOL2")
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=1, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/trainbow_binary.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-bowbinary-noL2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/testbow_binary.npy'))
    evaluate(predictions, y_test, "LOGISTIC_BINARY_NOL2")
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=1, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/train_tfidf.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-tfidf-noL2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/test_tfidf.npy'))
    evaluate(predictions, y_test, "LOGISTIC_TFIDF_NOL2")

    #stem no l2
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=1, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/trainbow_stem_freq.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-bowfreq-stem-noL2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/testbow_stem_freq.npy'))
    evaluate(predictions, y_test, "LOGISTIC_FREQ_STEM_NOL2")
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=1, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/trainbow_stem_binary.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-bowbinary-stem-noL2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/testbow_stem_binary.npy'))
    evaluate(predictions, y_test, "LOGISTIC_BINARY_STEM_NOL2")
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=1, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/train_tfidf_stem.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-tfidf-stem-noL2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/test_tfidf_stem.npy'))
    evaluate(predictions, y_test, "LOGISTIC_TFIDF_STEM_NOL2")

    #l2
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=0.1,l2=5, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/trainbow_freq.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-bowfreq-L2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/testbow_freq.npy'))
    evaluate(predictions, y_test, "LOGISTIC_FREQ_L2")
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=0.1,l2=5, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/trainbow_binary.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-bowbinary-L2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/testbow_binary.npy'))
    evaluate(predictions, y_test, "LOGISTIC_BINARY_L2")
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=0.1,l2=5, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/train_tfidf.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-tfidf-L2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/test_tfidf.npy'))
    evaluate(predictions, y_test, "LOGISTIC_TFIDF_L2")

    #stem l2
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=0.1,l2=5, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/trainbow_stem_freq.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-bowfreq-stem-L2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/trainbow_stem_freq.npy'))
    evaluate(predictions, y_test, "LOGISTIC_FREQ_STEM_L2")
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=0.1,l2=5, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/trainbow_stem_binary.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-bowbinary-stem-L2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/trainbow_stem_binary.npy'))
    evaluate(predictions, y_test, "LOGISTIC_BINARY_STEM_L2")
    LR_model = Logistic_Regression_L2_SGD(n_iter=15,eta=0.1,l2=5, batch_size=10000)
    LR_model.fit(np.load('Stored/Vectors/train_tfidf_stem.npy'), np.load('Stored/DocsVocab/y_train.npy'))
    save_obj('Stored/Models/LR-tfidf-stem-L2',LR_model)
    predictions = LR_model.predict(np.load('Stored/Vectors/test_tfidf_stem.npy'))
    evaluate(predictions, y_test, "LOGISTIC_TFIDF_STEM_L2")

def get_trainandtest_vocabanddocs():
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

    Computes vocabulary for train and test- stemmed and unstemmed.
    Tokenizes train and test documents - stemmed and unstemmed.
    sets y_train and y_test.

    Emoticon face tokenization and unicode is utilized as shown above in the emoticon string. Numbers are removed.
    HTML tags are removed. Punctuation and stop words are kept as per the instructions. A Porter stemmer is used where needed.
    Words which begin with a capitalized letter are lowercased. 

    '''

    def tokenize(txt,stem=False):
        """
        Tokenizer that tokenizes text. Can also stem words.
        """
        # from nltk.corpus import stopwords #this is not used as per the requirements, although it improves performance
        # stopwords = set(stopwords.words('english')) 
        # txt = txt.translate(str.maketrans('', '', string.punctuation)) #removes punctuation - not used as per requirements
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
        tokens= tokensfinal
        # tokens = [w for w in tokensfinal if w not in stopwords] #not used as per requirements
        if stem:
            stemmer = PorterStemmer()
            stemmed = [stemmer.stem(item) for item in tokens]
            tokens = stemmed
        return tokens

    #initalize train variables
    trainingdocs = []
    trainingdocs_stemmed = []
    vocabulary = []
    vocabulary_stemmed = []
    y_train = []

    #create megadocument of all training tweets stemmed and not stemmed
    for folder in os.listdir('Data/train'):
        for f in os.listdir(os.path.join('Data/train',folder)):
            tweet = open(os.path.join('Data/train',folder,f),encoding="utf8").read()
            y_train.append(folder)
            trainingdocs.append(tokenize(tweet,False)) #don't stem
            trainingdocs_stemmed.append(tokenize(tweet,True))# stem

    #raw python to get unique vocabulary words from the mega training documents
    vocabulary = list(set(list(itertools.chain.from_iterable(trainingdocs))))
    vocabulary_stemmed = list(set(list(itertools.chain.from_iterable(trainingdocs_stemmed))))
    y_train = np.array(list(map(lambda x: 1 if x == 'pos' else 0, y_train))) #add labels 
    #save results
    np.save('Stored/DocsVocab/trainingdocs',trainingdocs)
    np.save('Stored/DocsVocab/trainingdocs_stemmed',trainingdocs_stemmed)
    np.save('Stored/DocsVocab/vocabulary',vocabulary)
    np.save('Stored/DocsVocab/vocabulary_stemmed',vocabulary_stemmed)
    np.save('Stored/DocsVocab/y_train',y_train)
    print("Train Docs Prepared --- %s seconds ---" % (round((time.time() - start_time)/60,2)))

    #initalize test variables
    testdocs = []
    testdocs_stemmed = []
    y_test = []
    #create megadocument of all testing tweets stemmed and not stemmed
    for folder in os.listdir('Data/test'):
        for f in os.listdir(os.path.join('Data/test',folder)):
            tweet = open(os.path.join('Data/test',folder,f),encoding="utf8").read()
            y_test.append(folder)
            testdocs.append(tokenize(tweet,False)) #don't stem
            testdocs_stemmed.append(tokenize(tweet,True))# stem
    y_test = np.array(list(map(lambda x: 1 if x == 'pos' else 0, y_test))) #add labels
    #save results
    np.save('Stored/DocsVocab/testdocs',testdocs)
    np.save('Stored/DocsVocab/testdocs_stemmed',testdocs_stemmed)
    np.save('Stored/DocsVocab/y_test',y_test)
    print("Test Docs Prepared --- %s minutes ---" % (round((time.time() - start_time)/60,2)))
    return # print('vocabulary, vocabulary_stemmed, trainingdocs, trainingdocs_stemmed, y_train, testdocs, testdocs_stemmed, y_test')

def get_vectors(): 
    '''
    Extract Features: Convert documents to vectors using Bag of Words (BoW) representation. Do
    this in two ways: keeping frequency count where each word is represented by its count in each
    document, keeping binary representation that only keeps track of presence (or not) of a word in
    a document.

    This function creates a BOW vector for all tokenized training and testing documents - both stemmed and unstemmed
    This function also creates a TFIDF vector for all tokenized training and testing documents
    '''
    trainingdocs = np.load('Stored/DocsVocab/trainingdocs.npy')
    trainingdocs_stemmed = np.load('Stored/DocsVocab/trainingdocs_stemmed.npy')
    vocabulary = np.load('Stored/DocsVocab/vocabulary.npy')
    vocabulary_stemmed = np.load('Stored/DocsVocab/vocabulary_stemmed.npy')
    testdocs = np.load('Stored/DocsVocab/testdocs.npy')
    testdocs_stemmed = np.load('Stored/DocsVocab/testdocs_stemmed.npy')

    ##### Bag of Words Frequency Count #####
    #initialize dimensions of final vectors
    ncol = len(vocabulary)
    nrow = len(trainingdocs)
    ncol_stem = len(vocabulary_stemmed)
    nrow_stem = len(trainingdocs_stemmed)

    # creating a dictionary where the key is the distinct vocab word and the
    # value is the index that will be used in the matrix - for speed
    vocab_dict = defaultdict(int)
    for k, v in enumerate(vocabulary):
        vocab_dict[v] = k
        
    stem_vocab_dict = defaultdict(int)
    for k, v in enumerate(vocabulary_stemmed):
        stem_vocab_dict[v] = k
    #save dictionaries for word position information
    save_obj('Stored/DocsVocab/vocab_dict',vocab_dict)
    save_obj('Stored/DocsVocab/stem_vocab_dict',stem_vocab_dict)

    # mapping the word counts to the matrix
    #train freq       
    trainbow_freq = np.zeros((nrow,ncol), dtype=np.int8)
    for n, doc in enumerate(trainingdocs): 
        for word in doc:
            if word in vocab_dict:
                trainbow_freq[n, vocab_dict[word]] += 1 #add 1 if the word appears in tweet
    np.save('Stored/Vectors/trainbow_freq',trainbow_freq) #save
    del trainbow_freq #free memory

    #train freq stemmed
    trainbow_stem_freq = np.zeros((nrow_stem,ncol_stem), dtype=np.int8)
    for n, doc in enumerate(trainingdocs_stemmed): 
        for word in doc:
            if word in stem_vocab_dict:
                trainbow_stem_freq[n, stem_vocab_dict[word]] += 1 #add 1 if the word appears in tweet
    np.save('Stored/Vectors/trainbow_stem_freq',trainbow_stem_freq)
    del trainbow_stem_freq #free memory

    #test freq
    testbow_freq = np.zeros((nrow,ncol), dtype=np.int8)
    for n, doc in enumerate(testdocs): 
        for word in doc:
            if word in vocab_dict:
                testbow_freq[n, vocab_dict[word]] += 1 #add 1 if the word appears in tweet
    np.save('Stored/Vectors/testbow_freq',testbow_freq)
    del testbow_freq #free memory

    #test freq stemmed
    testbow_stem_freq = np.zeros((nrow_stem,ncol_stem), dtype=np.int8)
    for n, doc in enumerate(testdocs_stemmed): 
        for word in doc:
            if word in stem_vocab_dict:
                testbow_stem_freq[n, stem_vocab_dict[word]] += 1 #add 1 if the word appears in tweet
    np.save('Stored/Vectors/testbow_stem_freq',testbow_stem_freq)
    del testbow_stem_freq #free memory

    ##### Bag of Words Binary Count #####
    # mapping the word counts to the matrix
    #train binary 
    trainbow_binary = np.zeros((nrow,ncol), dtype=np.int8)
    for n, doc in enumerate(trainingdocs): 
        for word in doc:
            if word in vocab_dict:
                trainbow_binary[n, vocab_dict[word]] = 1 #set to 1 if the word appears in tweet
    np.save('Stored/Vectors/trainbow_binary',trainbow_binary)
    del trainbow_binary 

    #train binary stemmed
    trainbow_stem_binary = np.zeros((nrow_stem,ncol_stem), dtype=np.int8)                       
    for n, doc in enumerate(trainingdocs_stemmed): 
        for word in doc:
            if word in stem_vocab_dict:
                trainbow_stem_binary[n, stem_vocab_dict[word]] = 1 #set to 1 if the word appears in tweet
    np.save('Stored/Vectors/trainbow_stem_binary',trainbow_stem_binary)
    del trainbow_stem_binary 

    #test binary 
    testbow_binary = np.zeros((nrow,ncol), dtype=np.int8)
    for n, doc in enumerate(testdocs): 
        for word in doc:
            if word in vocab_dict:
                testbow_binary[n, vocab_dict[word]] = 1 #set to 1 if the word appears in tweet
    np.save('Stored/Vectors/testbow_binary',testbow_binary)
    del testbow_binary  

    #test binary stemmed
    testbow_stem_binary = np.zeros((nrow_stem,ncol_stem), dtype=np.int8)                        
    for n, doc in enumerate(testdocs_stemmed): 
        for word in doc:
            if word in stem_vocab_dict:
                testbow_stem_binary[n, stem_vocab_dict[word]] = 1 #set to 1 if the word appears in tweet
    np.save('Stored/Vectors/testbow_stem_binary',testbow_stem_binary)
    del testbow_stem_binary 

    ## TFIDF ##
    # get train idf
    trainbow_freq = np.load('Stored/Vectors/trainbow_freq.npy') #use frequency count for TF
    vector = np.int32(np.count_nonzero(trainbow_freq, axis = 0)) #count number of documents for each word
    idf = np.int16(np.log(len(trainbow_freq)//vector)) # idf = log(total documents / numer of documents for each word)
    del vector #free memory
    train_tfidf = np.int8(np.multiply(trainbow_freq, idf)) #multiply Tf * idf for tfidf
    del idf
    np.save('Stored/Vectors/train_tfidf',train_tfidf) #save
    del train_tfidf
    trainbow_stem_freq = np.load('Stored/Vectors/trainbow_stem_freq.npy') #repeat for stem
    vector = np.int32(np.count_nonzero(trainbow_stem_freq, axis = 0))
    idf_stem = np.int16(np.log(np.true_divide(len(trainbow_stem_freq),vector)))
    del vector
    train_tfidf_stem = np.int8(np.multiply(trainbow_stem_freq,idf_stem))
    del idf_stem
    np.save('Stored/Vectors/train_tfidf_stem',train_tfidf_stem)
    del train_tfidf_stem
    del trainbow_freq
    del trainbow_stem_freq

    # get test idf
    testbow_freq = np.load('Stored/Vectors/testbow_freq.npy')
    vector = np.int32(np.count_nonzero(testbow_freq, axis = 0))
    idf = np.int16(np.log(len(testbow_freq)//vector))
    del vector
    test_tfidf = np.int8(np.multiply(testbow_freq, idf))
    del idf
    np.save('Stored/Vectors/test_tfidf',test_tfidf)
    del test_tfidf
    testbow_stem_freq = np.load('Stored/Vectors/testbow_stem_freq.npy') #repeat for stem
    vector = np.int32(np.count_nonzero(testbow_stem_freq, axis = 0))
    idf_stem = np.int16(np.log(np.true_divide(len(testbow_stem_freq),vector)))
    del vector
    test_tfidf_stem = np.int8(np.multiply(testbow_stem_freq,idf_stem))
    del idf_stem
    np.save('Stored/Vectors/test_tfidf_stem',test_tfidf_stem)
    del test_tfidf_stem
    del testbow_freq
    del testbow_stem_freq

    #output vectors
    # train_vecs = [trainbow_freq, trainbow_stem_freq, trainbow_binary, trainbow_stem_binary, train_tfidf, train_tfidf_stem]
    # test_vecs = [testbow_freq, testbow_stem_freq, testbow_binary, testbow_stem_binary, test_tfidf, test_tfidf_stem]

    print("Vectors Created --- %s minutes ---" % (round((time.time() - start_time)/60,2)))    
    return # print('train_vecs, vocab_dict, stem_vocab_dict, test_vecs') 

def get_class_priors():
    '''
    calculate the prior for each class = number of samples of class C in training set / total number of samples in training set (25000)
    Pˆ(c) = Nc/N
    '''
    y_train = np.load('Stored/DocsVocab/y_train.npy')
    P_negative = list(y_train).count(0)/y_train.size
    P_positive = list(y_train).count(1)/y_train.size

    print("Priors Assessed --- %s minutes ---" % (round((time.time() - start_time)/60,2)))
    return P_positive,P_negative

def get_perword_likelihood():
    '''
    Calculates P^(w|c) for each word for each class for each set of vectors
    The calculation depends on what type of vector is being used.
    FREQUENCY:
    Pˆ(w | c) = count(w, c)+1 /(count(c)+ |V|) 

    BINARY (Follows Multi-variate Bernoulli Naive Bayes) https://sebastianraschka.com/Articles/2014_naive_bayes_1.html
    P^(wi∣cj)=dfwi,c+1 / dfc+2 -- dfxi is documents word is in for class, dfy is total number of documents for class

    TFIDF
    P^(w|c)=(Tf*idf(wi,cj)) + 1 / (Tf*idf(W,cj)+|V|)
    '''
    #load needed data
    trainbow_freq = np.load('Stored/Vectors/trainbow_freq.npy')
    y_train = np.load('Stored/DocsVocab/y_train.npy')
    vocab_dict = load_obj('Stored/DocsVocab/vocab_dict')
    stem_vocab_dict = load_obj('Stored/DocsVocab/stem_vocab_dict')

    ## frequency Pˆ(w | c) = count(w, c)+1 /(count(c)+ |V|) 
    #initialize output vars
    frequencydict_pos = defaultdict()
    frequencydict_neg = defaultdict()
    frequencydict_pos_stem = defaultdict()
    frequencydict_neg_stem = defaultdict()
    num_pos = list(y_train).count(1) #count number of positive trainnig docs for indexing

    ## frequency not stemmed ##  
    denom_pos =trainbow_freq[(1+num_pos):,:].sum()+len(vocab_dict.keys()) #sum of all positive words and vocab size
    denom_neg = trainbow_freq[:num_pos:,:].sum()+len(vocab_dict.keys()) #sum of all negative words and vocab size
    trainbow_freq_pos_sum = trainbow_freq[(1+num_pos):,:].sum(axis = 0) #per word summing for positive class
    trainbow_freq_neg_sum = trainbow_freq[:num_pos,:].sum(axis = 0) # per word summing for negative class
    
    for v in vocab_dict.keys():

        frequencydict_pos[v] = (trainbow_freq_pos_sum[vocab_dict[v]]+1) /(denom_pos) #calculate probability of each word given class
        frequencydict_neg[v] = (trainbow_freq_neg_sum[vocab_dict[v]]+1) /(denom_neg) #uses +1 smoothing

    save_obj('Stored/Likelihoods/frequencydict_pos',frequencydict_pos) #save
    save_obj('Stored/Likelihoods/frequencydict_neg',frequencydict_neg)
    del frequencydict_pos #free memory
    del frequencydict_neg
    del trainbow_freq

    ## frequency stemmed ## 
    trainbow_stem_freq = np.load('Stored/Vectors/trainbow_stem_freq.npy') 
    denom_stem_pos = trainbow_stem_freq[(1+num_pos):,:].sum()+len(stem_vocab_dict.keys()) #sum of all positive words and vocab size
    denom_stem_neg = trainbow_stem_freq[:num_pos,:].sum()+len(stem_vocab_dict.keys()) #sum of all negative words and vocab size
    trainbow_freq_stem_pos_sum = trainbow_stem_freq[(1+num_pos):,:].sum(axis = 0) #per word summing for positive class
    trainbow_freq_stem_neg_sum = trainbow_stem_freq[:num_pos,:].sum(axis = 0) # per word summing for negative class

    for v in stem_vocab_dict.keys():

        frequencydict_pos_stem[v] = (trainbow_freq_stem_pos_sum[stem_vocab_dict[v]]+1) /(denom_stem_pos) #calculate probability of each word given class
        frequencydict_neg_stem[v] = (trainbow_freq_stem_neg_sum[stem_vocab_dict[v]]+1) /(denom_stem_neg) #uses +1 smoothing
    
    save_obj('Stored/Likelihoods/frequencydict_pos_stem',frequencydict_pos_stem)
    save_obj('Stored/Likelihoods/frequencydict_neg_stem',frequencydict_neg_stem)
    del frequencydict_pos_stem
    del frequencydict_neg_stem
    del trainbow_stem_freq


    # binary ##  Multi-variate Bernoulli Naive Bayes https://sebastianraschka.com/Articles/2014_naive_bayes_1.html
    #initialize variables
    trainbow_binary = np.load('Stored/Vectors/trainbow_binary.npy')
    binarydict_pos = defaultdict()
    binarydict_neg = defaultdict()
    binarydict_pos_stem = defaultdict()
    binarydict_neg_stem = defaultdict()

    ## binary not stemmed ## 
    trainbow_binary_pos_sum = trainbow_binary[(1+num_pos):,:].sum(axis = 0) #per word summing for positive class
    trainbow_binary_neg_sum = trainbow_binary[:num_pos,:].sum(axis = 0) # per word summing for negative class
    pos_docs = num_pos #number of positive documents
    neg_docs = list(y_train).count(0) #number of negative documents
    
    for v in vocab_dict.keys():
            
        binarydict_pos[v] = (trainbow_binary_pos_sum[vocab_dict[v]]+1) /((pos_docs) + 2) #calculate per word probabilities based on
        binarydict_neg[v] = (trainbow_binary_neg_sum[vocab_dict[v]]+1) /((neg_docs) + 2) #multivariate Bernoulli Naive Bayes with smoothing

    save_obj('Stored/Likelihoods/binarydict_pos', binarydict_pos)
    save_obj('Stored/Likelihoods/binarydict_neg', binarydict_neg)
    del binarydict_pos
    del binarydict_neg
    del trainbow_binary

    ## binary stemmed ##
    trainbow_stem_binary = np.load('Stored/Vectors/trainbow_stem_binary.npy')
    trainbow_binary_stem_pos_sum = trainbow_stem_binary[(1+num_pos):,:].sum(axis = 0) #per word summing for positive class
    trainbow_binary_stem_neg_sum = trainbow_stem_binary[:num_pos,:].sum(axis = 0) # per word summing for negative class

    for v in stem_vocab_dict.keys():
            
        binarydict_pos_stem[v] = (trainbow_binary_stem_pos_sum[stem_vocab_dict[v]]+1) /((pos_docs) + 2) #calculate per word probabilities based on
        binarydict_neg_stem[v] = (trainbow_binary_stem_neg_sum[stem_vocab_dict[v]]+1) /((neg_docs) + 2) #multivariate Bernoulli Naive Bayes with smoothing

    save_obj('Stored/Likelihoods/binarydict_pos_stem', binarydict_pos_stem)
    save_obj('Stored/Likelihoods/binarydict_neg_stem', binarydict_neg_stem)
    del binarydict_pos_stem
    del binarydict_neg_stem
    del trainbow_stem_binary


    ## tfidf ##
    #load and initalize
    train_tfidf = np.load('Stored/Vectors/train_tfidf.npy')
    tfidf_pos = defaultdict()
    tfidf_neg = defaultdict()
    tfidf_pos_stem = defaultdict()
    tfidf_neg_stem = defaultdict()
    #not stemmed
    tfidf_pos_sum_word = train_tfidf[(1+num_pos):,:].sum(axis = 0) #per word summing for positive class
    tfidf_neg_sum_word = train_tfidf[:num_pos,:].sum(axis = 0) # per word summing for negative class
    tfidf_pos_denom = train_tfidf[(1+num_pos):,:].sum()+ len(vocab_dict.keys()) #summing all tfidf for positive class
    tfidf_neg_denom = train_tfidf[:num_pos,:].sum()+ len(vocab_dict.keys()) #summing all tfidf for negative class

    for v in vocab_dict.keys():
        tfidf_pos[v] = (tfidf_pos_sum_word[vocab_dict[v]]+1) /(tfidf_pos_denom) # per word probabilities given class
        tfidf_neg[v] = (tfidf_neg_sum_word[vocab_dict[v]]+1) /(tfidf_neg_denom) # (TFIDF for word in class + 1 )/ (TFIDF all words in class + |V|)
    
    save_obj('Stored/Likelihoods/tfidf_pos', tfidf_pos)
    save_obj('Stored/Likelihoods/tfidf_neg', tfidf_neg)
    del tfidf_pos
    del tfidf_neg
    del train_tfidf

    #stemmed
    train_tfidf_stem = np.load('Stored/Vectors/train_tfidf_stem.npy')
    tfidf_stem_pos_sum_word = train_tfidf_stem[(1+num_pos):,:].sum(axis = 0) #per word summing for positive class
    tfidf_stem_neg_sum_word = train_tfidf_stem[:num_pos,:].sum(axis = 0) # per word summing for negative class
    tfidf_stem_pos_denom = train_tfidf_stem[(1+num_pos):,:].sum()+ len(stem_vocab_dict.keys()) #summing all tfidf for positive class
    tfidf_stem_neg_denom = train_tfidf_stem[:num_pos,:].sum()+ len(stem_vocab_dict.keys()) #summing all tfidf for negative class

    for v in stem_vocab_dict.keys():
        tfidf_pos_stem[v] = (tfidf_stem_pos_sum_word[stem_vocab_dict[v]]+1) /(tfidf_stem_pos_denom)
        tfidf_neg_stem[v] = (tfidf_stem_neg_sum_word[stem_vocab_dict[v]]+1) /(tfidf_stem_neg_denom)

    save_obj('Stored/Likelihoods/tfidf_pos_stem', tfidf_pos_stem)
    save_obj('Stored/Likelihoods/tfidf_neg_stem', tfidf_neg_stem)
    del tfidf_pos_stem
    del tfidf_neg_stem
    del train_tfidf_stem


                            
    print("Word Likelihoods Calculated --- %s minutes ---" % (round((time.time() - start_time)/60,2)))
    return print('wordlikelihood_dict')

def predict_NB(P_positive, P_negative):
    '''
    Using Naive Bayes:
    Compute the most likely class for each document in the test set using each of the
    combinations of stemming + frequency count, stemming + binary, no-stemming + frequency
    count, no-stemming + binary.
    This is also done for TFIDF vectors - stemming and no-stemming
    '''
    # log scale: http://www.cs.rhodes.edu/~kirlinp/courses/ai/f18/projects/proj3/naive-bayes-log-probs.pdf
    #load variables
    frequencydict_pos = load_obj('Stored/Likelihoods/frequencydict_pos')
    frequencydict_neg = load_obj('Stored/Likelihoods/frequencydict_neg')
    frequencydict_pos_stem = load_obj('Stored/Likelihoods/frequencydict_pos_stem')
    frequencydict_neg_stem = load_obj('Stored/Likelihoods/frequencydict_neg_stem')
    binarydict_pos = load_obj('Stored/Likelihoods/binarydict_pos')
    binarydict_neg = load_obj('Stored/Likelihoods/binarydict_neg')
    binarydict_pos_stem = load_obj('Stored/Likelihoods/binarydict_pos_stem')
    binarydict_neg_stem = load_obj('Stored/Likelihoods/binarydict_neg_stem')
    tfidf_pos = load_obj('Stored/Likelihoods/tfidf_pos')
    tfidf_neg = load_obj('Stored/Likelihoods/tfidf_neg')
    tfidf_pos_stem = load_obj('Stored/Likelihoods/tfidf_pos_stem')
    tfidf_neg_stem = load_obj('Stored/Likelihoods/tfidf_neg_stem')
    testdocs = np.load('Stored/DocsVocab/testdocs.npy')
    testdocs_stemmed = np.load('Stored/DocsVocab/testdocs_stemmed.npy')
    y_test = np.load('Stored/DocsVocab/y_test.npy')
    #put into workable form
    wordlikelihood_dict = {"frequencydict_pos":frequencydict_pos, "frequencydict_neg":frequencydict_neg, \
                                "frequencydict_pos_stem":frequencydict_pos_stem, "frequencydict_neg_stem":frequencydict_neg_stem, \
                                "binarydict_pos":binarydict_pos, "binarydict_neg":binarydict_neg, \
                                "binarydict_pos_stem":binarydict_pos_stem, "binarydict_neg_stem":binarydict_neg_stem, \
                                "tfidf_pos":tfidf_pos,"tfidf_neg":tfidf_neg, \
                                "tfidf_pos_stem":tfidf_pos_stem, "tfidf_neg_stem":tfidf_neg_stem}
    # No-Stemming -- predict P(w |C) for each word in test documents take log and add to log of class priors - highest is chosen class
    y_pred_freq = np.zeros(y_test.size)
    y_pred_binary = np.zeros(y_test.size)
    y_pred_tfidf = np.zeros(y_test.size)
    for i, documt in enumerate(testdocs):
        prob_pos_case1 = np.log(P_positive)
        prob_pos_case2 = np.log(P_positive)
        prob_pos_case3 = np.log(P_positive)
        prob_neg_case1 = np.log(P_negative)
        prob_neg_case2 = np.log(P_negative)
        prob_neg_case3= np.log(P_negative)
        for word in documt:
            if word in wordlikelihood_dict["frequencydict_pos"].keys():
                #pos
                prob_pos_case1 += np.log(wordlikelihood_dict["frequencydict_pos"][word])
                prob_pos_case2 += np.log(wordlikelihood_dict["binarydict_pos"][word])
                prob_pos_case3 += np.log(wordlikelihood_dict["tfidf_pos"][word])
                #neg
                prob_neg_case1 += np.log(wordlikelihood_dict["frequencydict_neg"][word])
                prob_neg_case2 += np.log(wordlikelihood_dict["binarydict_neg"][word])
                prob_neg_case3 += np.log(wordlikelihood_dict["tfidf_neg"][word])
            
        # No-Stemming + Frequency count - assign class to test data
        if prob_pos_case1 > prob_neg_case1:
            y_pred_freq[i] = 1
        else:
            y_pred_freq[i] = 0
            
        # No-Stemming + Binary
        if prob_pos_case2 > prob_neg_case2:
            y_pred_binary[i] = 1
        else:
            y_pred_binary[i] = 0

        # No-Stemming + TFIDF
        if prob_pos_case3 > prob_neg_case3:
            y_pred_tfidf[i] = 1
        else:
            y_pred_tfidf[i] = 0

    # Stemming
    y_pred_freq_stem = np.zeros(y_test.size)
    y_pred_binary_stem = np.zeros(y_test.size)
    y_pred_tfidf_stem = np.zeros(y_test.size)
    for i, documt in enumerate(testdocs_stemmed):
        prob_pos_case3 = np.log(P_positive)
        prob_pos_case4 = np.log(P_positive)
        prob_pos_case5 = np.log(P_positive)
        prob_neg_case3 = np.log(P_negative)
        prob_neg_case4 = np.log(P_negative)
        prob_neg_case5 = np.log(P_negative)
        for word in documt:
            if word in wordlikelihood_dict["frequencydict_pos_stem"].keys():
                #pos
                prob_pos_case3 += np.log(wordlikelihood_dict["frequencydict_pos_stem"][word])
                prob_pos_case4 += np.log(wordlikelihood_dict["binarydict_pos_stem"][word])
                prob_pos_case5 += np.log(wordlikelihood_dict["tfidf_pos_stem"][word])
                #neg
                prob_neg_case3 += np.log(wordlikelihood_dict["frequencydict_neg_stem"][word])
                prob_neg_case4 += np.log(wordlikelihood_dict["binarydict_neg_stem"][word])
                prob_neg_case5 += np.log(wordlikelihood_dict["tfidf_neg_stem"][word])
                        
        # Stemming + Frequency count
        if prob_pos_case3 > prob_neg_case3:
            y_pred_freq_stem[i] = 1
        else:
            y_pred_freq_stem[i] = 0
            
        # Stemming + Binary
        if prob_pos_case4 > prob_neg_case4:
            y_pred_binary_stem[i] = 1
        else:
            y_pred_binary_stem[i] = 0

        # Stemming + TFIDF
        if prob_pos_case5 > prob_neg_case5:
            y_pred_tfidf_stem[i] = 1
        else:
            y_pred_tfidf_stem[i] = 0

    #output predicted test data for evaluation
    predictions = [y_pred_freq, y_pred_binary, y_pred_tfidf, y_pred_freq_stem, y_pred_binary_stem,y_pred_tfidf_stem] 
    print("NB Predictions Acquired --- %s seconds ---" % (time.time() - start_time))
    return predictions

def evaluate(predictions,y_test, model):
    '''
        For each of your classifiers, compute and report accuracy. Accuracy is
    number of correctly classified reviews/number of all reviews in test (25k for our case). Also
    create a confusion matrix for each classifier. Save your output in a .txt or .log file.
    '''
    print("Evaulating  %s --- %s minutes ---" % (model,round((time.time() - start_time)/60,2)))
    score = 0
    for i in range(y_test.size):
        if predictions[i] == y_test[i]:
            score +=1
    accuracy = score*100/float(y_test.size)
    print('Accuracy: ' + str(accuracy))

    # To calculate confusion matrix (eg.)
    cm = pd.DataFrame(confusion_matrix(predictions, y_test),
                 columns = ['neg', 'pos'], index = ['neg', 'pos'])
    print (cm)
    return 

class Logistic_Regression_L2_SGD:
    """ Defining a Logistic Regression class with L2 regularization and 
        Stochastic Gradient Descent
    
    The parameters are:
        l2: lambda value for l2 regularization
        n_iter: number of iterations over the dataset
        eta: learning rate
        batch_size: size of each batch (SGD=1 and full batch = len(X))
    """
    
    def __init__(self, l2=0.0, n_iter=1000, eta=0.05, batch_size=1):
        self.l2 = l2
        self.n_iter = n_iter
        self.eta = eta
        self.batch_size = batch_size
            
    def sigmoid(self, z):
        # This is the sigmoid function of z
        return 1/(1+ np.exp(-z))
    
    def fit(self, X, y):
        print("Fitting Logistic Regression, eta = %s, %s iterations, L2 = %s --- %s minutes ---" % (self.eta, self.n_iter,self.l2,round((time.time() - start_time)/60,2)))
        # fit the training data
        
        y = y.reshape(-1,1)
        # initialize the values of the weights to zero
        self.theta = np.zeros((X.shape[1],1))
        m = y.shape[0]
        pad = 1e-6
        self.cost_values = []
        for _ in range(self.n_iter):
            # shuffling each iteration as to prevent overfitting
            shuffled_values = np.random.permutation(m)
            X_shuffled = X[shuffled_values]
            y_shuffled = y[shuffled_values]
            # iterating over each batch
            for batch in range(0, m, self.batch_size):
                x_batch = X_shuffled[batch:batch+self.batch_size]
                y_batch = y_shuffled[batch:batch+self.batch_size]
                z = self.sigmoid(np.dot(x_batch, self.theta))
                # calculating the gradient with the derived formula
                gradient = x_batch.T.dot(z-y_batch)/m + (self.l2/m*self.theta)
                self.theta -= self.eta * gradient
                # implementing the cost (objective) function given
                cost = np.average(-y_batch*np.log(z+pad) - ((1-y_batch)*np.log(1-z+pad)))
                l2_cost = cost + (self.l2/(2*m) * np.linalg.norm(self.theta[1:])**2)  # we don't regularize the intersect
                self.cost_values.append(l2_cost)

        return self
    

    
    def predict(self, X, threshold=0.5):
        # return the predicted values in (0,1) format
        return np.where(self.sigmoid(X.dot(self.theta)) >= threshold,1,0)
    
    def predict_prob(self, X):
        # return the predicted values in percentage format
        return self.sigmoid(X.dot(self.theta))



if __name__ == "__main__":
    main()