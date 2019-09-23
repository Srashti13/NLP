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
import operator
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.metrics import confusion_matrix

def main():
    '''
    main function
    '''
    vocabulary, vocabulary_stemmed, trainingdocs, trainingdocs_stemmed, y_train, testdocs, testdocs_stemmed, y_test = get_trainandtest_vocabanddocs()
    trainbow_freq, trainbow_stem_freq, trainbow_binary, trainbow_stem_binary, vocab_dict, stem_vocab_dict = get_BOW(trainingdocs, trainingdocs_stemmed, vocabulary, vocabulary_stemmed)
    P_positive, P_negative  = get_class_priors(y_train)
    wordlikelihood_dicts_list = get_perword_likelihood(trainbow_freq, trainbow_stem_freq, trainbow_binary, trainbow_stem_binary,vocab_dict, stem_vocab_dict, y_train)
    evaluate_NB(wordlikelihood_dicts_list,P_positive, P_negative, testdocs, testdocs_stemmed, y_test)

    #LR_model = Logistic_Regression_L2_SGD(n_iter=1, batch_size=len(train_BOW_freq))
    #LR_model.fit(train_BOW_binary, y_train)
    #predictions = LR_model.predict(train_BOW_freq)
    #print("Accuracy: {:.0f}%".format(sum(predictions.flatten() == y_train)/len(y_train)*100))

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
    '''

    def tokenize(txt,stem=False):
        """
        Tokenizer that tokenizes text. Can also stem words.
        """
        def lower_repl(match):
            return match.group(1).lower()

        txt = re.sub('<[^<]+?>', '', txt)# remove html tags
        txt = re.sub('([A-Z][a-z]+)',lower_repl,txt) #lowercase words that start with captial 
        tokens = TweetTokenizer().tokenize(txt) #emoticon stemming
# >>>

        for word in tokens:
            new_words_toadd = []
            if any(p in word for p in punctuations) and len(word) > 1:
                word1 = re.sub("\s?-\s?", r' - ', word)                                 # only to remove '-'
                new_words = word_tokenize(re.sub("(?<=[.,|-])(?=[^\s])", r' ', word1))  # to remove other punctuations
                new_words_toadd.extend(new_words)
            if len(new_words_toadd) >= 1:
                tokens.remove(word)
                tokens.extend(new_words_toadd)
# ------
        
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

    return vocabulary, vocabulary_stemmed, trainingdocs, trainingdocs_stemmed, y_train, testdocs, testdocs_stemmed, y_test

def get_BOW(trainingdocs, trainingdocs_stemmed, vocabulary, vocabulary_stemmed): 
    '''
    Extract Features: Convert documents to vectors using Bag of Words (BoW) representation. Do
    this in two ways: keeping frequency count where each word is represented by its count in each
    document, keeping binary representation that only keeps track of presence (or not) of a word in
    a document.
    '''
    ##### Bag of Words Frequency Count #####
    ncol = len(vocabulary)
    nrow = len(trainingdocs)
    trainbow_freq = np.zeros((nrow,ncol), dtype=np.int8)

    ncol_stem = len(vocabulary_stemmed)
    nrow_stem = len(trainingdocs_stemmed)
    trainbow_stem_freq = np.zeros((nrow_stem,ncol_stem), dtype=np.int8)
    
    # creating a dictionary where the key is the distinct vocab word and the
    # value is the index that will be used in the matrix
    vocab_dict = defaultdict(int)
    for k, v in enumerate(vocabulary):
        vocab_dict[v] = k
        
    stem_vocab_dict = defaultdict(int)
    for k, v in enumerate(vocabulary_stemmed):
        stem_vocab_dict[v] = k
        
    # mapping the word counts to the matrix
    for n, doc in enumerate(trainingdocs):
        for word in doc:
            if word in vocab_dict:
                trainbow_freq[n, vocab_dict[word]] += 1
    
    for n, doc in enumerate(trainingdocs_stemmed):
        for word in doc:
            if word in stem_vocab_dict:
                trainbow_stem_freq[n, stem_vocab_dict[word]] += 1
    

    ##### Bag of Words Binary Count #####
    trainbow_binary = np.zeros((nrow,ncol), dtype=np.int8)
    trainbow_stem_binary = np.zeros((nrow_stem,ncol_stem), dtype=np.int8)
    
    # mapping the word counts to the matrix
    for n, doc in enumerate(trainingdocs):
        for word in doc:
            if word in vocab_dict:
                trainbow_binary[n, vocab_dict[word]] = 1
                                
    for n, doc in enumerate(trainingdocs_stemmed):
        for word in doc:
            if word in stem_vocab_dict:
                trainbow_stem_binary[n, stem_vocab_dict[word]] = 1

    return trainbow_freq, trainbow_stem_freq, trainbow_binary, trainbow_stem_binary, vocab_dict, stem_vocab_dict

def get_class_priors(y_train):
    '''
    calculate the prior for each class = number of samples of class C in training set / total number of samples in training set (25000)
    Pˆ(c) = Nc/N
    '''
    P_negative = list(y_train).count(0)/y_train.size
    P_positive = list(y_train).count(1)/y_train.size
    return P_negative, P_positive


def get_perword_likelihood( trainbow_freq, trainbow_stem_freq, trainbow_binary, trainbow_stem_binary,vocab_dict, stem_vocab_dict, y_train):
    '''
    Pˆ(w | c) = count(w, c)+1 /(count(c)+ |V|)  

    depends on vocabulary being stemmed/non-stemmed and the type of vectors being used
    '''
    ## frequency Pˆ(w | c) = count(w, c)+1 /(count(c)+ |V|) 
    frequencydict_pos = defaultdict(int)            # by adding "int" if the word does not exist in the dictionary
                                                    # it automatically assigned a value of 0
    frequencydict_neg = defaultdict(int)
    frequencydict_pos_stem = defaultdict(int)
    frequencydict_neg_stem = defaultdict(int)
    num_pos = list(y_train).count(0)                # The code should be 'list(y_train).count(1)', as pos == 1 in 
                                                    # line ~84, even thou it does not make a difference due to 
                                                    # equal pos and neg tweets/docs

    ## frequency not stemmed ##  
    denom_pos = trainbow_freq[:num_pos,:].sum()+len(vocab_dict.keys()) #sum of all positive words and vocab size
    denom_neg = trainbow_freq[(1+num_pos):,:].sum()+len(vocab_dict.keys()) #sum of all negative words and vocab size
    trainbow_freq_pos_sum = trainbow_freq[:num_pos,:].sum(axis = 0) #per word summing for positive class
    trainbow_freq_neg_sum = trainbow_freq[(1+num_pos):,:].sum(axis = 0) # per word summing for negative class

    for v in vocab_dict.keys():
        
        frequencydict_pos[v] = trainbow_freq_pos_sum[vocab_dict[v]]+1 /\
                                        (denom_pos)

        frequencydict_neg[v] = trainbow_freq_neg_sum[vocab_dict[v]]+1 /\
                                        (denom_neg)

    ## frequency stemmed ##  
    denom_stem_pos = trainbow_stem_freq[:num_pos,:].sum()+len(stem_vocab_dict.keys()) #sum of all positive words and vocab size
    denom_stem_neg = trainbow_stem_freq[(1+num_pos):,:].sum()+len(stem_vocab_dict.keys()) #sum of all negative words and vocab size
    trainbow_freq_stem_pos_sum = trainbow_stem_freq[:num_pos,:].sum(axis = 0) #per word summing for positive class
    trainbow_freq_stem_neg_sum = trainbow_stem_freq[(1+num_pos):,:].sum(axis = 0) # per word summing for negative class

    for v in vocab_dict.keys():

        frequencydict_pos_stem[v] = trainbow_freq_stem_pos_sum[stem_vocab_dict[v]]+1 /\
                                        (denom_stem_pos)

        frequencydict_neg_stem[v] = trainbow_freq_stem_neg_sum[stem_vocab_dict[v]]+1 /\
                                        (denom_stem_neg)


    # binary ##   P^(xi∣ωj)=dfxi,y+1 / dfy+2 Multi-variate Bernoulli Naive Bayes https://sebastianraschka.com/Articles/2014_naive_bayes_1.html
    binarydict_pos = defaultdict(int)
    binarydict_neg = defaultdict(int)
    binarydict_pos_stem = defaultdict(int)
    binarydict_neg_stem = defaultdict(int)

    ## binary not stemmed ## 
    trainbow_binary_pos_sum = trainbow_binary[:num_pos,:].sum(axis = 0) #per word summing for positive class
    trainbow_binary_neg_sum = trainbow_binary[(1+num_pos):,:].sum(axis = 0) # per word summing for negative class
    pos_docs = num_pos #number of positive documents
    neg_docs = list(y_train).count(1) #number of negative documents

    for v in vocab_dict.keys():
            
        binarydict_pos[v] = trainbow_binary_pos_sum[stem_vocab_dict[v]]+1 /\
                                    (pos_docs) + 2

        binarydict_neg[v] = trainbow_binary_neg_sum[stem_vocab_dict[v]]+1 /\
                        (neg_docs) + 2

    ## binary stemmed ##
    trainbow_binary_stem_pos_sum = trainbow_stem_binary[:num_pos,:].sum(axis = 0) #per word summing for positive class
    trainbow_binary_stem_neg_sum = trainbow_stem_binary[(1+num_pos):,:].sum(axis = 0) # per word summing for negative class

    for v in vocab_dict.keys():
            
        binarydict_pos_stem[v] = trainbow_binary_stem_pos_sum[stem_vocab_dict[v]]+1 /\
                                    (pos_docs) + 2

        binarydict_neg_stem[v] = trainbow_binary_stem_neg_sum[stem_vocab_dict[v]]+1 /\
                        (neg_docs) + 2


    wordlikelihood_dicts_list = [frequencydict_pos, frequencydict_neg, \
                                frequencydict_pos_stem, frequencydict_neg_stem, \
                                binarydict_pos, binarydict_neg, \
                                binarydict_pos_stem, binarydict_neg_stem]
    return wordlikelihood_dicts_list

def evaluate_NB(wordlikelihood_dicts_list,P_positive, P_negative, testdocs, testdocs_stemmed, y_test):
    '''
    Compute the most likely class for each document in the test set using each of the
combinations of stemming + frequency count, stemming + binary, no-stemming + frequency
count, no-stemming + binary.
     For each of your classifiers, compute and report accuracy. Accuracy is
    number of correctly classified reviews/number of all reviews in test (25k for our case). Also
    create a confusion matrix for each classifier. Save your output in a .txt or .log file.
    '''
    return print('')

#>>>>
def evaluate_NB(wordlikelihood_dicts_list,P_positive, P_negative, testdocs, testdocs_stemmed, y_test):
    '''
    Compute the most likely class for each document in the test set using each of the
combinations of stemming + frequency count, stemming + binary, no-stemming + frequency
count, no-stemming + binary.
     For each of your classifiers, compute and report accuracy. Accuracy is
    number of correctly classified reviews/number of all reviews in test (25k for our case). Also
    create a confusion matrix for each classifier. Save your output in a .txt or .log file.
    '''
    #(p(w1/c)*p(c)) + (p(w2/c)*p(c)) + (p(w3/c)*p(c))
    
    # No-Stemming
    NoStem_FreqCount_doc_class = []
    NoStem_Binary_doc_class = []
    for documt in testdocs:
        prob_pos_case1 = 0
        prob_pos_case2 = 0
        prob_neg_case1 = 0
        prob_neg_case2 = 0
        for word in documt:
        #pos
            prob_pos_case1 += wordlikelihood_dicts_list[0][word]*P_positive
            prob_pos_case2 += wordlikelihood_dicts_list[4][word]*P_positive
        #neg
            prob_neg_case1 += wordlikelihood_dicts_list[1][word]*P_negative
            prob_neg_case2 += wordlikelihood_dicts_list[5][word]*P_negative
    
    # No-Stemming + Frequency count
        if prob_pos_case1 > prob_neg_case1:
            NoStem_FreqCount_doc_class.append('pos')
        else:
            NoStem_FreqCount_doc_class.append('neg')
            
    # No-Stemming + Binary
        if prob_pos_case2 > prob_neg_case2:
            NoStem_Binary_doc_class.append('pos')
        else:
            NoStem_Binary_doc_class.append('neg')
    
    NoStem_FreqCount_accuray = sum(1 for x,y in zip(NoStem_FreqCount_doc_class,y_test) if x == y) / len(NoStem_FreqCount_doc_class)
    NoStem_Binary_accuracy = sum(1 for x,y in zip(NoStem_Binary_doc_class,y_test) if x == y) / len(NoStem_Binary_doc_class)
    
    # Stemming
    Stem_FreqCount_doc_class = []
    Stem_Binary_doc_class = []
    for documt in testdocs_stemmed:
        prob_pos_case3 = 0
        prob_pos_case4 = 0
        prob_neg_case3 = 0
        prob_neg_case4 = 0
        for word in documt:
        #pos
            prob_pos_case3 += wordlikelihood_dicts_list[2][word]*P_positive
            prob_pos_case4 += wordlikelihood_dicts_list[6][word]*P_positive
        #neg
            prob_neg_case3 += wordlikelihood_dicts_list[3][word]*P_negative
            prob_neg_case4 += wordlikelihood_dicts_list[7][word]*P_negative
                
    # Stemming + Frequency count
        if prob_pos_case3 > prob_neg_case3:
            Stem_FreqCount_doc_class.append("pos")
        else:
            Stem_FreqCount_doc_class.append("neg")
            
    # Stemming + Binary
        if prob_pos_case4 > prob_neg_case4:
            Stem_Binary_doc_class.append("pos")
        else:
            Stem_Binary_doc_class.append("neg")
    
    Stem_FreqCount_accuray = sum(1 for x,y in zip(Stem_FreqCount_doc_class,y_test) if x == y) / len(NoStem_FreqCount_doc_class)
    Stem_Binary_accuracy = sum(1 for x,y in zip(Stem_Binary_doc_class,y_test) if x == y) / len(NoStem_FreqCount_doc_class)
    
    return({"No-Stemming + Frequency Count docs labels" : NoStem_FreqCount_doc_class,
            "No-Stemming + Binary docs labels" : NoStem_Binary_doc_class,
            "Stemming + Frequency Count docs labels" : Stem_FreqCount_doc_class,
            "Stemming + Binary docs labels" : Stem_Binary_doc_class,
            "No-Stemming + Frequency Count Accuracy" : NoStem_FreqCount_accuray,
            "No-Stemming + Binary Accuracy" : NoStem_Binary_accuracy,
            "Stemming + Frequency Count Accuracy" : Stem_FreqCount_accuray,
            "Stemming + Binary Accuracy" : Stem_Binary_accuracy
           })

#-----

# To calculate confusion matrix (eg.)
pd.DataFrame(confusion_matrix(evaluation_metrics['Stemming + Frequency Count docs labels'], y_test),
             columns = ['neg', 'pos'], index = ['neg', 'pos'])


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