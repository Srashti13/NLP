"""
AIT726 HW 1 Due 9/26/2019
Sentiment classificaiton using Naive Bayes and Logistic Regression on a dataset of 25000 training and 25000 testing tweets.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman

"""
import os
import pandas as pd
import re
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

def main():
    '''
    main function
    '''
    vocabulary, vocabulary_stemmed, trainingdocs, trainingdocs_stemmed, y_train = get_train_vocab()
    train_BOW_freq, train_BOW_binary, train_BOW_freq_stemmed, train_BOW_binary_stemmed = get_BOW(trainingdocs, trainingdocs_stemmed, vocabulary, vocabulary_stemmed)
    P_positive, P_negative  = get_class_priors()
    wordlikelihood_freq, wordlikelihood_freq_stem, wordlikelihood_binary, wordlikelihood_binary_stem = get_perword_likelihood(vocabulary, vocabulary_stemmed, train_BOW_freq, train_BOW_binary)

    #LR_model = Logistic_Regression_L2_SGD(n_iter=1, batch_size=len(train_BOW_freq))
    #LR_model.fit(train_BOW_binary, y_train)
    #predictions = LR_model.predict(train_BOW_freq)
    #print("Accuracy: {:.0f}%".format(sum(predictions.flatten() == y_train)/len(y_train)*100))

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
        txt = re.sub(r'\d+', '', txt) #remove numbers
        txt = re.sub(r'\d+', '', txt) # do other tokenizing here...
        tokenizer = RegexpTokenizer(r'\w+') #remove punctuation
        tokens = tokenizer.tokenize(txt)
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
    y_train = np.array(list(map(lambda x: 1 if x == 'pos' else 0, y_train)))


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
            
    return trainbow_freq, trainbow_stem_freq, trainbow_binary, trainbow_stem_binary

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
        for i in range(self.n_iter):
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