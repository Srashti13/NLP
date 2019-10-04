"""
AIT726 HW 2 Due 10/10/2019
Sentiment classificaiton using Naive Bayes and Logistic Regression on a dataset of 25000 training and 25000 testing tweets.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW1.py 
i. main - runs all of the functions
    ii. get_trainandtest_vocabanddocs() - converts dataset into tokens (stemmed and unstemmed), creates megatraining document and extracts vocabulary


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

def main(method='NB'):
    '''
    The main function can be utilized to create vectors, vocabulary, likelihoods for NB, test set predictions and test set evaluation. 
    Comment out the creation steps and merely evaluate the results on the saved data to save time.
    '''

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

    def tokenize(txt,stem=False, model=NB):
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