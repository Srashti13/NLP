"""
AIT726 HW 1 Due 9/26/2019
Sentiment classificaiton using Naive Bayes and Logistic Regression on a dataset of 25000 training and 25000 testing tweets.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman

"""
import os
import pandas as pd
import re
import time
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from collections import defaultdict
from sklearn.metrics import confusion_matrix


#emoji regex
start_time = time.time()
emoticon_string = r"(:\)|:-\)|:\(|:-\(|;\);-\)|:-O|8-|:P|:D|:\||:S|:\$|:@|8o\||\+o\(|\(H\)|\(C\)|\(\?\))"
#https://www.regexpal.com/96995


def main():
    '''
    main function
    '''
    print("Start Program --- %s seconds ---" % (time.time() - start_time))
    vocabulary, vocabulary_stemmed, trainingdocs, trainingdocs_stemmed, y_train, testdocs, testdocs_stemmed, y_test = get_trainandtest_vocabanddocs()
    trainbow_freq, trainbow_stem_freq, trainbow_binary, trainbow_stem_binary, vocab_dict, stem_vocab_dict = get_BOW(trainingdocs, trainingdocs_stemmed, vocabulary, vocabulary_stemmed)
    P_positive, P_negative  = get_class_priors(y_train)
    wordlikelihood_dict = get_perword_likelihood(trainbow_freq, trainbow_stem_freq, trainbow_binary, trainbow_stem_binary,vocab_dict, stem_vocab_dict, y_train)
    predictions = predict_NB(wordlikelihood_dict,P_positive, P_negative, testdocs, testdocs_stemmed, y_test)
    evaluate(predictions[0], y_test, "NB-NOSTEM-FREQ")
    evaluate(predictions[1], y_test, "NB-NOSTEM-BINARY")
    evaluate(predictions[2], y_test, "NB-STEM-FREQ")
    evaluate(predictions[3], y_test, "NB-STEM-BINARY")
    LR_model = Logistic_Regression_L2_SGD(n_iter=10,eta=0.05, batch_size=len(trainbow_freq))
    LR_model.fit(trainbow_freq, y_train)
    predictions = LR_model.predict(trainbow_freq)
    evaluate(predictions, y_test, "LOGISTIC_FREQ_NOL2")


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
        tokens = tokensfinal
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

    print("Train Docs Prepared --- %s seconds ---" % (time.time() - start_time))
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

    print("Test Docs Prepared --- %s seconds ---" % (time.time() - start_time))
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

    print("Vectors Created --- %s seconds ---" % (time.time() - start_time))    
    return trainbow_freq, trainbow_stem_freq, trainbow_binary, trainbow_stem_binary, vocab_dict, stem_vocab_dict

def get_class_priors(y_train):
    '''
    calculate the prior for each class = number of samples of class C in training set / total number of samples in training set (25000)
    Pˆ(c) = Nc/N
    '''
    P_negative = list(y_train).count(0)/y_train.size
    P_positive = list(y_train).count(1)/y_train.size

    print("Priors Assessed --- %s seconds ---" % (time.time() - start_time))
    return P_negative, P_positive

def get_perword_likelihood( trainbow_freq, trainbow_stem_freq, trainbow_binary, trainbow_stem_binary,vocab_dict, stem_vocab_dict, y_train):
    '''
    Pˆ(w | c) = count(w, c)+1 /(count(c)+ |V|)  

    depends on vocabulary being stemmed/non-stemmed and the type of vectors being used
    '''
    ## frequency Pˆ(w | c) = count(w, c)+1 /(count(c)+ |V|) 
    frequencydict_pos = defaultdict()
    frequencydict_neg = defaultdict()
    frequencydict_pos_stem = defaultdict()
    frequencydict_neg_stem = defaultdict()
    num_pos = list(y_train).count(1)

    ## frequency not stemmed ##  
    denom_pos =trainbow_freq[(1+num_pos):,:].sum()+len(vocab_dict.keys()) #sum of all positive words and vocab size
    denom_neg = trainbow_freq[:num_pos:,:].sum()+len(vocab_dict.keys()) #sum of all negative words and vocab size
    trainbow_freq_pos_sum = trainbow_freq[(1+num_pos):,:].sum(axis = 0) #per word summing for positive class
    trainbow_freq_neg_sum = trainbow_freq[:num_pos,:].sum(axis = 0) # per word summing for negative class
    
    for v in vocab_dict.keys():

        frequencydict_pos[v] = (trainbow_freq_pos_sum[vocab_dict[v]]+1) /(denom_pos)
        frequencydict_neg[v] = (trainbow_freq_neg_sum[vocab_dict[v]]+1) /(denom_neg)

    ## frequency stemmed ##  
    denom_stem_pos = trainbow_stem_freq[(1+num_pos):,:].sum()+len(stem_vocab_dict.keys()) #sum of all positive words and vocab size
    denom_stem_neg = trainbow_stem_freq[:num_pos,:].sum()+len(stem_vocab_dict.keys()) #sum of all negative words and vocab size
    trainbow_freq_stem_pos_sum = trainbow_stem_freq[(1+num_pos):,:].sum(axis = 0) #per word summing for positive class
    trainbow_freq_stem_neg_sum = trainbow_stem_freq[:num_pos,:].sum(axis = 0) # per word summing for negative class

    for v in stem_vocab_dict.keys():

        frequencydict_pos_stem[v] = (trainbow_freq_stem_pos_sum[stem_vocab_dict[v]]+1) /(denom_stem_pos)
        frequencydict_neg_stem[v] = (trainbow_freq_stem_neg_sum[stem_vocab_dict[v]]+1) /(denom_stem_neg)

    # binary ##   P^(xi∣ωj)=dfxi,y+1 / dfy+2 Multi-variate Bernoulli Naive Bayes https://sebastianraschka.com/Articles/2014_naive_bayes_1.html
    binarydict_pos = defaultdict()
    binarydict_neg = defaultdict()
    binarydict_pos_stem = defaultdict()
    binarydict_neg_stem = defaultdict()

    ## binary not stemmed ## 
    trainbow_binary_pos_sum = trainbow_binary[(1+num_pos):,:].sum(axis = 0) #per word summing for positive class
    trainbow_binary_neg_sum = trainbow_binary[:num_pos,:].sum(axis = 0) # per word summing for negative class
    pos_docs = num_pos #number of positive documents
    neg_docs = list(y_train).count(1) #number of negative documents
    
    for v in vocab_dict.keys():
            
        binarydict_pos[v] = (trainbow_binary_pos_sum[vocab_dict[v]]+1) /((pos_docs) + 2)
        binarydict_neg[v] = (trainbow_binary_neg_sum[vocab_dict[v]]+1) /((neg_docs) + 2)

    ## binary stemmed ##
    trainbow_binary_stem_pos_sum = trainbow_stem_binary[(1+num_pos):,:].sum(axis = 0) #per word summing for positive class
    trainbow_binary_stem_neg_sum = trainbow_stem_binary[:num_pos,:].sum(axis = 0) # per word summing for negative class

    for v in stem_vocab_dict.keys():
            
        binarydict_pos_stem[v] = (trainbow_binary_stem_pos_sum[stem_vocab_dict[v]]+1) /((pos_docs) + 2)
        binarydict_neg_stem[v] = (trainbow_binary_stem_neg_sum[stem_vocab_dict[v]]+1) /((neg_docs) + 2)

    wordlikelihood_dict = {"frequencydict_pos":frequencydict_pos, "frequencydict_neg":frequencydict_neg, \
                                "frequencydict_pos_stem":frequencydict_pos_stem, "frequencydict_neg_stem":frequencydict_neg_stem, \
                                "binarydict_pos":binarydict_pos, "binarydict_neg":binarydict_neg, \
                                "binarydict_pos_stem":binarydict_pos_stem, "binarydict_neg_stem":binarydict_neg_stem}
                            
    print("Word Likelihoods Calculated --- %s seconds ---" % (time.time() - start_time))
    return wordlikelihood_dict

def predict_NB(wordlikelihood_dict,P_positive, P_negative, testdocs, testdocs_stemmed, y_test):
    '''
    Compute the most likely class for each document in the test set using each of the
    combinations of stemming + frequency count, stemming + binary, no-stemming + frequency
    count, no-stemming + binary.
    '''
    # log scale: http://www.cs.rhodes.edu/~kirlinp/courses/ai/f18/projects/proj3/naive-bayes-log-probs.pdf
    
    # No-Stemming
    y_pred_freq = np.zeros(y_test.size)
    y_pred_binary = np.zeros(y_test.size)
    for i, documt in enumerate(testdocs):
        prob_pos_case1 = np.log(P_positive)
        prob_pos_case2 = np.log(P_positive)
        prob_neg_case1 = np.log(P_negative)
        prob_neg_case2 = np.log(P_negative)
        for word in documt:
            if word in wordlikelihood_dict["frequencydict_pos"].keys():
                #pos
                prob_pos_case1 += np.log(wordlikelihood_dict["frequencydict_pos"][word])
                prob_pos_case2 += np.log(wordlikelihood_dict["binarydict_pos"][word])
                #neg
                prob_neg_case1 += np.log(wordlikelihood_dict["frequencydict_neg"][word])
                prob_neg_case2 += np.log(wordlikelihood_dict["binarydict_neg"][word])
        
        # No-Stemming + Frequency count
        if prob_pos_case1 > prob_neg_case1:
            y_pred_freq[i] = 1
        else:
            y_pred_freq[i] = 0
            
        # No-Stemming + Binary
        if prob_pos_case2 > prob_neg_case2:
            y_pred_binary[i] = 1
        else:
            y_pred_binary[i] = 0
            
    # Stemming
    y_pred_freq_stem = np.zeros(y_test.size)
    y_pred_binary_stem = np.zeros(y_test.size)
    for i, documt in enumerate(testdocs_stemmed):
        prob_pos_case3 = np.log(P_positive)
        prob_pos_case4 = np.log(P_positive)
        prob_neg_case3 = np.log(P_negative)
        prob_neg_case4 = np.log(P_negative)
        for word in documt:
            if word in wordlikelihood_dict["frequencydict_pos_stem"].keys():
                #pos
                prob_pos_case3 += np.log(wordlikelihood_dict["frequencydict_pos_stem"][word])
                prob_pos_case4 += np.log(wordlikelihood_dict["binarydict_pos_stem"][word])
                #neg
                prob_neg_case3 += np.log(wordlikelihood_dict["frequencydict_neg_stem"][word])
                prob_neg_case4 += np.log(wordlikelihood_dict["binarydict_neg_stem"][word])
                        
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

    predictions = [y_pred_freq, y_pred_binary, y_pred_freq_stem, y_pred_binary_stem]
    print("NB Predictions Acquired --- %s seconds ---" % (time.time() - start_time))
    return predictions

def evaluate(predictions,y_test, model):
    '''
        For each of your classifiers, compute and report accuracy. Accuracy is
    number of correctly classified reviews/number of all reviews in test (25k for our case). Also
    create a confusion matrix for each classifier. Save your output in a .txt or .log file.
    '''
    print("Evaulating  %s --- %s seconds ---" % (model,(time.time() - start_time)))
    score = 0
    for i in range(y_test.size):
        if predictions[i] == y_test[i]:
            score +=1
    accuracy = score*100/float(y_test.size)
    print(accuracy)

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
        print("Fitting Logistic Regression --- %s seconds ---" % (time.time() - start_time))
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