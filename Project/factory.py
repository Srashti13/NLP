"""
AIT726 Project -- Insincere Question Classification Due 10/10/2019
https://www.kaggle.com/c/quora-insincere-questions-classification

This project deals with the 
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW2.py 
i. main - runs all of the functions
    i. get_docs - tokenizes all tweets, returns a list of tokenized sentences and a list of all tokens
    ii. get_ngrams_vector 
"""
import os
import re
import time
import numpy as np
import itertools
import csv
from nltk.util import ngrams
from nltk import word_tokenize
from nltk import sent_tokenize
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from statistics import mean
import string
import random
import torch.utils.data as data_utils
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn 
import pandas as pd
import gc #garbage collector for gpu memory 
# from GPUtil import showUtilization as gpu_usage



start_time = time.time()


def main():
    '''
    The main function. This is used to get/tokenize the documents, create vectors for input into the language model based on
    a number of grams, and input the vectors into the model for training and evaluation.
    '''
    train_size = 100000 #1306112 is full dataset
    readytosubmit=False
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    vocab, train_questions, train_labels, test_questions, train_ids, test_ids = get_docs(train_size, readytosubmit) 
    train_context_array, train_context_label_array, test_context_array, totalpadlength = get_context_vector(vocab, train_questions, train_labels, test_questions)
    unique, cnts = np.unique(train_context_label_array, return_counts=True)
    print(dict(zip(unique, cnts)))
    run_neural_network(train_context_array, train_context_label_array,test_context_array,test_ids, len(vocab), train_size, totalpadlength,readytosubmit)
    # pretrained_embedding_run_NN(train_context_array, train_context_label_array,test_context_array,test_ids, len(vocab), vocab, train_size,totalpadlength,readytosubmit)
    # run_RNN(train_context_array, train_context_label_array,test_context_array,test_ids, len(vocab), train_size, totalpadlength,readytosubmit)
    # baseline_models(train_context_array, train_context_label_array,test_context_array,test_ids, vocab, train_size, totalpadlength)
    return

def get_docs(train_size, readytosubmit):

    '''
    Pre-processing: Read the complete data word by word. Remove any markup tags, e.g., HTML
    tags, from the data. Lower case capitalized words (i.e., starts with a capital letter) but not all
    capital words (e.g., USA). Do not remove stopwords. Tokenize at white space and also at each
    punctuation. Consider emoticons in this process. You can use an emoticon tokenizer, if you so
    choose. If yes, specify which one. 

    This function tokenizes and gets all of the text from the documents. it also divides the text into sentences 
    and tokenizes each sentence. That way our model doesn't learn weird crossovers between the end of one sentence
    to the start of another. 
    '''

    def tokenize(txt):
        """
        Remove any markup tags, e.g., HTML
        tags, from the data. Lower case capitalized words (i.e., starts with a capital letter) but not all
        capital words (e.g., USA). Do not remove stopwords. Tokenize at white space and also at each
        punctuation. Consider emoticons in this process. You can use an emoticon tokenizer, if you so
        choose.
        Tokenizer that tokenizes text. Also finds and tokenizes emoji faces.
        """
        txt = re.sub(r'\d+', '', txt) #remove numbers
        txt = txt.translate(str.maketrans('', '', string.punctuation)) #removes punctuation - not used as per requirements
        def lower_repl(match):
            return match.group(1).lower()

        # txt = r"This is a practice tweet :). Let's hope our-system can get it right. \U0001F923 something."
        txt = re.sub('(?:<[^>]+>)', '', txt)# remove html tags
        txt = re.sub('([A-Z][a-z]+)',lower_repl,txt) #lowercase words that start with captial
        txt=txt.lower()
        tokens = word_tokenize(txt)
        return tokens

    #initalize variables
    questions = defaultdict()
    labels = defaultdict()
    docs = []
    #laod data and tokenize
    train = pd.read_csv(r'kaggle/input/quora-insincere-questions-classification/train.csv',nrows=train_size)
    train_questions = train['question_text']
    train_labels = train[:train_size]['target']
    train_ids = train['qid']
    train_questions = train_questions[:train_size].apply(tokenize)
    
    if readytosubmit:
        test = pd.read_csv(r'kaggle/input/quora-insincere-questions-classification/test.csv')
    else:
        test = pd.read_csv(r'kaggle/input/quora-insincere-questions-classification/test.csv',nrows=500)
    test_questions = test['question_text']
    test_ids = test['qid']
    test_questions = test_questions.apply(tokenize)
    
    
    
    
    total_questions = pd.concat((train_questions,test_questions), axis=0)
    vocab = list(set([item for sublist in total_questions.values for item in sublist]))
    print("--- Text Extracted --- %s seconds ---" % (round((time.time() - start_time),2)))  
    return vocab, train_questions, train_labels, test_questions, train_ids, test_ids

def get_context_vector(vocab, train_questions, train_labels, test_questions):
    '''
    Construct your n-grams: Create positive n-gram samples by collecting all pairs of adjacent
    tokens. Create 2 negative samples for each positive sample by keeping the first word the same
    as the positive sample, but randomly sampling the rest of the corpus for the second word. The
    second word can be any word in the corpus except for the first word itself. 
    
    This functions takes the docs and tokenized sentences and creates the numpyarrays needed for the neural network.
    --creates 2 fake grams for every real gram 
    '''

    word_to_ix = {word: i for i, word in enumerate(vocab)} #index vocabulary

    train_context_values = [] #array of word index for context 
    for context in train_questions.values:
        train_context_values.append([word_to_ix[w] for w in context])

    test_context_values = [] #array of word index for context 
    for context in test_questions.values:
        test_context_values.append([word_to_ix[w] for w in context])
    
    train_context_labels = [] # list of labels for context
    for label in train_labels.values:
        train_context_labels.append([label])
        
    #convert to numpy array for use in torch  -- padding with index 0 for padding.... Should change to a random word...
    totalpadlength = 70 #max(map(len, test_context_values)) #the longest question is in the test set 
    train_context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in train_context_values]) #needed because without padding we are lost 
    test_context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in test_context_values]) #needed because without padding we are lost 
    train_context_label_array = np.array(train_context_labels) 


    print("--- Grams Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return train_context_array, train_context_label_array, test_context_array, totalpadlength

def run_neural_network(context_array, context_label_array,test_context_array, test_ids, vocab_size, train_size, totalpadlength, readytosubmit):
    '''
    regular FeedForward without pretrained embeddings
    '''
    
    BATCH_SIZE = 5000 # 1000 maxes memory for 8GB GPU -- keep set to 1 to predict all test cases in current implementation

    #randomly split into test and validation sets
    X_train, y_train = context_array, context_label_array

    X_test, y_test = test_context_array, np.zeros(len(test_context_array))

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 
                                                       random_state=1234, shuffle=True, stratify=y_train)
    
    #set datatypes 
    X_train = torch.from_numpy(X_train)
    X_train = X_train.long()
    y_train = torch.from_numpy(y_train)
    y_train = y_train.float()
    X_valid = torch.from_numpy(X_valid)
    X_valid = X_valid.long()
    y_valid = torch.from_numpy(y_valid)
    y_valid = y_valid.float()
    X_test = torch.from_numpy(X_test)
    X_test = X_test.long()
    y_test = torch.from_numpy(y_test)
    y_test = y_test.float()
    
    #create datsets for loading into models
    train = data_utils.TensorDataset(X_train, y_train)
    valid = data_utils.TensorDataset(X_valid, y_valid)
    test = data_utils.TensorDataset(X_test, y_test)
    trainloader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    validloader = data_utils.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    testloader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    
    
    #edit as deisred
    EMBEDDING_DIM = 25 # embeddings dimensions
    CONTEXT_SIZE = totalpadlength # total length of padded questions size
    HIDDEN_SIZE = 20 # nodes in hidden layer

    class FeedForward(nn.Module):
        '''
        Build and train a feed forward neural network: Build your FFNN with 2 layers (1 hidden layer and
        1 output layer) with hidden vector size 20. Initialize the weights with random numbers.
        Experiment with mean squared error and cross entropy as your loss function.
        Creates a Ngram based feedforward neural network with an embeddings layer, 1 hidden layer of 'hidden_size' units (20 in
        this case seemed to work best- changing to higher values had litte improvmeent), and a single output unit for 
        binary classification. Sigmoid activation function is used to obtain a percentage. Learning rate of .00001 was 
        too low to effectively implement in a resonable amount of time. It is set to 0.0001 for demonstration purposes. 
        '''
        def __init__(self, vocab_size, embedding_dim, context_size, hidden_size):
            super(FeedForward, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim) 
            self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
            self.linear2 = nn.Linear(hidden_size, 1)
            self.out_act = nn.Sigmoid()
    
        def forward(self, inputs, context_size, embedding_dim):
            embeds = self.embeddings(inputs).view((-1, context_size*embedding_dim)) #required dimensions for batching 
            out1 = self.linear1(embeds)
            out3 = self.linear2(out1)
            yhat = self.out_act(out3)
            return yhat

    

            
    #initalize model parameters and variables
    losses = []
    loss_function = nn.BCELoss() #binary cross entropy produced best results
    # Experimenting with MSE Loss
    #loss_function = nn.MSELoss()
    model = FeedForward(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE) #.to_fp16() for memory
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01) #learning rate set to 0.0001 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    f1_list = []
    best_f1 = 0 
    print("Start Training --- %s seconds ---" % (round((time.time() - start_time),2)))
    for epoch in range(50): 
        iteration = 0
        running_loss = 0.0 
        for i, (context, label) in enumerate(trainloader):
            # zero out the gradients from the old instance
            optimizer.zero_grad()
            # Run the forward pass and get predicted output
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM) #required dimensions for batching
            yhat = yhat.view(-1,1)
            # Compute Binary Cross-Entropy
            loss = loss_function(yhat, label)
            #clear memory 
            del context, label #memory 
            # Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
            iteration += 1
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            running_loss += float(loss.item())
            torch.cuda.empty_cache() #memory
        losses.append(float(loss.item()))
        del loss #memory 
        gc.collect() #memory
        torch.cuda.empty_cache() #memory

    # Get the accuracy on the validation set for each epoch
        with torch.no_grad():
            predictionsfull = []
            labelsfull = []
            for a, (context, label) in enumerate(validloader):
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
                predictions = (yhat > 0.5)
                predictionsfull.extend(predictions.int().tolist())
                labelsfull.extend(label.int().tolist())
                del context, label, predictions #memory
            gc.collect()#memory
            torch.cuda.empty_cache()#memory
            # print('\n')
            # gpu_usage()
            f1score = f1_score(labelsfull,predictionsfull,average='macro') #not sure if they are using macro or micro in competition
            f1_list.append(f1score)
        print('--- Epoch: {} | Validation F1: {} ---'.format(epoch+1, f1_list[-1])) 

        if f1_list[-1] > best_f1: #save if it improves validation accuracy 
            best_f1 = f1_list[-1]
            bestmodelparams = torch.save(model.state_dict(), 'train_valid_best.pth') #save best model
        #early stopping condition
        if epoch+1 >= 5: #start looking to stop after this many epochs
            if f1_list[-1] < min(f1_list[-5:-1]): #if accuracy lower than lowest of last 4 values
                print('...Stopping Early...')
                break

    print("Training Complete --- %s seconds ---" % (round((time.time() - start_time),2)))
    # Get the accuracy on the test set after training complete -- will have to submit to KAGGLE --IGNORE THIS
    model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
    with torch.no_grad():
        total = 0
        num_correct = 0
        predictionsfull = []
        labelsfull = []
        for a, (context, label) in enumerate(testloader):
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
            yhat = yhat.view(-1,1)
            predictions = (yhat > 0.5)
            total += label.nelement()
            predictionsfull.extend(predictions.int().tolist())

    #outputs results to csv
    predictionsfinal = []
    for element in predictionsfull:
        predictionsfinal.append(element[0])
    output = pd.DataFrame(np.array([test_ids,predictionsfinal])).transpose()
    output.columns = ['qid', 'prediction']
    print(output.head())
    if readytosubmit:
        output.to_csv('/samplesubmission', index=False)
    return

def pretrained_embedding_run_NN(context_array, context_label_array,test_context_array, test_ids, vocab_size, vocab, train_size,totalpadlength, readytosubmit):
    '''
    This function is the same as run_neural_network except it uses pretrained embeddings loaded from a file
    '''
    BATCH_SIZE = 500 # 1000 maxes memory for 8GB GPU

    #randomly split into test and validation sets
    X_train, y_train = context_array, context_label_array

    X_test, y_test = test_context_array, np.zeros(len(test_context_array))

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 
                                                       random_state=1234, shuffle=True, stratify=y_train)
    
    #set datatypes 
    X_train = torch.from_numpy(X_train)
    X_train = X_train.long()
    y_train = torch.from_numpy(y_train)
    y_train = y_train.float()
    X_valid = torch.from_numpy(X_valid)
    X_valid = X_valid.long()
    y_valid = torch.from_numpy(y_valid)
    y_valid = y_valid.float()
    X_test = torch.from_numpy(X_test)
    X_test = X_test.long()
    y_test = torch.from_numpy(y_test)
    y_test = y_test.float()
    
    #create datsets for loading into models
    train = data_utils.TensorDataset(X_train, y_train)
    valid = data_utils.TensorDataset(X_valid, y_valid)
    test = data_utils.TensorDataset(X_test, y_test)
    trainloader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    validloader = data_utils.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    testloader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    
    print("--- Building Pretrained Embedding Index  --- %s seconds ---" % (round((time.time() - start_time),2)))
    EMBEDDING_DIM = 200 # embeddings dimensions
    CONTEXT_SIZE = totalpadlength #sentence size
    
    # getting embeddings from the file
    EMBEDDING_FILE = "kaggle/input/quora-insincere-questions-classification/embeddings/glove.6B.200d.txt"
    embeddings_index = {}
    words = []
    with open (EMBEDDING_FILE, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            words.append(word)
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    matrix_len = vocab_size
    weights_matrix = np.zeros((matrix_len, EMBEDDING_DIM)) # 200 is depth of embedding matrix
    words_found = 0
    words_not_found = 0
    for i, word in enumerate(vocab):
        try:
            weights_matrix[i] = embeddings_index[word]
            words_found += 1
        
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,)) #randomize out of vocabulary words
            words_not_found += 1
    
    print("{:.2f}% ({}/{}) of the vocabulary were in the pre-trained embedding.".format((words_found/vocab_size)*100,words_found,vocab_size))
    
    weights_matrix_torch = torch.from_numpy(weights_matrix)
    
    def create_emb_layer(weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim
    
    class FeedForward(nn.Module):
        def __init__(self, weights_matrix, context_size):
            super(FeedForward, self).__init__()
            self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
            self.linear1 = nn.Linear(embedding_dim*context_size, 1)
            self.out_act = nn.Sigmoid()
            
        def forward(self, inputs, context_size, embedding_dim):
            embeds = self.embedding(inputs).view((-1, context_size*embedding_dim))
            out1 = self.linear1(embeds)
            yhat = self.out_act(out1)
            return yhat 

    #initalize model parameters and variables
    losses = []
    loss_function = nn.BCELoss() #binary cross entropy produced best results
    # Experimenting with MSE Loss
    #loss_function = nn.MSELoss()
    model = FeedForward(weights_matrix_torch, CONTEXT_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...
    # The random weight method isn't as effective as the default pytorch method
    #model.apply(random_weights)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01) #learning rate set to 0.0001 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    f1_list = []
    best_f1 = 0 
    print("Start Training (Pre-trained) --- %s seconds ---" % (round((time.time() - start_time),2)))
    for epoch in range(50): 
        iteration = 0
        running_loss = 0.0 
        for i, (context, label) in enumerate(trainloader):
            # zero out the gradients from the old instance
            optimizer.zero_grad()
            # Run the forward pass and get predicted output
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM) #required dimensions for batching
            yhat = yhat.view(-1,1)
            # Compute Binary Cross-Entropy
            loss = loss_function(yhat, label)
            #clear memory 
            del context, label #memory 
            # Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
            iteration += 1
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            running_loss += float(loss.item())
            torch.cuda.empty_cache() #memory
        losses.append(float(loss.item()))
        del loss #memory 
        gc.collect() #memory
        torch.cuda.empty_cache() #memory

    # Get the accuracy on the validation set for each epoch
        with torch.no_grad():
            predictionsfull = []
            labelsfull = []
            for a, (context, label) in enumerate(validloader):
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
                predictions = (yhat > 0.5)
                predictionsfull.extend(predictions.int().tolist())
                labelsfull.extend(label.int().tolist())
                del context, label, predictions #memory
            gc.collect()#memory
            torch.cuda.empty_cache()#memory
            # print('\n')
            # gpu_usage()
            f1score = f1_score(labelsfull,predictionsfull,average='macro') #not sure if they are using macro or micro in competition
            f1_list.append(f1score)
        print('--- Epoch: {} | Validation F1: {} ---'.format(epoch+1, f1_list[-1])) 

        if f1_list[-1] > best_f1: #save if it improves validation accuracy 
            best_f1 = f1_list[-1]
            bestmodelparams = torch.save(model.state_dict(), 'train_valid_best.pth') #save best model
        #early stopping condition
        if epoch+1 >= 5: #start looking to stop after this many epochs
            if f1_list[-1] < min(f1_list[-5:-1]): #if accuracy lower than lowest of last 4 values
                print('...Stopping Early...')
                break

    print("Training Complete --- %s seconds ---" % (round((time.time() - start_time),2)))
    # Get the accuracy on the test set after training complete -- will have to submit to KAGGLE --IGNORE THIS
    model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
    with torch.no_grad():
        total = 0
        num_correct = 0
        predictionsfull = []
        labelsfull = []
        for a, (context, label) in enumerate(testloader):
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
            yhat = yhat.view(-1,1)
            predictions = (yhat > 0.5)
            total += label.nelement()
            predictionsfull.extend(predictions.int().tolist())

    #outputs results to csv
    predictionsfinal = []
    for element in predictionsfull:
        predictionsfinal.append(element[0])
    output = pd.DataFrame(np.array([test_ids,predictionsfinal])).transpose()
    output.columns = ['qid', 'prediction']
    print(output.head())
    if readytosubmit:
        output.to_csv('samplesubmission', index=False)
def baseline_models(context_array, context_label_array,test_context_array, test_ids, vocab, train_size, totalpadlength):
    '''
    Baseline Logistic and NB using pretrained embeddings of each word as the feature vectors.
    '''
    print("--- Building Embedding Index --- %s seconds ---" % (round((time.time() - start_time),2)))
    #get embedding DFs from context array
    EMBEDDING_DIM = 200 # embeddings dimensions
    
    # getting embeddings from the file
    EMBEDDING_FILE = "Embeddings/glove.6B.200d.txt"
    embeddings_index = {}
    words = []
    with open (EMBEDDING_FILE, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            words.append(word)
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, EMBEDDING_DIM)) # 200 is depth of embedding matrix
    words_found = 0
    words_not_found = 0
    for i, word in enumerate(vocab):
        try:
            weights_matrix[i] = embeddings_index[word]
            words_found += 1
        
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,)) #randomize out of vocabulary words
            words_not_found += 1
    
    print("{:.2f}% ({}/{}) of the vocabulary were in the pre-trained embedding.".format((words_found/len(vocab))*100,words_found,len(vocab)))
    
    weights_matrix  #embedding given word
    average_weights = np.mean(weights_matrix,axis=1)
    del weights_matrix
    df = []
    for x in np.nditer(context_array):
        avgvalue = average_weights[x]
        df.extend([str(avgvalue)]) #mean of embeddings to save memory

    # print(df) # context array with embeddings as features instead
    df = np.asarray(df).astype(np.float)
    df = df.reshape((len(context_array),totalpadlength)) #reshape to number of questions by  number of words
    print(df.shape)

    #randomly split into test and validation sets
    X_train, y_train = context_array, context_label_array

    X_test, y_test = test_context_array, np.zeros(len(test_context_array))

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.97, 
                                                       random_state=1234, shuffle=True, stratify=y_train)
    print("--- Start Training (Baselines) --- %s seconds ---" % (round((time.time() - start_time),2)))

    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    NB_scores = cross_val_score(GaussianNB(), X_train, y_train.ravel(), cv=2, scoring="f1_macro")   #could also use x_small
    print(NB_scores)
    
    LR_scores = cross_val_score(LogisticRegression(n_jobs=-1,max_iter=300, solver='lbfgs'), X_train, y_train.ravel(), cv=2, scoring="f1_macro")   
    print(LR_scores)
    return

    #do baseline models with embedding feature vectors

def run_RNN(context_array, context_label_array,test_context_array, test_ids, vocab_size, train_size, totalpadlength, readytosubmit):
    '''
    RNN version 
    '''
    
    BATCH_SIZE = 500 # 1000 maxes memory for 8GB GPU -- keep set to 1 to predict all test cases in current implementation

    #randomly split into test and validation sets
    X_train, y_train = context_array, context_label_array

    X_test, y_test = test_context_array, np.zeros(len(test_context_array))

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 
                                                       random_state=1234, shuffle=True, stratify=y_train)
    
    #set datatypes 
    X_train = torch.from_numpy(X_train)
    X_train = X_train.long()
    y_train = torch.from_numpy(y_train)
    y_train = y_train.float()
    X_valid = torch.from_numpy(X_valid)
    X_valid = X_valid.long()
    y_valid = torch.from_numpy(y_valid)
    y_valid = y_valid.float()
    X_test = torch.from_numpy(X_test)
    X_test = X_test.long()
    y_test = torch.from_numpy(y_test)
    y_test = y_test.float()
    
    #create datsets for loading into models
    train = data_utils.TensorDataset(X_train, y_train)
    valid = data_utils.TensorDataset(X_valid, y_valid)
    test = data_utils.TensorDataset(X_test, y_test)
    trainloader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    validloader = data_utils.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    testloader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    #edit as deisred
    EMBEDDING_DIM = 25 # embeddings dimensions
    CONTEXT_SIZE = totalpadlength # total length of padded questions size
    HIDDEN_SIZE = 40 # nodes in hidden layer

    class RNNmodel(nn.Module):
        '''
        LSTM 
        '''
        def __init__(self, vocab_size, embedding_dim, context_size, hidden_size):
            super(RNNmodel, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim) 
            self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size*context_size, 1)
            self.out_act = nn.Sigmoid()
    
        def forward(self, inputs, context_size, embedding_dim):
            embeds = self.embeddings(inputs) # [batch, seq, embed dim]
            # print(embeds.shape)
            # print(embeds[0,:3,:3])
            out1, _ = self.rnn(embeds) # [batch, seq_len, num_directions * hidden_size]
            # print(out1.shape)
            # print(out1[:,-1,:].shape)
            out1 = torch.cat([out1[:,:,i] for i in range(out1.shape[2])], dim=1)
            out2 = self.linear(out1) # -> batch size, embed
            # print(out2.shape)
            yhat = self.out_act(out2)
            # print(yhat)
            return yhat
            
        # def __init__(self, vocab_size, embedding_dim, context_size, hidden_size):
        #     super(RNNmodel, self).__init__()
        #     self.embeddings = nn.Embedding(vocab_size, embedding_dim) 
        #     self.rnn = nn.RNN(embedding_dim, hidden_size=hidden_size, batch_first=True)
        #     self.linear = nn.Linear(hidden_size*hidden_size, 1)
        #     self.out_act = nn.Sigmoid()
    
        # def forward(self, inputs, context_size, embedding_dim):
        #     embeds = self.embeddings(inputs) # [batch, seq, embed dim]

        #     #packing
        #     input_packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, 61, batch_first=True)

        #     # print(embeds.shape)
        #     # print(embeds[0,:3,:3])
        #     raw_output, _ = self.rnn(input_packed) # [batch, seq_len, num_directions * hidden_size]
            
        #     output_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(raw_output,total_length=context_size,batch_first=True)
        #     output = output_padded.contiguous()
        #     output_reshaped = output.view(output.shape[0], -1)
        #     # print(out1.shape)
        #     # print(out1[:,-1,:].shape)
        #     out2 = self.linear(output_reshaped) # -> batch size, embed dim
        #     # print(out2.shape)
        #     yhat = self.out_act(out2)
        #     print(yhat)
        #     return yhat
    #initalize model parameters and variables
    losses = []
    loss_function = nn.BCELoss() #binary cross entropy produced best results
    # Experimenting with MSE Loss
    #loss_function = nn.MSELoss()
    model = RNNmodel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE) #.to_fp16() for memory
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1) #learning rate set to 0.0001 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    f1_list = []
    best_f1 = 0 
    print("Start Training --- %s seconds ---" % (round((time.time() - start_time),2)))
    for epoch in range(50): 
        iteration = 0
        running_loss = 0.0 
        for i, (context, label) in enumerate(trainloader):
            # zero out the gradients from the old instance
            optimizer.zero_grad()
            # Run the forward pass and get predicted output
            context = context.to(device)
            # print('start with input')
            # print(context.shape)
            label = label.to(device)
            yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM) #required dimensions for batching
            yhat = yhat.view(-1,1)
            # Compute Binary Cross-Entropy
            loss = loss_function(yhat, label)
            #clear memory 
            del context, label #memory 
            # Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
            iteration += 1
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            running_loss += float(loss.item())
            torch.cuda.empty_cache() #memory
        losses.append(float(loss.item()))
        del loss #memory 
        gc.collect() #memory
        torch.cuda.empty_cache() #memory

    # Get the accuracy on the validation set for each epoch
        with torch.no_grad():
            predictionsfull = []
            labelsfull = []
            for a, (context, label) in enumerate(validloader):
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
                predictions = (yhat > 0.5)
                predictionsfull.extend(predictions.int().tolist())
                labelsfull.extend(label.int().tolist())
                del context, label, predictions #memory
            gc.collect()#memory
            torch.cuda.empty_cache()#memory
            # print('\n')
            # gpu_usage()
            f1score = f1_score(labelsfull,predictionsfull,average='macro') #not sure if they are using macro or micro in competition
            f1_list.append(f1score)
        print('--- Epoch: {} | Validation F1: {} ---'.format(epoch+1, f1_list[-1])) 

        if f1_list[-1] > best_f1: #save if it improves validation accuracy 
            best_f1 = f1_list[-1]
            bestmodelparams = torch.save(model.state_dict(), 'train_valid_best.pth') #save best model
        # #early stopping condition
        # if epoch+1 >= 5: #start looking to stop after this many epochs
        #     if f1_list[-1] < min(f1_list[-5:-1]): #if accuracy lower than lowest of last 4 values
        #         print('...Stopping Early...')
        #         break

    print("Training Complete --- %s seconds ---" % (round((time.time() - start_time),2)))
    # Get the accuracy on the test set after training complete -- will have to submit to KAGGLE --IGNORE THIS
    model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
    with torch.no_grad():
        total = 0
        num_correct = 0
        predictionsfull = []
        labelsfull = []
        for a, (context, label) in enumerate(testloader):
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
            yhat = yhat.view(-1,1)
            predictions = (yhat > 0.5)
            total += label.nelement()
            predictionsfull.extend(predictions.int().tolist())

    #outputs results to csv
    predictionsfinal = []
    for element in predictionsfull:
        predictionsfinal.append(element[0])
    output = pd.DataFrame(np.array([test_ids,predictionsfinal])).transpose()
    output.columns = ['qid', 'prediction']
    print(output.head())
    if readytosubmit:
        output.to_csv('samplesubmission', index=False)
    return

if __name__ == "__main__":
    main()
