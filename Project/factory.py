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
import gc #garbage collector for gpu memory 
from GPUtil import showUtilization as gpu_usage
import pandas as pd



start_time = time.time()


def main():
    '''
    The main function. This is used to get/tokenize the documents, create vectors for input into the language model based on
    a number of grams, and input the vectors into the model for training and evaluation.
    '''
    train_size = 5000 #1306112 is full dataset
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    vocab, train_questions, train_labels, test_questions = get_docs(train_size) 
    train_context_array, train_context_label_array, test_context_array ,train_ids,test_ids , totalpadlength = get_context_vector(vocab, train_questions, train_labels, test_questions) 
    # run_neural_network(context_array, context_label_array, len(vocab), train_size, totalpadlength)
    # pretrained_embedding_run_NN(context_array, context_label_array, len(vocab), vocab, train_size,totalpadlength)
    run_RNN(train_context_array, train_context_label_array, test_context_array, len(vocab), 4000, totalpadlength)

    # baseline_models(context_array, context_label_array, vocab, train_size, totalpadlength)
    return

def get_docs(train_size):

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


    #laod data and tokenize
    train = pd.read_csv(r'quora-insincere-questions-classification/train.csv')
    train_questions = train['question_text']
    train_labels = train[:train_size]['target']
    train_questions = train_questions[:train_size].apply(tokenize)
    
    test = pd.read_csv(r'quora-insincere-questions-classification/train.csv')
    test_questions = test[:500]['question_text']
    test_questions = test_questions[:500].apply(tokenize)
    
    
    
    
    total_questions = pd.concat((train_questions,test_questions), axis=0)
    vocab = list(set([item for sublist in total_questions.values for item in sublist]))
    print("--- Text Extracted --- %s seconds ---" % (round((time.time() - start_time),2)))   
    return vocab, train_questions, train_labels, test_questions

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
        
    
    train_ids = []
    for entry in train_questions.index:
        train_ids.append(entry)
        
    test_ids = []
    for entry in test_questions.index:
        test_ids.append(entry)
        
    #convert to numpy array for use in torch  -- padding with index 0 for padding.... Should change to a random word...
    totalpadlength = max(map(len, train_context_values))
    train_context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in train_context_values]) #needed because without padding we are lost 
    test_context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in test_context_values]) #needed because without padding we are lost 
    train_context_label_array = np.array(train_context_labels) 


    print("--- Grams Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return train_context_array, train_context_label_array, test_context_array ,train_ids,test_ids, totalpadlength

def run_RNN(train_context_array, train_context_label_array, test_context_array, vocab_size, train_size, totalpadlength):
    '''
    RNN version 
    '''
    
    BATCH_SIZE = 50 # 1000 maxes memory for 8GB GPU -- keep set to 1 to predict all test cases in current implementation

    #randomly split into test and validation sets
    X_train, y_train = train_context_array[:(train_size)][:], train_context_label_array[:(train_size)][:]

    X_test, y_test = train_context_array[(train_size):][:], train_context_label_array[(train_size):][:]

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

    class RNNmodel(nn.Module):
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
            super(RNNmodel, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim) 
            self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size)
            self.linear = nn.Linear(hidden_size, 1)
            self.out_act = nn.Sigmoid()
    
        def forward(self, inputs, context_size, embedding_dim):
            embeds = self.embeddings(inputs).view((context_size,-1,embedding_dim)) #required dimensions for batching 
            # print(embeds.shape)
            out1, _ = self.rnn(embeds)
            # print(out1.shape)
            out2 = self.linear(out1[-1,:,:])
            # print(out2.shape)
            yhat = self.out_act(out2)
            # print(yhat)
            return yhat

            
    #initalize model parameters and variables
    losses = []
    loss_function = nn.BCELoss() #binary cross entropy produced best results
    model = RNNmodel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE) #.to_fp16() for memory
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
            labelsfull.extend(label.int().tolist())
        f1score = f1_score(labelsfull,predictionsfull,average='macro') #not sure if they are using macro or micro in competition
        print('Test F1: {} %'.format(round(f1score,5)))
    return

if __name__ == "__main__":
    main()
