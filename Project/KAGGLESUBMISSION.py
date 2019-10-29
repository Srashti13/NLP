# %% [markdown]
# **Code for the AIT 726 Final Project**

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
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
from tqdm.notebook import tqdm_notebook

# %% [code]
start_time = time.time()
def main():
    '''
    The main function. This is used to get/tokenize the documents, create vectors for input into the language model based on
    a number of grams, and input the vectors into the model for training and evaluation.
    '''
    train_size = 1306112 #1306112 is full dataset
    test_size = 500
    readytosubmit=True
    RNN="GRU"
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    vocab, train_questions, train_labels, test_questions, train_ids, test_ids = get_docs(train_size, test_size, readytosubmit) 
    train_context_array, train_context_label_array, test_context_array, totalpadlength = get_context_vector(vocab, train_questions, train_labels, test_questions)
    unique, cnts = np.unique(train_context_label_array, return_counts=True)
    print(dict(zip(unique, cnts)))
    run_RNN(train_context_array, train_context_label_array,test_context_array,test_ids, len(vocab), train_size, totalpadlength,readytosubmit, RNN)
    # pretrained_embedding_run_NN(train_context_array, train_context_label_array,test_context_array,test_ids, len(vocab), vocab, train_size,totalpadlength,readytosubmit)
    # run_RNN(train_context_array, train_context_label_array,test_context_array,test_ids, len(vocab), train_size, totalpadlength,readytosubmit)
    # baseline_models(train_context_array, train_context_label_array,test_context_array,test_ids, vocab, train_size, totalpadlength)
    return

# %% [code]
def get_docs(train_size, test_size, readytosubmit):

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
    train = pd.read_csv(r"../input/quora-insincere-questions-classification/train.csv",nrows=train_size)
    train_questions = train['question_text']
    train_labels = train[:train_size]['target']
    train_ids = train['qid']
    tqdm_notebook.pandas()
    print("----Tokenizing Train Questions----")
    train_questions = train_questions[:train_size].progress_apply(tokenize)
    
    if readytosubmit:
        test = pd.read_csv(r"../input/quora-insincere-questions-classification/test.csv")
    else:
        test = pd.read_csv(r"../input/quora-insincere-questions-classification/test.csv",nrows=test_size)
    test_questions = test['question_text']
    test_ids = test['qid']
    tqdm_notebook.pandas()
    print("----Tokenizing Test Questions----")
    test_questions = test_questions.progress_apply(tokenize)
    
    
    
    
    total_questions = pd.concat((train_questions,test_questions), axis=0)
    vocab = list(set([item for sublist in total_questions.values for item in sublist]))
    print("--- Text Extracted --- %s seconds ---" % (round((time.time() - start_time),2)))  
    return vocab, train_questions, train_labels, test_questions, train_ids, test_ids

# %% [code]
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
    totalpadlength = max(max(map(len, train_context_values)),max(map(len, test_context_values))) #the longest question 
    train_context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in train_context_values]) #needed because without padding we are lost 
    test_context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in test_context_values]) #needed because without padding we are lost 
    train_context_label_array = np.array(train_context_labels) 


    print("--- Grams Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return train_context_array, train_context_label_array, test_context_array, totalpadlength

# %% [code]
def run_RNN(context_array, context_label_array,test_context_array, test_ids, vocab_size, train_size, totalpadlength, readytosubmit, LSTM):
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
        def __init__(self, vocab_size, embedding_dim, context_size, hidden_size, RNN="LSTM"):
            super(RNNmodel, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim) 
            if RNN=="LSTM":
                print("----Using LSTM-----")
                self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, batch_first=True)
            elif RNN=="GRU":
                print("----Using GRU-----")
                self.rnn = nn.GRU(embedding_dim, hidden_size=hidden_size, batch_first=True)
            else:
                print("----Using RNN-----")
                self.rnn = nn.RNN(embedding_dim, hidden_size=hidden_size, batch_first=True)
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
    model = RNNmodel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE,LSTM) #.to_fp16() for memory
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
        output.to_csv('GRUresults.csv', index=False)
    return


# %% [code]
main()

# %% [code]
