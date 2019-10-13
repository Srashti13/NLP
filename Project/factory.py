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
import random
import torch.utils.data as data_utils
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn 
import gc #garbage collector for gpu memory 
from GPUtil import showUtilization as gpu_usage


#emoji regex -- used for emoji tokenization
start_time = time.time()
emoticon_string = r"(:\)|:-\)|:\(|:-\(|;\);-\)|:-O|8-|:P|:D|:\||:S|:\$|:@|8o\||\+o\(|\(H\)|\(C\)|\(\?\))"
#https://www.regexpal.com/96995

def main():
    '''
    The main function. This is used to get/tokenize the documents, create vectors for input into the language model based on
    a number of grams, and input the vectors into the model for training and evaluation.
    '''
    train_size = 1306112#1306112
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    vocab, questions, labels = get_docs(train_size) 
    context_array, context_label_array, ids, totalpadlength = get_context_vector(vocab, questions, labels) 
    run_neural_network(context_array, context_label_array, len(vocab), train_size, totalpadlength)
    pretrained_embedding_run_NN(context_array, context_label_array, vocab_size, vocab, train_size)
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
        def lower_repl(match):
            return match.group(1).lower()

        # txt = r"This is a practice tweet :). Let's hope our-system can get it right. \U0001F923 something."
        txt = re.sub('(?:<[^>]+>)', '', txt)# remove html tags
        txt = re.sub('([A-Z][a-z]+)',lower_repl,txt) #lowercase words that start with captial
        tokens = re.split(emoticon_string,txt) #split based on emoji faces first 
        tokensfinal = []
        for i in tokens: #after the emoji tokenizing is done, do basic word tokenizing with what is left
            if not re.match(emoticon_string,i):
                to_add = word_tokenize(i)
                tokensfinal = tokensfinal + to_add
            else:
                tokensfinal.append(i)
        # tokensfinal.insert(0, '_start_') #for use if trying to actually make sentences
        # tokensfinal.append('_end_')
        return tokensfinal

    #initalize variables
    questions = defaultdict()
    labels = defaultdict()
    docs = []
    #laod data and tokenize
    with open('Data/train.csv',encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        rownum = 0
        for row in csv_reader:
            if line_count < train_size and rownum > 0: #skip first header row
                questions[row[0]] = tokenize(row[1])
                labels[row[0]] = int(row[2])
                line_count += 1
            rownum += 1
    
    with open('Data/test.csv',encoding="utf8") as csv_file: 
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        rownum = 0
        for row in csv_reader:
            if line_count < 375806 and rownum > 0:#skip header row 375806
                questions[row[0]] = tokenize(row[1])
                labels[row[0]] = int(0) #doesn't matter really... 
                line_count += 1
            rownum +=1
    
    
    vocab = list(set([item for sublist in questions.values() for item in sublist]))

    print("--- Text Extracted --- %s seconds ---" % (round((time.time() - start_time),2)))   
    return vocab, questions, labels

def get_context_vector(vocab, questions, labels):
    '''
    Construct your n-grams: Create positive n-gram samples by collecting all pairs of adjacent
    tokens. Create 2 negative samples for each positive sample by keeping the first word the same
    as the positive sample, but randomly sampling the rest of the corpus for the second word. The
    second word can be any word in the corpus except for the first word itself. 
    
    This functions takes the docs and tokenized sentences and creates the numpyarrays needed for the neural network.
    --creates 2 fake grams for every real gram 
    '''
    

    word_to_ix = {word: i for i, word in enumerate(vocab)} #index vocabulary

    context_values = [] #array of word index for context 
    for context in questions.values():
        context_values.append([word_to_ix[w] for w in context])
    # print(context_values)
    
    context_labels = [] # list of labels for context
    for label in labels.values():
        context_labels.append([label])

    ids = []
    for entry in questions.keys():
        ids.append(entry)
    
    #convert to numpy array for use in torch  -- padding with index 0 for padding.... Should change to a random word...
    totalpadlength = max(map(len, context_values))
    context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in context_values]) #needed because without badding we are lost 
    context_label_array = np.array(context_labels) 


    print("--- Grams Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return context_array, context_label_array, ids, totalpadlength

def run_neural_network(context_array, context_label_array, vocab_size, train_size, totalpadlength):
    '''
    Create your training and test data: Split your generated samples into training and test sets
    randomly. Keep 20% for testing. Use the rest for training.
    Build and train a feed forward neural network: Build your FFNN with 2 layers (1 hidden layer and
    1 output layer) with hidden vector size 20. Initialize the weights with random numbers.
    Experiment with mean squared error and cross entropy as your loss function. Experiment with
    different hidden vector sizes. Use sigmoid as the activation function and a learning rate of
    0.00001. You must tune any parameters using cross-validation on the training data only. Once
    you have finalized your system, you are ready to evaluate on test data.

    This takes the input vectors and randomly splits it into a training, validation, and test set. Training is performed on 
    feedforward neural net to create a langage model. This is validated and the results of the predictions on the test set is
    provided.
    '''
    
    BATCH_SIZE = 50 # 1000 maxes memory for 8GB GPU -- keep set to 1 to predict all test cases in current implementation

    #randomly split into test and validation sets
    X_train, y_train = context_array[:(train_size)][:], context_label_array[:(train_size)][:]

    X_test, y_test = context_array[(train_size):][:], context_label_array[(train_size):][:]

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

    class NGramLanguageModeler(nn.Module):
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
            super(NGramLanguageModeler, self).__init__()
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

    
    # randomly drawing values from a uniform distribution for weight initialization
    def random_weights(model):
        if type(model) == nn.Linear:
            torch.nn.init.uniform_(model.weight)
            
    #initalize model parameters and variables
    losses = []
    loss_function = nn.BCELoss() #binary cross entropy produced best results
    # Experimenting with MSE Loss
    #loss_function = nn.MSELoss()
    model = NGramLanguageModeler(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE) #.to_fp16() for memory
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...
    model.apply(random_weights)
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
