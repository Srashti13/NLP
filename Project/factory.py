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
from nltk import word_tokenize, sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer, SnowballStemmer
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
from tqdm import tqdm



# from GPUtil import showUtilization as gpu_usage



start_time = time.time()


def main():
    '''
    The main function. This is used to get/tokenize the documents, create vectors for input into the language model based on
    a number of grams, and input the vectors into the model for training and evaluation.
    '''
    readytosubmit=False
    train_size = 100000 #1306112 is full dataset
    BATCH_SIZE = 500
    erroranalysis = True
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    vocab, train_questions, train_labels, test_questions, train_ids, test_ids = get_docs(train_size, readytosubmit) 
    train_context_array, train_context_label_array, test_context_array, totalpadlength, wordindex, vocab = get_context_vector(vocab, train_questions, train_labels, test_questions)
    unique, cnts = np.unique(train_context_label_array, return_counts=True) #get train class sizes
    print(dict(zip(unique, cnts)))
    weights_matrix_torch = build_weights_matrix(vocab, r"kaggle/input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300.txt", embedding_dim=300, wordindex=wordindex)
    
    
    pretrained_embedding_run_RNN(train_context_array, train_context_label_array,test_context_array, 
                                test_ids, vocab,totalpadlength, wordindex,
                                weights_matrix_torch, hidden_dim=256, readytosubmit=readytosubmit, 
                                erroranalysis=erroranalysis, RNNTYPE="LSTM", bidirectional=True,
                                batch_size=BATCH_SIZE)
    
    
#    run_neural_network(train_context_array, train_context_label_array,test_context_array,test_ids, len(vocab), train_size, totalpadlength,readytosubmit, erroranalysis, wordindex)
#    RNNTYPE = "RNN"
#    run_RNN(train_context_array, train_context_label_array,test_context_array,test_ids, len(vocab), train_size, totalpadlength,readytosubmit,RNNTYPE,erroranalysis, wordindex)
#    RNNTYPE = "GRU"
#    run_RNN(train_context_array, train_context_label_array,test_context_array,test_ids, len(vocab), train_size, totalpadlength,readytosubmit,RNNTYPE,erroranalysis, wordindex)
#    RNNTYPE = "LSTM"
#    run_RNN(train_context_array, train_context_label_array,test_context_array,test_ids, len(vocab), train_size, totalpadlength,readytosubmit,RNNTYPE,erroranalysis, wordindex)
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
    if readytosubmit:
        train = pd.read_csv(r'kaggle/input/quora-insincere-questions-classification/train.csv')
    else:
        train = pd.read_csv(r'kaggle/input/quora-insincere-questions-classification/train.csv',nrows=train_size)
    train_questions = train['question_text']
    train_labels = train['target']
    train_ids = train['qid']
    tqdm.pandas()
    print("----Tokenizing Train Questions----")
    train_questions = train_questions.progress_apply(tokenize)
    
    if readytosubmit:
        test = pd.read_csv(r'kaggle/input/quora-insincere-questions-classification/test.csv')
    else:
        test = pd.read_csv(r'kaggle/input/quora-insincere-questions-classification/test.csv',nrows=10) #doesnt matter
    test_questions = test['question_text']
    test_ids = test['qid']
    tqdm.pandas()
    print("----Tokenizing Test Questions----")
    test_questions = test_questions.progress_apply(tokenize)
    
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
    word_to_ix = {word: i+1 for i, word in enumerate(vocab)} #index vocabulary
    word_to_ix['XXPADXX'] = 0 #set up padding
    vocab.append('XXPADXX')

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

    #used to back convert to words from index
    ix_to_word = {} 
    for key, value in word_to_ix.items(): 
        if value in ix_to_word: 
            ix_to_word[value].append(key) 
            print(value)
            print(key)
            print(ix_to_word[value])
        else: 
            ix_to_word[value]=[key] 

    print("--- Grams Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return train_context_array, train_context_label_array, test_context_array, totalpadlength, ix_to_word, vocab


def build_weights_matrix(vocab, embedding_file, embedding_dim, wordindex):
    """
    used to apply pretrained embeddings to vocabulary
    """
    ps = PorterStemmer()
    lc = LancasterStemmer()
    sb = SnowballStemmer("english")
    print("--- Building Pretrained Embedding Index  --- %s seconds ---" % (round((time.time() - start_time),2)))
    words = []
    embeddings_index = {}
    with open (embedding_file, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            words.append(word)
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, embedding_dim)) 
    words_found = 0
    words_not_found = 0
    # assigning pretrained embeddings
    for i, word in tqdm(wordindex.items()):
        word = "".join(word)
        if embeddings_index.get(word) is not None:
            weights_matrix[i] = embeddings_index[word] #assign the pretrained embedding
            words_found += 1
            continue
        # if the word in the vocab doesn't match anything in the pretrained embedding,
        # we are adjusting the word to see if any adjustment matches a word in the embedding
        adjusted_word = word.lower()
        if embeddings_index.get(adjusted_word) is not None:
            weights_matrix[i] = embeddings_index[adjusted_word] 
            words_found += 1
            continue
        adjusted_word = word.upper()
        if embeddings_index.get(adjusted_word) is not None:
            weights_matrix[i] = embeddings_index[adjusted_word] 
            words_found += 1
            continue
        adjusted_word = word.capitalize()
        if embeddings_index.get(adjusted_word) is not None:
            weights_matrix[i] = embeddings_index[adjusted_word]
            words_found += 1
            continue
        adjusted_word = ps.stem(word)
        if embeddings_index.get(adjusted_word) is not None:
            weights_matrix[i] = embeddings_index[adjusted_word] 
            words_found += 1
            continue
        adjusted_word = lc.stem(word)
        if embeddings_index.get(adjusted_word) is not None:
            weights_matrix[i] = embeddings_index[adjusted_word] 
            words_found += 1
            continue
        adjusted_word = sb.stem(word)
        if embeddings_index.get(adjusted_word) is not None:
            weights_matrix[i] = embeddings_index[adjusted_word] 
            words_found += 1
            continue
        
        # if the word still isn't in the embedding, even after trying all the 
        # adjustments, then we assign it a random normal set of numbers
        weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,)) #randomize out of vocabulary words
        words_not_found += 1
            
    print("{:.2f}% ({}/{}) of the vocabulary were in the pre-trained embedding.".format((words_found/len(vocab))*100,words_found,len(vocab)))
    return torch.from_numpy(weights_matrix)


def run_neural_network(context_array, context_label_array,test_context_array, test_ids, 
vocab_size, train_size, totalpadlength, readytosubmit, erroranalysis, wordindex):
    '''
    regular FeedForward without pretrained embeddings
    '''
    
    BATCH_SIZE = 500 # 1000 maxes memory for 8GB GPU -- keep set to 1 to predict all test cases in current implementation

    #randomly split into test and validation sets
    X_train, y_train = context_array, context_label_array

    X_test, y_test = test_context_array, np.zeros(len(test_context_array))

    if readytosubmit:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.02, 
                                                            random_state=1234, shuffle=True, stratify=y_train)
    else:
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
    EMBEDDING_DIM = 300 # embeddings dimensions
    CONTEXT_SIZE = totalpadlength # total length of padded questions size
    HIDDEN_SIZE = 70 # nodes in hidden layer

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
            print('----Using Feed Forward----')
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
    for epoch in range(5): 
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

    #error analysis if desired
    if erroranalysis:
        model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
        with torch.no_grad():
            contextsfull = []
            predictionsfull = []
            labelsfull = []
            for a, (context, label) in enumerate(validloader):
                for (k, element) in enumerate(context): #per batch
                    contextsfull.append(" ".join(list(itertools.chain.from_iterable([wordindex[x] for x in context[k].tolist()]))))
                    labelsfull.extend(label.int().tolist())
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
                yhat = yhat.view(-1,1)
                predictions = (yhat > 0.5)
                predictionsfull.extend(predictions.int().tolist())
        #print 20 errors 
        printed = 0
        for (i, pred) in enumerate(predictionsfull):
            if pred != labelsfull[i]:
                if printed < 20:
                    print(' '.join([word for word in contextsfull[i].split() if word not in ['XXPADXX']]))
                    print('predicted: %s' % (pred))
                    print('labeled: %s' % (labelsfull[i]))
                    printed +=1

    # Get the accuracy on the test set after training complete -- will have to submit to KAGGLE 
    if readytosubmit:
        model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
        with torch.no_grad():
            predictionsfull = []
            for a, (context, label) in enumerate(testloader):
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
                yhat = yhat.view(-1,1)
                predictions = (yhat > 0.5)
                predictionsfull.extend(predictions.int().tolist())

        #outputs results to csv
        predictionsfinal = []
        for element in predictionsfull:
            predictionsfinal.append(element[0])
        output = pd.DataFrame(np.array([test_ids,predictionsfinal])).transpose()
        output.columns = ['qid', 'prediction']
        print(output.head())
        output.to_csv('submission.csv', index=False)
    return
    
def pretrained_embedding_run_RNN(context_array, context_label_array,test_context_array, 
                                test_ids, vocab,totalpadlength, wordindex,
                                weights_matrix_torch, hidden_dim, readytosubmit=False, 
                                erroranalysis=False, RNNTYPE="RNN", bidirectional=False,
                                batch_size=500):
    '''
    This function uses pretrained embeddings loaded from a file to build an RNN of various types based on the parameters
    bidirectional will make the network bidirectional
    '''

    BATCH_SIZE = batch_size # 1000 maxes memory for 8GB GPU
    CONTEXT_SIZE = totalpadlength # total length of padded questions size
    
    #randomly split into test and validation sets
    X_train, y_train = context_array, context_label_array

    X_test, y_test = test_context_array, np.zeros(len(test_context_array))

    if readytosubmit:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.02, 
                                                            random_state=1234, shuffle=True, stratify=y_train)
    else:
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
    
        
    def create_emb_layer(weights_matrix, non_trainable=False):
        '''
        creates torch embeddings layer from matrix
        '''
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim
    

    class RNNmodel(nn.Module):
        '''
        RNN model that can be changed to LSTM or GRU and made bidirectional if needed 
        '''
        def __init__(self, context_size, hidden_size, 
                     weights_matrix, bidirectional=False, rnntype="RNN", pre_trained=True):
            super(RNNmodel, self).__init__()
            if bidirectional:
                num_directions = 2
            else:
                num_directions = 1
            self.embedding, num_embeddings, embedding_dim = create_emb_layer(
                    weights_matrix)
            if RNNTYPE=="LSTM":
                print("----Using LSTM-----")
                self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=bidirectional)
            elif RNNTYPE=="GRU":
                print("----Using GRU-----")
                self.rnn = nn.GRU(embedding_dim, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=bidirectional)
            else:
                print("----Using RNN-----")
                self.rnn = nn.RNN(embedding_dim, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size*context_size*num_directions,1)
            
        def forward(self, inputs):
            embeds = self.embedding(inputs)
            out, _ = self.rnn(embeds)
            out1 = torch.cat([out[:,:,i] for i in range(out.shape[2])], dim=1)
            yhats = self.fc(out1)
            return yhats
            
        
        
    #initalize model parameters and variables
    losses = []
    def class_proportional_weights(train_labels):
        '''
        helper function to scale weights of classes in loss function based on their sampled proportions
        # This custom loss function is defined to reduce the effect of class imbalance.
        # Since there are so many samples labeled as "O", this allows the RNN to not 
        # be weighted too heavily in that area.
        '''
        weights = []
        flat_train_labels = [item for sublist in train_labels for item in sublist]
        for lab in range(1,2):
            weights.append(1-(flat_train_labels.count(lab)/(len(flat_train_labels)))) #proportional to number without tags
        return weights
    
    weights = class_proportional_weights(context_label_array)
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    

    #initalize model parameters and variables
    model = RNNmodel(CONTEXT_SIZE, hidden_dim, weights_matrix_torch, bidirectional=True, rnntype=RNNTYPE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005) #learning rate set to 0.005 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    sig_fn = nn.Sigmoid()
    f1_list = []
    best_f1 = 0 
    print("Start Training (Pre-trained) --- %s seconds ---" % (round((time.time() - start_time),2)))
    for epoch in range(20): 
        iteration = 0
        running_loss = 0.0 
        for i, (context, label) in enumerate(trainloader):
            # zero out the gradients from the old instance
            optimizer.zero_grad()
            # Run the forward pass and get predicted output
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context) #required dimensions for batching
            yhat = yhat.view(-1,1)
            # Compute Binary Cross-Entropy
            loss = criterion(yhat, label)
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
                yhat = model.forward(context)
                predictions = (sig_fn(yhat) > 0.5)
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

    #error analysis if desired
    if erroranalysis:
        model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
        with torch.no_grad():
            contextsfull = []
            predictionsfull = []
            labelsfull = []
            for a, (context, label) in enumerate(validloader):
                for (k, element) in enumerate(context): #per batch
                    contextsfull.append(" ".join(list(itertools.chain.from_iterable([wordindex[x] for x in context[k].tolist()]))))
                    labelsfull.extend(label.int().tolist())
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context)
                yhat = yhat.view(-1,1)
                predictions = (sig_fn(yhat) > 0.5)
                predictionsfull.extend(predictions.int().tolist())
        #print 20 errors 
        printed = 0
        for (i, pred) in enumerate(predictionsfull):
            if pred != labelsfull[i]:
                if printed < 20:
                    print(' '.join([word for word in contextsfull[i].split() if word not in ['XXPADXX']]))
                    print('predicted: %s' % (pred))
                    print('labeled: %s' % (labelsfull[i]))
                    printed +=1

    if readytosubmit:
        # Get the accuracy on the test set after training complete -- will have to submit to KAGGLE
        model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
        with torch.no_grad():
            predictionsfull = []
            for a, (context, label) in enumerate(testloader):
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context)
                yhat = yhat.view(-1,1)
                predictions = (sig_fn(yhat) > 0.5)
                predictionsfull.extend(predictions.int().tolist())

        #outputs results to csv
        predictionsfinal = []
        for element in predictionsfull:
            predictionsfinal.append(element[0])
        output = pd.DataFrame(np.array([test_ids,predictionsfinal])).transpose()
        output.columns = ['qid', 'prediction']
        print(output.head())
        output.to_csv('submission.csv', index=False)
    return

def run_RNN(context_array, context_label_array,test_context_array, test_ids, vocab_size, 
train_size, totalpadlength, readytosubmit, RNNTYPE,erroranalysis, wordindex):
    '''
    RNN version 
    '''
    
    BATCH_SIZE = 500 # 1000 maxes memory for 8GB GPU -- keep set to 1 to predict all test cases in current implementation

    #randomly split into test and validation sets
    X_train, y_train = context_array, context_label_array

    X_test, y_test = test_context_array, np.zeros(len(test_context_array))

    if readytosubmit:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.02, 
                                                            random_state=1234, shuffle=True, stratify=y_train)
    else:
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
    EMBEDDING_DIM = 300 # embeddings dimensions
    CONTEXT_SIZE = totalpadlength # total length of padded questions size
    HIDDEN_SIZE = 70 # nodes in hidden layer

    class RNNmodel(nn.Module):
        '''
        LSTM 
        '''
        def __init__(self, vocab_size, embedding_dim, context_size, hidden_size, RNNTYPE="LSTM"):
            super(RNNmodel, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim) 
            if RNNTYPE=="LSTM":
                print("----Using LSTM-----")
                self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, batch_first=True)
            elif RNNTYPE=="GRU":
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
            

    #initalize model parameters and variables
    losses = []
    loss_function = nn.BCELoss() #binary cross entropy produced best results
    # Experimenting with MSE Loss
    #loss_function = nn.MSELoss()
    model = RNNmodel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE,RNNTYPE) #.to_fp16() for memory
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) #learning rate set to 0.0001 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    f1_list = []
    best_f1 = 0 
    print("Start Training --- %s seconds ---" % (round((time.time() - start_time),2)))
    for epoch in range(2): 
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
        #early stopping condition
        if epoch+1 >= 5: #start looking to stop after this many epochs
            if f1_list[-1] < min(f1_list[-5:-1]): #if accuracy lower than lowest of last 4 values
                print('...Stopping Early...')
                break

    print("Training Complete --- %s seconds ---" % (round((time.time() - start_time),2)))

    #error analysis if desired
    if erroranalysis:
        model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
        with torch.no_grad():
            contextsfull = []
            predictionsfull = []
            labelsfull = []
            for a, (context, label) in enumerate(validloader):
                for (k, element) in enumerate(context): #per batch
                    contextsfull.append(" ".join(list(itertools.chain.from_iterable([wordindex[x] for x in context[k].tolist()]))))
                    labelsfull.extend(label.int().tolist())
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
                yhat = yhat.view(-1,1)
                predictions = (yhat > 0.5)
                predictionsfull.extend(predictions.int().tolist())
        #print 20 errors 
        printed = 0
        for (i, pred) in enumerate(predictionsfull):
            if pred != labelsfull[i]:
                if printed < 20:
                    print(' '.join([word for word in contextsfull[i].split() if word not in ['XXPADXX']]))
                    print('predicted: %s' % (pred))
                    print('labeled: %s' % (labelsfull[i]))
                    printed +=1

    if readytosubmit:
        # Get the accuracy on the test set after training complete -- will have to submit to KAGGLE 
        model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
        with torch.no_grad():
            total = 0
            num_correct = 0
            predictionsfull = []
            for a, (context, label) in enumerate(testloader):
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
                yhat = yhat.view(-1,1)
                predictions = (yhat > 0.5)
                predictionsfull.extend(predictions.int().tolist())

        #outputs results to csv
        predictionsfinal = []
        for element in predictionsfull:
            predictionsfinal.append(element[0])
        output = pd.DataFrame(np.array([test_ids,predictionsfinal])).transpose()
        output.columns = ['qid', 'prediction']
        print(output.head())
        output.to_csv('submission.csv', index=False)
    return

if __name__ == "__main__":
    main()
