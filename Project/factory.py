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
    erroranalysis = False
    pretrained_embeddings_status = True

    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    #get data into vectorized format and extract vocab 
    vocab, train_questions, train_labels, test_questions, train_ids, test_ids = get_docs(train_size, readytosubmit) 
    vectorized_data, wordindex, vocab, totalpadlength = get_context_vector(vocab, train_questions, train_labels, test_questions, readytosubmit)
    
    #shows proportions of training set
    unique, cnts = np.unique(vectorized_data['train_context_label_array'], return_counts=True) #get train class sizes
    print(dict(zip(unique, cnts)))

    #setting up embeddings if pretrained embeddings used 
    if pretrained_embeddings_status:
        glove_embedding = build_weights_matrix(vocab, r"kaggle/input/quora-insincere-questions-classification/embeddings/glove.840B.300d\glove.840B.300d.txt", wordindex=wordindex)
        para_embedding = build_weights_matrix(vocab, r"kaggle/input/quora-insincere-questions-classification/embeddings/paragram_300_sl999\paragram_300_sl999.txt", wordindex=wordindex)
        combined_embedding = para_embedding*0.3+glove_embedding*0.7
    else:
        combined_embedding = None

    #run models
    # run_FF(vectorized_data, test_ids, wordindex, len(vocab), totalpadlength, weights_matrix_torch=combined_embedding,
    #         hidden_dim=256, readytosubmit=readytosubmit, erroranalysis=erroranalysis, batch_size=BATCH_SIZE,
    #         learning_rate=0.1, pretrained_embeddings_status=pretrained_embeddings_status)

    # run_RNN(vectorized_data, test_ids, wordindex, len(vocab), totalpadlength, weights_matrix_torch=combined_embedding,
    #         hidden_dim=256, readytosubmit=readytosubmit, erroranalysis=erroranalysis, rnntype="LSTM", bidirectional_status=True,batch_size=BATCH_SIZE,
    #         learning_rate=0.1, pretrained_embeddings_status=pretrained_embeddings_status)

    # run_RNN_CNN(vectorized_data, test_ids, wordindex, len(vocab), totalpadlength, weights_matrix_torch=combined_embedding,
    #         hidden_dim=256, readytosubmit=readytosubmit, erroranalysis=erroranalysis, rnntype="LSTM", bidirectional_status=True,batch_size=BATCH_SIZE,
    #         learning_rate=0.1, pretrained_embeddings_status=pretrained_embeddings_status)

    run_Attention_RNN(vectorized_data, test_ids, wordindex, len(vocab), totalpadlength, weights_matrix_torch=combined_embedding,
        hidden_dim=256, readytosubmit=readytosubmit, erroranalysis=erroranalysis, rnntype="LSTM", bidirectional_status=True,batch_size=BATCH_SIZE,
        learning_rate=0.005, pretrained_embeddings_status=pretrained_embeddings_status)

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

def get_context_vector(vocab, train_questions, train_labels, test_questions, readytosubmit):
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

    if readytosubmit:
        testsize = 0.02
    else:
        test_size = 0.2
    valid_context_array = np.zeros(5)
    valid_context_label_array = np.zeros(5)
    test_context_label_array = np.zeros(len(test_context_array))
    train_context_array, valid_context_array, train_context_label_array, valid_context_label_array = train_test_split(train_context_array, train_context_label_array, 
                                                                                                                        test_size=test_size,  random_state=1234, shuffle=True, 
                                                                                                                        stratify=train_context_label_array)
    arrays_and_labels = defaultdict()
    arrays_and_labels = {"train_context_array":train_context_array,
                        "train_context_label_array":train_context_label_array,
                        "valid_context_array":valid_context_array,
                        "valid_context_label_array":valid_context_label_array,
                        "test_context_array":test_context_array,
                        "test_context_label_array":test_context_label_array}
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
    return arrays_and_labels, ix_to_word, vocab, totalpadlength

def build_weights_matrix(vocab, embedding_file, wordindex):
    """
    used to apply pretrained embeddings to vocabulary
    """
    ps = PorterStemmer()
    lc = LancasterStemmer()
    sb = SnowballStemmer("english")
    print("--- Building Pretrained Embedding Index  --- %s seconds ---" % (round((time.time() - start_time),2)))
    
    embeddings_index = {}
    with open (embedding_file, encoding="utf8", errors='ignore') as f:
        for line in f:
            values = line.split(" ")
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    
    embedding_dim = embeddings_index[word].shape[0]
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

def run_FF(vectorized_data, test_ids, wordindex,  vocablen, totalpadlength=70,weights_matrix_torch=[], hidden_dim=100, readytosubmit=False, 
            erroranalysis=False, batch_size=500, learning_rate=0.1, pretrained_embeddings_status=True):
    '''
    This function uses pretrained embeddings loaded from a file to build an RNN of various types based on the parameters
    bidirectional will make the network bidirectional
    '''
    def format_tensors(vectorized_data, dataset_type, batch_size):
        '''
        helper function to format numpy vectors to the correct type, also determines the batch size for train, valid, and test sets
        based on minibatch size
        '''
        X = torch.from_numpy(vectorized_data[dataset_type+'_context_array'])
        X = X.long()
        y = torch.from_numpy(vectorized_data[dataset_type+'_context_label_array'])
        y = y.long()
        tensordata = data_utils.TensorDataset(X,y)
        loader = data_utils.DataLoader(tensordata, batch_size=batch_size,shuffle=False)
        return loader
    
    #randomly split into test and validation sets
    trainloader = format_tensors(vectorized_data,'train',batch_size)
    validloader = format_tensors(vectorized_data,'valid',batch_size)
    testloader = format_tensors(vectorized_data,'test',batch_size)

    def create_emb_layer(weights_matrix):
        '''
        creates torch embeddings layer from matrix
        '''
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':weights_matrix})
        return emb_layer, embedding_dim
    

    class FeedForward(nn.Module):
        '''
        FF model
        '''
        def __init__(self, hidden_dim, weights_matrix_torch, context_size, vocablen, pre_trained=True):
            super(FeedForward, self).__init__()
            if pre_trained:
                self.embedding, embedding_dim = create_emb_layer(weights_matrix_torch)
            else:
                embedding_dim = 300
                self.embedding = nn.Embedding(vocablen, embedding_dim)
            self.linear1 = nn.Linear(embedding_dim, hidden_dim)
            self.relu = nn.ReLU() 
            self.linear2 = nn.Linear(hidden_dim*context_size, 1)
            
        def forward(self, inputs):
            embeds = self.embedding(inputs) #[batch , context , embed_dim] 
            out = self.linear1(embeds) #[batch,  context, hidden_dim]
            out = self.relu(out) #[batch , context , hidden_dim]
            out = out.contiguous().view(out.shape[0],-1) #[batch , context x hidden_dim]
            yhat = self.linear2(out) #[batch ,1]
            return yhat
            
        
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
        weights.append(1-(flat_train_labels.count(1)/(len(flat_train_labels)))) #proportional to number without tags
        print(weights)
        return weights
    
    weights = class_proportional_weights(vectorized_data['train_context_label_array'])
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    

    #initalize model parameters and variables
    model = FeedForward(hidden_dim, weights_matrix_torch, totalpadlength,vocablen, pretrained_embeddings_status)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #learning rate set to 0.005 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    sig_fn = nn.Sigmoid()
    f1_list = []
    best_f1 = 0 
    print("Start Training --- %s seconds ---" % (round((time.time() - start_time),2)))
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
            # Compute Binary Cross-Entropy
            loss = criterion(yhat, label.float())
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
    
def run_RNN(vectorized_data, test_ids, wordindex, vocablen, totalpadlength=70,weights_matrix_torch=[], hidden_dim=100, readytosubmit=False, 
            erroranalysis=False, rnntype="RNN", bidirectional_status=False, batch_size=500, learning_rate=0.1, pretrained_embeddings_status=True):
    '''
    This function uses pretrained embeddings loaded from a file to build an RNN of various types based on the parameters
    bidirectional will make the network bidirectional
    '''
    def format_tensors(vectorized_data, dataset_type, batch_size):
        '''
        helper function to format numpy vectors to the correct type, also determines the batch size for train, valid, and test sets
        based on minibatch size
        '''
        X = torch.from_numpy(vectorized_data[dataset_type+'_context_array'])
        X = X.long()
        y = torch.from_numpy(vectorized_data[dataset_type+'_context_label_array'])
        y = y.long()
        tensordata = data_utils.TensorDataset(X,y)
        loader = data_utils.DataLoader(tensordata, batch_size=batch_size,shuffle=False)
        return loader
    
    #randomly split into test and validation sets
    trainloader = format_tensors(vectorized_data,'train',batch_size)
    validloader = format_tensors(vectorized_data,'valid',batch_size)
    testloader = format_tensors(vectorized_data,'test',batch_size)

    def create_emb_layer(weights_matrix):
        '''
        creates torch embeddings layer from matrix
        '''
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':weights_matrix})
        return emb_layer, embedding_dim
    

    class RNNmodel(nn.Module):
        '''
        RNN model that can be changed to LSTM or GRU and made bidirectional if needed 
        '''
        def __init__(self, hidden_size, weights_matrix, context_size, vocablen, bidirectional_status=False, rnntype="RNN", pre_trained=True):
            super(RNNmodel, self).__init__()
            if bidirectional_status:
                num_directions = 2
            else:
                num_directions = 1

            if pre_trained:
                self.embedding, embedding_dim = create_emb_layer(weights_matrix_torch)
            else:
                embedding_dim = 300
                self.embedding = nn.Embedding(vocablen, embedding_dim)

            if rnntype=="LSTM":
                print("----Using LSTM-----")
                self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=bidirectional_status)
            elif rnntype=="GRU":
                print("----Using GRU-----")
                self.rnn = nn.GRU(embedding_dim, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=bidirectional_status)
            else:
                print("----Using RNN-----")
                self.rnn = nn.RNN(embedding_dim, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=bidirectional_status)
            self.fc = nn.Linear(hidden_size*num_directions,1)
            
        def forward(self, inputs):
            embeds = self.embedding(inputs)
            out, (ht, ct) = self.rnn(embeds)
            out = torch.mean(out, 1)
            yhat = self.fc(out)
            return yhat
            
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
        weights.append(1-(flat_train_labels.count(1)/(len(flat_train_labels)))) #proportional to number without tags
        return weights
    
    weights = class_proportional_weights(vectorized_data['train_context_label_array'])
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    

    #initalize model parameters and variables
    model = RNNmodel(hidden_dim, weights_matrix_torch, totalpadlength, vocablen, bidirectional_status, rnntype, pretrained_embeddings_status)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #learning rate set to 0.005 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    sig_fn = nn.Sigmoid()
    f1_list = []
    best_f1 = 0 
    print("Start Training --- %s seconds ---" % (round((time.time() - start_time),2)))
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
            # yhat = yhat.view(-1,1)
            # Compute Binary Cross-Entropy
            loss = criterion(yhat, label.float())
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

def run_RNN_CNN(vectorized_data, test_ids, wordindex, vocablen, totalpadlength=70,weights_matrix_torch=[], hidden_dim=100, readytosubmit=False, 
            erroranalysis=False, rnntype="RNN", bidirectional_status=False, batch_size=500, learning_rate=0.1, pretrained_embeddings_status=True):
    '''
    This function uses pretrained embeddings loaded from a file to build an RNN of various types based on the parameters
    bidirectional will make the network bidirectional
    '''
    def format_tensors(vectorized_data, dataset_type, batch_size):
        '''
        helper function to format numpy vectors to the correct type, also determines the batch size for train, valid, and test sets
        based on minibatch size
        '''
        X = torch.from_numpy(vectorized_data[dataset_type+'_context_array'])
        X = X.long()
        y = torch.from_numpy(vectorized_data[dataset_type+'_context_label_array'])
        y = y.long()
        tensordata = data_utils.TensorDataset(X,y)
        loader = data_utils.DataLoader(tensordata, batch_size=batch_size,shuffle=False)
        return loader
    
    #randomly split into test and validation sets
    trainloader = format_tensors(vectorized_data,'train',batch_size)
    validloader = format_tensors(vectorized_data,'valid',batch_size)
    testloader = format_tensors(vectorized_data,'test',batch_size)

    def create_emb_layer(weights_matrix):
        '''
        creates torch embeddings layer from matrix
        '''
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':weights_matrix})
        return emb_layer, embedding_dim
    

    class RNNmodel(nn.Module):
        '''
        RNN model that can be changed to LSTM or GRU and made bidirectional if needed 
        '''
        def __init__(self, hidden_size, weights_matrix, context_size, vocablen, bidirectional_status=False, rnntype="RNN", pre_trained=True):
            super(RNNmodel, self).__init__()
            if bidirectional_status:
                num_directions = 2
            else:
                num_directions = 1

            if pre_trained:
                self.embedding, embedding_dim = create_emb_layer(weights_matrix_torch)
            else:
                embedding_dim = 300
                self.embedding = nn.Embedding(vocablen, embedding_dim)

            if rnntype=="LSTM":
                print("----Using LSTM-----")
                self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=bidirectional_status)
            elif rnntype=="GRU":
                print("----Using GRU-----")
                self.rnn = nn.GRU(embedding_dim, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=bidirectional_status)
            else:
                print("----Using RNN-----")
                self.rnn = nn.RNN(embedding_dim, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=bidirectional_status)
            self.fc = nn.Linear(hidden_size*num_directions,1)
            self.conv = nn.Conv1d(context_size, 64, kernel_size=3)
            self.maxpool = nn.MaxPool1d(2)
            self.fc = nn.Linear(((hidden_size)-1)*64,1)
            
        def forward(self, inputs):
            embeds = self.embedding(inputs) # [batch_size x totalpadlength x embedding_dim]
            out, _ = self.rnn(embeds) # [batch_size x totalpadlength x hidden_dim*num_directions]
            out = self.conv(out)
            out = self.maxpool(out)
            out1 = torch.cat([out[:,:,i] for i in range(out.shape[2])], dim=1) # [batch_size x totalpadlength*hidden_dim*num_directions]
            yhat = self.fc(out1) # [batch_size, 1]
            return yhat
            
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
        weights.append(1-(flat_train_labels.count(1)/(len(flat_train_labels)))) #proportional to number without tags
        return weights
    
    weights = class_proportional_weights(vectorized_data['train_context_label_array'])
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    

    #initalize model parameters and variables
    model = RNNmodel(hidden_dim, weights_matrix_torch, totalpadlength, vocablen, bidirectional_status, rnntype, pretrained_embeddings_status)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #learning rate set to 0.005 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    sig_fn = nn.Sigmoid()
    f1_list = []
    best_f1 = 0 
    print("Start Training --- %s seconds ---" % (round((time.time() - start_time),2)))
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
            # yhat = yhat.view(-1,1)
            # Compute Binary Cross-Entropy
            loss = criterion(yhat, label.float())
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

def run_Attention_RNN(vectorized_data, test_ids, wordindex, vocablen, totalpadlength=70,weights_matrix_torch=[], hidden_dim=100, readytosubmit=False, 
            erroranalysis=False, rnntype="RNN", bidirectional_status=False, batch_size=500, learning_rate=0.1, pretrained_embeddings_status=True):
    '''
    This function uses pretrained embeddings loaded from a file to build an RNN of various types based on the parameters
    bidirectional will make the network bidirectional
    '''
    def format_tensors(vectorized_data, dataset_type, batch_size):
        '''
        helper function to format numpy vectors to the correct type, also determines the batch size for train, valid, and test sets
        based on minibatch size
        '''
        X = torch.from_numpy(vectorized_data[dataset_type+'_context_array'])
        X = X.long()
        y = torch.from_numpy(vectorized_data[dataset_type+'_context_label_array'])
        y = y.long()
        tensordata = data_utils.TensorDataset(X,y)
        loader = data_utils.DataLoader(tensordata, batch_size=batch_size,shuffle=False)
        return loader
    
    #randomly split into test and validation sets
    trainloader = format_tensors(vectorized_data,'train',batch_size)
    validloader = format_tensors(vectorized_data,'valid',batch_size)
    testloader = format_tensors(vectorized_data,'test',batch_size)

    def create_emb_layer(weights_matrix):
        '''
        creates torch embeddings layer from matrix
        '''
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':weights_matrix})
        return emb_layer, embedding_dim
    

    class Attention(nn.Module):
        def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
            super(Attention, self).__init__(**kwargs)
            
            self.supports_masking = True
    
            self.bias = bias
            self.feature_dim = feature_dim
            self.step_dim = step_dim
            self.features_dim = 0
            
            weight = torch.zeros(feature_dim, 1)
            nn.init.kaiming_uniform_(weight)
            self.weight = nn.Parameter(weight)
            
            if bias:
                self.b = nn.Parameter(torch.zeros(step_dim))
            
        def forward(self, x, mask=None):
            feature_dim = self.feature_dim 
            step_dim = self.step_dim
    
            eij = torch.mm(
                x.contiguous().view(-1, feature_dim), 
                self.weight
            ).view(-1, step_dim)
            
            if self.bias:
                eij = eij + self.b
                
            eij = torch.tanh(eij)
            a = torch.exp(eij)
            
            if mask is not None:
                a = a * mask
    
            a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
    
            weighted_input = x * torch.unsqueeze(a, -1)
            return torch.sum(weighted_input, 1)
      
    class Neural_Network(nn.Module):
        def __init__(self, hidden_size, weights_matrix, context_size, vocablen, bidirectional_status=False, rnntype="RNN", pre_trained=True):
            super(Neural_Network, self).__init__()
            if bidirectional_status:
                num_directions = 2
            else:
                num_directions = 1

            if pre_trained:
                self.embedding, embedding_dim = create_emb_layer(weights_matrix)
            else:
                embedding_dim = 300
                self.embedding = nn.Embedding(vocablen, embedding_dim)
            
            self.embedding_dropout = nn.Dropout2d(0.1)
            self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True,
                                batch_first=True)
            self.gru = nn.GRU(hidden_size*2, hidden_size, bidirectional=True,
                              batch_first=True)
            
            self.attention = Attention(hidden_size*2, context_size)
            self.linear = nn.Linear(hidden_size*8, 16)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.fc = nn.Linear(16,1)
            
        def forward(self, inputs):            
            embeds = self.embedding(inputs) # [batch_size x totalpadlength x embedding_dim]
            h_lstm, _ = self.lstm(embeds) # [batch_size x totalpadlength x hidden_dim*num_directions]
            h_gru, _ = self.gru(h_lstm) # [batch_size x totalpadlength x hidden_dim*num_directions]
            h_lstm_attn = self.attention(h_lstm) # [batch_size x hidden_dim*num_directions]
            h_gru_attn = self.attention(h_gru) # [batch_size x hidden_dim*num_directions]
            
            avg_pool = torch.mean(h_gru, 1)
            max_pool, _ = torch.max(h_gru, 1)
            conc = torch.cat((h_lstm_attn, h_gru_attn, avg_pool, max_pool),1) #[batch_size x hidden_size*8]
            conc = self.relu(self.linear(conc)) # [batch_size x 16]
            conc = self.dropout(conc) # [batch_size x hidden_dim_2]
            yhat = self.fc(conc)
            return yhat
            
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
        weights.append(1-(flat_train_labels.count(1)/(len(flat_train_labels)))) #proportional to number without tags
        return weights
    
    weights = class_proportional_weights(vectorized_data['train_context_label_array'])
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    

    #initalize model parameters and variables
    model = Neural_Network(hidden_dim, weights_matrix_torch, totalpadlength, vocablen, bidirectional_status, rnntype, pretrained_embeddings_status)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #learning rate set to 0.005 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    sig_fn = nn.Sigmoid()
    f1_list = []
    best_f1 = 0 
    print("Start Training --- %s seconds ---" % (round((time.time() - start_time),2)))
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
            # yhat = yhat.view(-1,1)
            # Compute Binary Cross-Entropy
            loss = criterion(yhat, label.float())
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


if __name__ == "__main__":
    main()
