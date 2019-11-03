"""
AIT726 HW 3 Due 11/07/2019
Named Entity Recognition using different types of recurrent neural networks.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW3.py 
"""
#%%

import re
import os
from collections import defaultdict
import itertools
import numpy as np
import time
import torch.utils.data as data_utils
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn 
import gc
from sklearn.metrics import accuracy_score
from conlleval import evaluate_conll_file
import math

start_time = time.time()

def main():
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    train_vocab, train_sentences, totalpadlength = get_sentences(r"conll2003\train.txt")
    valid_vocab, valid_sentences, _ = get_sentences(r"conll2003\valid.txt")
    test_vocab, test_sentences, _ = get_sentences(r"conll2003\test.txt")
    vocab = dict_combination(train_vocab, valid_vocab, test_vocab)
    vectorized_data, wordindex, labelindex = get_context_vectors(vocab, train_sentences, valid_sentences, test_sentences)
    EMBEDDING_DIM=300
    HIDDEN_SIZE=256
    weights_matrix_torch = build_weights_matrix(vocab, "GoogleNews-vectors-negative300.txt", EMBEDDING_DIM)
    run_RNN(vectorized_data, vocab, totalpadlength, wordindex, labelindex, 
            weights_matrix_torch, HIDDEN_SIZE, bidirectional=False, RNNTYPE="RNN")
    return

def dict_combination(dictone,dicttwo,dictthree):
    '''
    Helper function to combine three dictionaries into one and keep the unique set of values for all keys
    '''
    combined = defaultdict()
    ds = [dictone, dicttwo,dictthree]
    for k in list(set(list(ds[0].keys()) + list(ds[1].keys())+list(ds[2].keys()))):
        value = list(d[k] for d in ds)
        combined[k] = list(set([item for sublist in value for item in sublist]))
    return combined

def get_sentences(docs):

    # tokenizing: lowercasing all words that have some, but not all, uppercase
    def lower_repl(match):
        return match.group().lower()
    def lowercase_text(txt):
        txt = re.sub('([A-Z]+[a-z]+)',lower_repl,txt) #lowercase words that start with captial    
        return txt   


    vocab = defaultdict(list)
    doc = []
    with open(docs) as f:
        for word in f.read().splitlines():
            a = word.split(" ")
            if len(a)>1:
                vocab[lowercase_text(a[0])].append(a[3])
                doc.append([lowercase_text(a[0]),a[3]])
            else: 
                doc.append(a[0])
    doc.insert(0,'')

    # retaining the unique tags for each vocab word
    for k,v in vocab.items():
        vocab[k] = (list(set(v)))

    # getting the indices of the end of each sentence
    sentence_ends = []
    for i, word in enumerate(doc):
        if not word:
            sentence_ends.append(i)
    sentence_ends.append(len(doc)-1)
    # creating a list of all the sentences 
    sentences = []
    for i in range(len(sentence_ends)-1):
        sentences.append(doc[sentence_ends[i]+1:sentence_ends[i+1]])
        
    # getting the longest sentence
    max(sentences, key=len)
    # getting the length of the longest sentence
    max_sent_len = len(max(sentences, key=len))

    ## padding all of the sentences to make them length 113
    for sentence in sentences:
        sentence.extend(['0','<pad>'] for i in range(max_sent_len-len(sentence)))
    # This is the code to read the embeddings
    vocab['0'] = '<pad>'
    print("--- Text Extracted --- %s seconds ---" % (round((time.time() - start_time),2)))
    return vocab, sentences, max_sent_len


def get_context_vectors(vocab, train_sentences, valid_sentences, test_sentences):
    '''
    convert to numpy vectors 
    '''
    word_to_ix = {word: i for i, word in enumerate(vocab)} #index vocabulary
    labels_to_ix = {"<pad>":0, "O":1, "B-ORG":2, "B-PER":3, "B-LOC":4, "B-MISC":5, "I-ORG":6, "I-PER":7, "I-LOC":8, "I-MISC":9}

    train_context_values = [] #array of word index for context
    train_label_values = [] 
    for sentence in train_sentences:
        train_context_values.append([word_to_ix[w[0]] for w in sentence])
        train_label_values.append([labels_to_ix[w[1]] for w in sentence])

    valid_context_values = [] #array of word index for context
    valid_label_values = [] 
    for sentence in valid_sentences:
        valid_context_values.append([word_to_ix[w[0]] for w in sentence])
        valid_label_values.append([labels_to_ix[w[1]] for w in sentence])

    test_context_values = [] #array of word index for context
    test_label_values = [] 
    for sentence in test_sentences:
        test_context_values.append([word_to_ix[w[0]] for w in sentence])
        test_label_values.append([labels_to_ix[w[1]] for w in sentence])
        
    #convert to numpy array for use in torch  -- padding with index 0 for padding.... Should change to a random word...
    train_context_array = np.array(train_context_values)
    valid_context_array = np.array(valid_context_values) 
    test_context_array = np.array(test_context_values)
    train_context_label_array = np.array(train_label_values)
    valid_context_label_array = np.array(valid_label_values)
    test_context_label_array = np.array(test_label_values) 
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

    #used to back convert to words from index
    ix_to_label = {} 
    for key, value in labels_to_ix.items(): 
        if value in ix_to_label: 
            ix_to_label[value].append(key) 
            print(value)
            print(key)
            print(ix_to_label[value])
        else: 
            ix_to_label[value]=[key] 

    print("--- Arrays Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return arrays_and_labels, ix_to_word, ix_to_label

def format_tensors(vectorized_data, dataset_type,num_mini_batches):
    X = torch.from_numpy(vectorized_data[dataset_type+'_context_array'])
    X = X.long()
    batch_size = math.ceil(X.size(0)/num_mini_batches) # 200 mini-batches per epoch
    y = torch.from_numpy(vectorized_data[dataset_type+'_context_label_array'])
    y = y.long()
    tensordata = data_utils.TensorDataset(X,y)
    loader = data_utils.DataLoader(tensordata, batch_size=batch_size,shuffle=False)
    return loader

def build_weights_matrix(vocab, embedding_file, embedding_dim):
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
    weights_matrix = np.zeros((matrix_len, embedding_dim)) # 200 is depth of embedding matrix
    words_found = 0
    words_not_found = 0
    for i, word in enumerate(vocab):
        try:
            weights_matrix[i] = embeddings_index[word]
            words_found += 1
        
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,)) #randomize out of vocabulary words
            words_not_found += 1
            
    print("{:.2f}% ({}/{}) of the vocabulary were in the pre-trained embedding.".format((words_found/len(vocab))*100,words_found,len(vocab)))
    return torch.from_numpy(weights_matrix)

def run_RNN(vectorized_data, vocab, totalpadlength, wordindex, labelindex, 
            weights_matrix_torch,hidden_size,bidirectional=False,RNNTYPE="RNN"):
    '''
    This function is the same as run_neural_network except it uses pretrained embeddings loaded from a file
    '''
    # building data loaders
    NUM_MINI_BATCHES = 2000
    trainloader = format_tensors(vectorized_data,'train',NUM_MINI_BATCHES)
    validloader = format_tensors(vectorized_data,'valid',NUM_MINI_BATCHES)
    testloader = format_tensors(vectorized_data,'test',NUM_MINI_BATCHES)

    EMBEDDING_DIM = 300 # embeddings dimensions
    CONTEXT_SIZE = totalpadlength #sentence size
    
    def create_emb_layer(weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim
    
    ## EDIT BELOW!-------------------------------------------------------------------------
    if not bidirectional:
        class RNNmodel(nn.Module):
            def __init__(self, weights_matrix, context_size):
                super(RNNmodel, self).__init__()
                self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
                if RNNTYPE=="LSTM":
                    print("----Using LSTM-----")
                    self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, batch_first=True)
                elif RNNTYPE=="GRU":
                    print("----Using GRU-----")
                    self.rnn = nn.GRU(embedding_dim, hidden_size=hidden_size, batch_first=True)
                else:
                    print("----Using RNN-----")
                    self.rnn = nn.RNN(embedding_dim, hidden_size=hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size,10)
                
            def forward(self, inputs, context_size, embedding_dim):
                # print(inputs.shape) # dim: batch_size x batch_max_len
                embeds = self.embedding(inputs) # dim: batch_size x batch_max_len x embedding_dim
                # print(embeds.shape)
                out, _ = self.rnn(embeds) # dim: batch_size x batch_max_len x lstm_hidden_dim 
                # print(out.shape)
                out = out.contiguous().view(-1, out.shape[2]) # dim: batch_size*batch_max_len x lstm_hidden_dim
                # print(out.shape)
                yhats = self.fc(out) # dim: batch_size*batch_max_len x num_tags                       #https://cs230-stanford.github.io/pytorch-nlp.html
                return yhats 
    else:
        class RNNmodel(nn.Module):
            def __init__(self, weights_matrix, context_size):
                # bidirectional LSTM
                super(RNNmodel, self).__init__()
                print('----Using LSTM (Pre-trained Embeddings)----')
                self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
                if RNNTYPE=="LSTM":
                    print("----Using LSTM-----")
                    self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, 
                                       batch_first=True,bidirectional=True)
                elif RNNTYPE=="GRU":
                    print("----Using GRU-----")
                    self.rnn = nn.GRU(embedding_dim, hidden_size=hidden_size, 
                                      batch_first=True,bidirectional=True)
                else:
                    print("----Using RNN-----")
                    self.rnn = nn.RNN(embedding_dim, hidden_size=hidden_size, 
                                      batch_first=True,bidirectional=True)
                
                self.fc = nn.Linear(hidden_size*2,10)
                
            def forward(self, inputs, context_size, embedding_dim):
                # print(inputs.shape) # dim: batch_size x batch_max_len
                embeds = self.embedding(inputs) # dim: batch_size x batch_max_len x embedding_dim
                # print(embeds.shape)
                out, _ = self.rnn(embeds) # dim: batch_size x batch_max_len x lstm_hidden_dim 
                combined_bi = torch.cat((out[:,:,:hidden_size],out[:,:,hidden_size:]),dim=-1)
                # print(out.shape)
                out = combined_bi.contiguous().view(-1, combined_bi.shape[2]) # dim: batch_size*batch_max_len x lstm_hidden_dim
                # print(out.shape)
                yhats = self.fc(out) # dim: batch_size*batch_max_len x num_tags                       #https://cs230-stanford.github.io/pytorch-nlp.html
                return yhats 
    

    #initalize model parameters and variables
    losses = []
    def class_proportional_weights(train_labels):
        '''
        helper function to scale weights of classes in loss function based on their sampled proportions
        '''
        weights = []
        flat_train_labels = [item for sublist in train_labels for item in sublist]
        for lab in range(1,10):
            weights.append(1-(flat_train_labels.count(lab)/(len(flat_train_labels)-flat_train_labels.count(0)))) #proportional to number without tags
        weights.insert(0,0) #zero padding values weight
        return weights
    weights = class_proportional_weights(vectorized_data['train_context_label_array'].tolist()) #zero out pads and reduce weights given to "O" objects in loss function
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = RNNmodel(weights_matrix_torch, CONTEXT_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01) #learning rate set to 0.0001 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    metric_list = []
    best_metric = 0 
    print("Start Training (Pre-trained) --- %s seconds ---" % (round((time.time() - start_time),2)))
    for epoch in range(5): 
        iteration = 0
        running_loss = 0.0 
        for i, (context, label) in enumerate(trainloader):
            # zero out the gradients from the old instance
            optimizer.zero_grad()
            # Run the forward pass and get predicted output
            label = label.contiguous().view(-1) # convert to length batch_size*seq_len
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM) #required dimensions for batching
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
                label = label.contiguous().view(-1) # convert to length batch_size*seq_len
                context = context.to(device)
                label = label.to(device)
                yhats = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
                index = yhats.max(1)[1] #index position of max value
                prediction = index.int().tolist()
                predictionsfull.extend(prediction)
                labelsfull.extend(label.int().tolist())
                del context, label, prediction #memory
            gc.collect()#memory
            torch.cuda.empty_cache()#memory
            # print('\n')
            # gpu_usage()

            # remove pads and "O" and do acc calculation:
            padindicies = [i for i, x in enumerate(labelsfull) if x == 0 or x==1] 
            for index in sorted(padindicies, reverse=True):
                del labelsfull[index]
                del predictionsfull[index]
            metricscore = accuracy_score(labelsfull,predictionsfull) #not sure if they are using macro or micro in competition
            metric_list.append(metricscore)
        print('--- Epoch: {} | Validation Accuracy (non-O): {} ---'.format(epoch+1, metric_list[-1])) 

        if metric_list[-1] > best_metric: #save if it improves validation accuracy 
            best_metric = metric_list[-1]
            bestmodelparams = torch.save(model.state_dict(), 'train_valid_best.pth') #save best model
        #early stopping condition
        if epoch+1 >= 5: #start looking to stop after this many epochs
            if metric_list[-1] < min(metric_list[-5:-1]): #if accuracy lower than lowest of last 4 values
                print('...Stopping Early...')
                break

    print("Training Complete --- %s seconds ---" % (round((time.time() - start_time),2)))
        # Get the accuracy on the validation set for each epoch
    model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
    with torch.no_grad():
        predictionsfull = []
        labelsfull = []
        contextfull = []
        for a, (context, label) in enumerate(testloader):
            label = label.contiguous().view(-1) # convert to length batch_size*seq_len
            labelsfull.extend(label.int().tolist()) #saving for pad removal and pack conversion
            contextfull.extend(context.int().tolist())
            context = context.to(device)
            label = label.to(device)
            yhats = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
            index = yhats.max(1)[1] #index position of max value
            prediction = index.int().tolist()
            predictionsfull.extend(prediction) #saving for pad removal and pack conversion
            del context, label, prediction #memory
        gc.collect()#memory
        torch.cuda.empty_cache()#memory
        # print('\n')
        # gpu_usage()

        #converting to flat list
        contextfull = [item for sublist in contextfull for item in sublist]
        print("--- Removing Pads and Finding Test Accuracy --- %s seconds ---" % (round((time.time() - start_time),2)))
        #remove pads and do acc calculation:
        padindicies = [i for i, x in enumerate(labelsfull) if x == 0]
        for index in sorted(padindicies, reverse=True):
            del labelsfull[index]
            del predictionsfull[index]
            del contextfull[index]
        metricscore = accuracy_score(labelsfull,predictionsfull) #not sure if they are using macro or micro in competition
    print('--- Test Accuracy: {} ---'.format(metricscore))
    print("--- Formatting Results for conlleval.py Official Evaluation --- %s seconds ---" % (round((time.time() - start_time),2)))
    formattedcontexts = []
    formattedlabels = []
    formattedpredictions = []
    for element in labelsfull: #convert to real words and labels
        formattedlabels.extend(labelindex[element])
    for element in predictionsfull:
        if element == 0:
            element = 1 #remove stray <pad> predictions
        formattedpredictions.extend(labelindex[element])
    for element in contextfull:
        formattedcontexts.extend(wordindex[element])
    #write to file
    fname = 'LSTMresults.txt'
    if os.path.exists(fname):
        os.remove(fname)
    f = open(fname,'w')
    for (i,element) in enumerate(labelsfull):
        f.write(formattedcontexts[i] + ' ' + formattedlabels[i] + ' ' + formattedpredictions[i] + '\n')
    f.close()
    evaluate_conll_file(open(fname,'r'))

if __name__ == "__main__":
    main()
