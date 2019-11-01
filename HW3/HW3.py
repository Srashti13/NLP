"""
AIT726 HW 3 Due 11/07/2019
Named Entity Recognition using different types of recurrent neural networks.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW3.py 
"""
#%%

import re
from collections import defaultdict
import numpy as np
import time
import torch.utils.data as data_utils
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn 
import gc
from sklearn.metrics import accuracy_score



start_time = time.time()

def main():
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    train_vocab, train_sentences, totalpadlength = get_sentences(r"conll2003\train.txt")
    valid_vocab, valid_sentences, _ = get_sentences(r"conll2003\valid.txt")
    test_vocab, test_sentences, _ = get_sentences(r"conll2003\test.txt")
    vocab = dict_combination(train_vocab, valid_vocab, test_vocab)
    vectorized_data = get_context_vectors(vocab, train_sentences, valid_sentences, test_sentences)
    run_RNN(vectorized_data, vocab, totalpadlength)
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

#%%
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
    #%% creating a list of all the sentences 
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

    print("--- Arrays Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return arrays_and_labels

def run_RNN(vectorized_data, vocab, totalpadlength):
    '''
    This function is the same as run_neural_network except it uses pretrained embeddings loaded from a file
    '''
    BATCH_SIZE = 500 # 1000 maxes memory for 8GB GPU

    #set datatypes 
    X_train = torch.from_numpy(vectorized_data['train_context_array'])
    X_train = X_train.long()
    y_train = torch.from_numpy(vectorized_data['train_context_label_array'])
    y_train = y_train.long()
    X_valid = torch.from_numpy(vectorized_data['valid_context_array'])
    X_valid = X_valid.long()
    y_valid = torch.from_numpy(vectorized_data['valid_context_label_array'])
    y_valid = y_valid.long()
    X_test = torch.from_numpy(vectorized_data['test_context_array'])
    X_test = X_test.long()
    y_test = torch.from_numpy(vectorized_data['test_context_label_array'])
    y_test = y_test.long()
    
    #create datsets for loading into models
    train = data_utils.TensorDataset(X_train, y_train)
    valid = data_utils.TensorDataset(X_valid, y_valid)
    test = data_utils.TensorDataset(X_test, y_test)
    trainloader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    validloader = data_utils.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    testloader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    
    print("--- Building Pretrained Embedding Index  --- %s seconds ---" % (round((time.time() - start_time),2)))
    EMBEDDING_DIM = 300 # embeddings dimensions
    CONTEXT_SIZE = totalpadlength #sentence size
    
    # getting embeddings from the file
    # from gensim.models.keyedvectors import KeyedVectors
    # model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)
    # model = KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True, limit=2000)
    EMBEDDING_FILE = "GoogleNews-vectors-negative300.txt"
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
    
    weights_matrix_torch = torch.from_numpy(weights_matrix)
    
    def create_emb_layer(weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim
    
    ## EDIT BELOW!-------------------------------------------------------------------------
    class RNNmodel(nn.Module):
        def __init__(self, weights_matrix, context_size):
            super(RNNmodel, self).__init__()
            print('----Using LSTM (Pre-trained Embeddings)----')
            self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
            self.rnn = nn.LSTM(embedding_dim,120, batch_first=True)
            self.fc = nn.Linear(120,10)
            self.act = nn.Softmax(dim=1) #CrossEntropyLoss takes care of this
            
        def forward(self, inputs, context_size, embedding_dim):
            # print(inputs.shape) # dim: batch_size x batch_max_len
            embeds = self.embedding(inputs) # dim: batch_size x batch_max_len x embedding_dim
            # print(embeds.shape)
            out, _ = self.rnn(embeds) # dim: batch_size x batch_max_len x lstm_hidden_dim 
            # print(out.shape)
            out = out.contiguous().view(-1, out.shape[2]) # dim: batch_size*batch_max_len x lstm_hidden_dim
            # print(out.shape)
            out = self.fc(out) # dim: batch_size*batch_max_len x num_tags                       #https://cs230-stanford.github.io/pytorch-nlp.html
            yhats = self.act(out)
            # print(yhats.shape)
            # print(yhats[2])
            return yhats 

    #initalize model parameters and variables
    losses = []
    def loss_fn(outputs, labels):  #custom loss function needed b/c don't want to test on pads # https://cs230-stanford.github.io/pytorch-nlp.html
        # reshape labels to give a flat vector of length batch_size*seq_len
        
        # mask out 'PAD' tokens
        mask = (labels >= 0).float()
        # the number of tokens is the sum of elements in mask
        num_tokens = int(torch.sum(mask).item())
        
        # pick the values corresponding to labels and multiply by mask
        outputs = outputs[range(outputs.shape[0]), labels]*mask
        
        # cross entropy loss for all non 'PAD' tokens
        return -torch.sum(outputs)/num_tokens

    model = RNNmodel(weights_matrix_torch, CONTEXT_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01) #learning rate set to 0.0001 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    metric_list = []
    best_metric = 0 
    print("Start Training (Pre-trained) --- %s seconds ---" % (round((time.time() - start_time),2)))
    for epoch in range(100): 
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
            loss = loss_fn(yhat, label)
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
                # print(yhats.shape)
                # print(yhats[1])
                index = yhats.max(1)[1] #index position of max value
                prediction = index.int().tolist()
                # print([prediction])
                # print('---')
                # print([label.int().tolist()])
                predictionsfull.extend(prediction)
                labelsfull.extend(label.int().tolist())
                del context, label, prediction #memory
            gc.collect()#memory
            torch.cuda.empty_cache()#memory
            # print('\n')
            # gpu_usage()
            # print(labelsfull)
            # print(predictionsfull)
            metricscore = accuracy_score(labelsfull,predictionsfull) #not sure if they are using macro or micro in competition
            metric_list.append(metricscore)
        print('--- Epoch: {} | Validation Accuracy: {} ---'.format(epoch+1, metric_list[-1])) 

        if metric_list[-1] > best_metric: #save if it improves validation accuracy 
            best_metric = metric_list[-1]
            bestmodelparams = torch.save(model.state_dict(), 'train_valid_best.pth') #save best model
        #early stopping condition
        if epoch+1 >= 5: #start looking to stop after this many epochs
            if metric_list[-1] < min(metric_list[-5:-1]): #if accuracy lower than lowest of last 4 values
                print('...Stopping Early...')
                break

    print("Training Complete --- %s seconds ---" % (round((time.time() - start_time),2)))

if __name__ == "__main__":
    main()