"""
AIT726 HW 2 Due 10/10/2019
Sentiment classificaiton using Naive Bayes and Logistic Regression on a dataset of 25000 training and 25000 testing tweets.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW2.py 
i. main - runs all of the functions
    i. get_docs - tokenizes all tweets, returns a list of tokenized sentences and a list of all tokens
    ii. get_ngrams_vector - creates the context and label numpy arrays from the tokenized sentences
    iii. run_neural_network - splits numpy arrays into train,validation, and test sets. Runs on NN. Outputs accuracy on test set.
    iv. pretrained_embedding_run_NN - same function as iii, except pretrained embedding vectors can be used
"""
import os
import re
import time
import numpy as np
import itertools
from nltk.util import ngrams
from nltk import word_tokenize
from nltk import sent_tokenize
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from statistics import mean
import random
import torch.utils.data as data_utils
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn 


#emoji regex -- used for emoji tokenization
start_time = time.time()
emoticon_string = r"(:\)|:-\)|:\(|:-\(|;\);-\)|:-O|8-|:P|:D|:\||:S|:\$|:@|8o\||\+o\(|\(H\)|\(C\)|\(\?\))"
#https://www.regexpal.com/96995

def main():
    '''
    The main function. This is used to get/tokenize the documents, create vectors for input into the language model based on
    a number of grams, and input the vectors into the model for training and evaluation.
    '''
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    docs, sentences = get_docs() 
    ngram_array, ngram_label_array, vocab_size, vocab = get_ngrams_vector(docs,sentences) 
    run_neural_network(ngram_array, ngram_label_array, vocab_size)
    pretrained_embedding_run_NN(ngram_array, ngram_label_array, vocab_size, vocab)
    return


def get_docs():

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
    alltext = ''
    docs = []
    sentences = []
    #create megadocument of all tweets
    for f in os.listdir('LanguageModelingData'):
        tweet = open(os.path.join('LanguageModelingData',f),encoding="utf8").read()
        alltext = alltext + ' ' + tweet #all tweet text joined
    
    sentences = sent_tokenize(alltext) #divide text into sentences
    token_sentences = [tokenize(sentence) for sentence in sentences] #tokenize each sentence
    #get all of the tokens in the tokenized sentences
    for sentence in token_sentences:
        docs.extend(sentence)
    print("--- Text Extracted --- %s seconds ---" % (round((time.time() - start_time),2)))   
    return docs, token_sentences 


def get_ngrams_vector(docs, sentences):
    '''
    Construct your n-grams: Create positive n-gram samples by collecting all pairs of adjacent
    tokens. Create 2 negative samples for each positive sample by keeping the first word the same
    as the positive sample, but randomly sampling the rest of the corpus for the second word. The
    second word can be any word in the corpus except for the first word itself. 
    This functions takes the docs and tokenized sentences and creates the numpyarrays needed for the neural network.
    --creates 2 fake grams for every real gram 
    '''
    
    ## creating the ngrams from each sentence, so that ngrams from the end
    # of one sentence aren't combined with the beginning of another sentence
    ngram_list = []
    for sentence in sentences:
        for gram in ngrams(sentence, 2):
            ngram_list.append(gram)
    
    # dictionary with the key being the first term in the bigram and the values
    # being the second terms
    ngram_dict = defaultdict(list)
    for gram in ngram_list:
        ngram_dict[gram[0]].append(gram[1]) 
    
    # creating fake ngrams by ensuring that the value is not a duplicate of the
    # key and is also not one of the values for a correct bigram
    fakegrams = []     
    for gram in ngram_list:
        # get two fake ngrams for each correct one
        for _ in range(2): 
            random_word = random.choice(docs)
            while random_word == gram[0] or random_word in ngram_dict[gram[0]]: #pick words until not duplicate word and not correct bigram
                random_word = random.choice(docs)
            fakegrams.append((gram[0], random_word))
    
    #assigning labels for both real and fake grams 
    gramvec, labels = [], [] 
    for element in ngram_list:
        gramvec.append([element[0],element[1]])
        labels.append([1])
    for element in fakegrams:
        gramvec.append([element[0],element[1]])
        labels.append([0])

    vocab = set(docs) #set vocab as unique words
    word_to_ix = {word: i for i, word in enumerate(vocab)} #index vocabulary
    
    ngramlabeled = [[gram] + label for gram, label in zip(gramvec,labels)] #put them together into a list of lists to be iterated

    ngram_values = [] #array of word index for ngrams 
    for context, label in ngramlabeled:
        ngram_values.append([word_to_ix[w] for w in context])
    
    ngram_labels = [] # list of labels for ngrams
    for context, label in ngramlabeled:
        ngram_labels.append([label])
    
    #convert to numpy array for use in torch
    ngram_array = np.array(ngram_values)
    ngram_label_array = np.array(ngram_labels) 
    
    print("--- Grams Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return ngram_array, ngram_label_array, len(vocab), vocab


def run_neural_network(ngram_array, ngram_label_array, vocab_size):
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
    
    BATCH_SIZE = 100 # 1000 maxes memory for 8GB GPU -- keep set to 1 to predict all test cases in current implementation

    #randomly split into test and validation sets
    X_train, X_test, y_train, y_test = train_test_split(ngram_array, ngram_label_array, test_size=0.2, 
                                                       random_state=1234, shuffle=True, stratify=ngram_label_array)
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
    CONTEXT_SIZE = 2 #bigram model
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
    model = NGramLanguageModeler(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...
    model.apply(random_weights)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001) #learning rate set to 0.0001 to converse faster -- change to 0.00001 if desired
    yhat_list = []
    context_list = []
    labels = []
    
    accuracy_list = []
    best_accuracy = 0 
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
            yhat_list.append(yhat)
            context_list.append(context)

            # Compute Binary Cross-Entropy
            labels.append(label)
            loss = loss_function(yhat, label)
    
            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
            iteration += 1
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            running_loss += loss.item()
        losses.append(loss.item())

    # Get the accuracy on the validation set for each epoch
        with torch.no_grad():
            total = 0
            num_correct = 0
            for a, (context, label) in enumerate(validloader):
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
                yhat = yhat.view(-1,1)
                predictions = (yhat > 0.5)
                total += label.nelement()
                num_correct += torch.sum(torch.eq(predictions, label.bool())).item()
            accuracy_list.append(num_correct/total*100) #add accuracy to running epoch list 
        print('--- Epoch: {} | Validation Accuracy: {} ---'.format(epoch+1, accuracy_list[-1])) 

        if accuracy_list[-1] > best_accuracy: #save if it improves validation accuracy 
            best_accuracy = accuracy_list[-1]
            bestmodelparams = torch.save(model.state_dict(), 'train_valid_best.pth') #save best model
        #early stopping condition
        if epoch+1 >= 5: #start looking to stop after this many epochs
            if accuracy_list[-1] < min(accuracy_list[-5:-1]): #if accuracy lower than lowest of last 4 values
                print('...Stopping Early...')
                break

    print("Training Complete --- %s seconds ---" % (round((time.time() - start_time),2)))
    # Get the accuracy on the test set after training complete
    model.load_state_dict(torch.load('train_valid_best.pth')) #load best model
    with torch.no_grad():
        total = 0
        num_correct = 0
        for a, (context, label) in enumerate(testloader):
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
            yhat = yhat.view(-1,1)
            predictions = (yhat > 0.5)
            total += label.nelement()
            num_correct += torch.sum(torch.eq(predictions, label.bool())).item()
        accuracy = num_correct/total*100
        print('Test Accuracy: {} %'.format(round(accuracy,5)))
    return


def pretrained_embedding_run_NN(ngram_array, ngram_label_array, vocab_size, vocab):
    '''
    This function is the same as run_neural_network except it uses pretrained embeddings loaded from a file
    '''
    BATCH_SIZE = 500 # 1000 maxes memory for 8GB GPU

    #randomly split into test and validation sets
    X_train, X_test, y_train, y_test = train_test_split(ngram_array, ngram_label_array, test_size=0.2, 
                                                       random_state=1234, shuffle=True, stratify=ngram_label_array)
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
    
    
    EMBEDDING_DIM = 200 # embeddings dimensions
    CONTEXT_SIZE = 2 #bigram model
    
    # getting embeddings from the file
    EMBEDDING_FILE = "glove.6B.200d.txt"
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
    
    print("{:.2f}% ({}/{}) of the vocabulary were in the pre-trained embedding.".format(words_found/vocab_size,words_found,vocab_size))
    
    weights_matrix_torch = torch.from_numpy(weights_matrix)
    
    def create_emb_layer(weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim
    
    class NGramLanguageModeler(nn.Module):
        def __init__(self, weights_matrix, context_size):
            super(NGramLanguageModeler, self).__init__()
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
    model = NGramLanguageModeler(weights_matrix_torch, CONTEXT_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...
    # The random weight method isn't as effective as the default pytorch method
    #model.apply(random_weights)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001) #learning rate set to 0.0001 to converse faster -- change to 0.00001 if desired
    yhat_list = []
    context_list = []
    labels = []

    best_accuracy = 0
    accuracy_list = []
    print("Start Training (Pre-Trained Embeddings) --- %s seconds ---" % (round((time.time() - start_time),2)))
    for epoch in range(50): #number of epochs
        iteration = 0
        running_loss = 0.0 
        for i, (context, label) in enumerate(trainloader):
            # zero out the gradients from the old instance
            optimizer.zero_grad()
            # Run the forward pass and get predicted output
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
            yhat = yhat.view(-1,1)
            yhat_list.append(yhat)
            context_list.append(context)

            # Compute Binary Cross-Entropy
            labels.append(label)
            loss = loss_function(yhat, label)
    
            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
            iteration += 1
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            running_loss += loss.item()
        losses.append(loss.item())

    # Get the accuracy on the validation set for each epoch
        with torch.no_grad():
            total = 0
            num_correct = 0
            for a, (context, label) in enumerate(validloader):
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
                yhat = yhat.view(-1,1)
                predictions = (yhat > 0.5)
                total += label.nelement()
                num_correct += torch.sum(torch.eq(predictions, label.bool())).item()
            accuracy_list.append(num_correct/total*100) #add accuracy to running epoch list 
        print('--- Epoch: {} | Validation Accuracy: {} ---'.format(epoch+1, accuracy_list[-1])) 

        if accuracy_list[-1] > best_accuracy: #save if it improves validation accuracy 
            best_accuracy = accuracy_list[-1]
            bestmodelparams = torch.save(model.state_dict(), 'train_valid_best_pretrained.pth') #save best model

        #early stopping condition
        if epoch+1 >= 5: #start looking to stop after this many epochs
            if accuracy_list[-1] < min(accuracy_list[-5:-1]): #if accuracy lower than lowest of last 4 values
                print('...Stopping Early...')
                break

    print("Training Complete --- %s seconds ---" % (round((time.time() - start_time),2)))
    # Get the accuracy on the test set after training complete
    model.load_state_dict(torch.load('train_valid_best_pretrained.pth')) #load best model
    with torch.no_grad():
        total = 0
        num_correct = 0
        for a, (context, label) in enumerate(testloader):
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context, CONTEXT_SIZE, EMBEDDING_DIM)
            yhat = yhat.view(-1,1)
            predictions = (yhat > 0.5)
            total += label.nelement()
            num_correct += torch.sum(torch.eq(predictions, label.bool())).item()
        accuracy = num_correct/total*100
        print('Test Accuracy: {} %'.format(round(accuracy,5)))
    return


if __name__ == "__main__":
    main()
