"""
AIT726 HW 2 Due 10/10/2019
Sentiment classificaiton using Naive Bayes and Logistic Regression on a dataset of 25000 training and 25000 testing tweets.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW2.py 
i. main - runs all of the functions
    i. get_docs - tokenizes all tweets, returns a list of tokenized sentences and a list of all tokens
    ii. get_ngrams_vector - creates the context and label numpy arrays from the tokenized sentences
    iii. run_neural_network - splits numpy arrays into train,validation, and test sets. Runs on NN. Outputs accuracy on test set.


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
    ngram_array, ngram_label_array, vocab_size = get_ngrams_vector(docs,sentences) 
    run_neural_network(ngram_array, ngram_label_array, vocab_size)
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
    return ngram_array, ngram_label_array, len(vocab)


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
    
    BATCH_SIZE = 1 # 1000 maxes memory for 8GB GPU

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
        too low to effectively implement in a resonable amount of time. It is set to 0.001 for demonstration purposes. 
        '''
        def __init__(self, vocab_size, embedding_dim, context_size, batch_size, hidden_size):
            super(NGramLanguageModeler, self).__init__()
            self.embeddings = nn.Embedding(vocab_size*batch_size, embedding_dim)
            self.linear1 = nn.Linear(context_size * embedding_dim*batch_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, batch_size)
            self.out_act = nn.Sigmoid()
    
        def forward(self, inputs):
            embeds = self.embeddings(inputs).view((1, -1))
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
    model = NGramLanguageModeler(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, BATCH_SIZE, HIDDEN_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...
    model.apply(random_weights)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001) #learning rate set to 0.01 to converse faster -- change to 0.00001 if desired
    yhat_list = []
    context_list = []
    labels = []
    
    # setting these up because the neural network won't run if the batch size 
    # is not the same for all instances due to matrix sizes not matching up
    train_maxiter = X_train.size(0)//BATCH_SIZE
    valid_maxiter = X_valid.size(0)//BATCH_SIZE
    
    accuracy = 0
    print("Start Training --- %s seconds ---" % (round((time.time() - start_time),2)))
    for epoch in range(5): 
        iteration = 0
        running_loss = 0.0 
        print('--- Epoch: {} | Current Validation Accuracy: {} ---'.format(epoch+1, accuracy)) 
        for i, (context, label) in enumerate(trainloader):
            if i+1 < train_maxiter:
                # zero out the gradients from the old instance
                optimizer.zero_grad()
                # Run the forward pass and get predicted output
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context)
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
                # print('Epoch: {}, Iteration: {}, loss: {} running loss: {}'.format(epoch,iteration, loss.item(), running_loss/train_maxiter))  
            losses.append(loss.item())

    # Get the accuracy on the validation set for each epoch
        with torch.no_grad():
            total = 0
            num_correct = 0
            for a, (context, label) in enumerate(validloader):
                if a+1 < valid_maxiter:
                    context = context.to(device)
                    label = label.to(device)
                    yhat = model.forward(context)
                    yhat = yhat.view(-1,1)
                    predictions = (yhat > 0.5)
                    total += label.nelement()
                    num_correct += torch.sum(torch.eq(predictions, label.bool())).item()
            oldaccuracy = accuracy
            accuracy = num_correct/total*100
        if accuracy < oldaccuracy: #if accuracy is lowering on the validation set its time to stop.
            break
            # print('Validation Accuracy {}'.format(accuracy))

    print("Training Complete --- %s seconds ---" % (round((time.time() - start_time),2)))
    # Get the accuracy on the test set after training complete
    with torch.no_grad():
        test_maxiter = X_test.size(0)//BATCH_SIZE
        total = 0
        num_correct = 0
        for a, (context, label) in enumerate(testloader):
            if a+1 < test_maxiter:
                context = context.to(device)
                label = label.to(device)
                yhat = model.forward(context)
                yhat = yhat.view(-1,1)
                predictions = (yhat > 0.5)
                total += label.nelement()
                num_correct += torch.sum(torch.eq(predictions, label.bool())).item()
        accuracy = num_correct/total*100
        print('Test Accuracy: {} %'.format(round(accuracy,5)))
    return

    
if __name__ == "__main__":
    main()