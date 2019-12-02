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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import time
import itertools
import csv
from nltk.util import ngrams
from nltk import word_tokenize, sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer, SnowballStemmer
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
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
from tqdm import tqdm
from sklearn.decomposition import PCA



localfolder = 'kaggle/input/quora-insincere-questions-classification/'
kagglefolder = '/kaggle/input/quora-insincere-questions-classification/'
start_time = time.time()


def main():
    '''
    The main function. This is used to get/tokenize the documents, create vectors for input into the language model based on
    a number of grams, and input the vectors into the model for training and evaluation.
    '''
    readytosubmit=True
    train_size = 3000 #1306112 is full dataset
    BATCH_SIZE = 512
    embedding_dim = 600
    erroranalysis = False
    
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    vocab, train_questions, train_labels, test_questions, train_ids, test_ids = get_docs(train_size, readytosubmit) 
    vectorized_data, wordindex, vocab, totalpadlength = get_context_vector(vocab, train_questions, train_labels, test_questions, readytosubmit)
    glove_embedding = build_weights_matrix(vocab, kagglefolder + r"embeddings/glove.840B.300d/glove.840B.300d.txt", wordindex=wordindex, embed_type='glove')
    para_embedding = build_weights_matrix(vocab, kagglefolder + r"embeddings/paragram_300_sl999/paragram_300_sl999.txt", wordindex=wordindex, embed_type='para')
    combined_embedding = torch.Tensor(np.hstack((para_embedding,glove_embedding)))
    del glove_embedding
    del para_embedding
    
    run_Attention_RNN(vectorized_data, test_ids, wordindex, len(vocab), combined_embedding, totalpadlength, num_epochs=3, 
                      threshold=0.5, nsplits=5, hidden_dim=60, learning_rate=0.001, batch_size=BATCH_SIZE)

    
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
        def replace_contractions(text):
            def replace(match):
                return contractions[match.group(0)]
            def _get_contractions(contraction_dict):
                contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
                return contraction_dict, contraction_re
            contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
            contractions, contractions_re = _get_contractions(contraction_dict)
            return contractions_re.sub(replace, text)
        def lower_repl(match):
            return match.group(1).lower()
        text=txt
        txt = replace_contractions(txt)
        txt = txt.translate(str.maketrans('', '', string.punctuation)) #removes punctuation - not used as per requirements  
        txt = re.sub(r'\d+', '#', txt) #replace numbers with a number token
        txt = re.sub('(?:<[^>]+>)', '', txt)# remove html tags
        txt = re.sub('([A-Z][a-z]+)',lower_repl,txt) #lowercase words that start with captial
        # Cleaning the number format
        txt = re.sub('[0-9]{5,}', '#####', txt)
        txt = re.sub('[0-9]{4}', '####', txt)
        txt = re.sub('[0-9]{3}', '###', txt)
        txt = re.sub('[0-9]{2}', '##', txt)
        # txt = r"This is a practice tweet :). Let's hope our-system can get it right. \U0001F923 something."
        tokens = word_tokenize(txt)
        if len(tokens) <=0:
            print(text)
            print(tokens)
        return tokens

    #initalize variables
    questions = defaultdict()
    labels = defaultdict()
    docs = []
    #laod data and tokenize
    if readytosubmit:
        train = pd.read_csv(kagglefolder + r'train.csv')
    else:
        train = pd.read_csv(localfolder + r'train.csv',nrows=train_size)
    #remove train questions that are less than 4 characters
    train = train[train['question_text'].map(len) > 2]
    train_questions = train['question_text']
    train_labels = train['target']
    train_ids = train['qid']
    tqdm.pandas()
    print("----Tokenizing Train Questions----")
    train_questions = train_questions.progress_apply(tokenize)
    
    if readytosubmit:
        test = pd.read_csv(kagglefolder + r'test.csv')
    else:
        test = pd.read_csv(localfolder + r'test.csv',nrows=1000) #doesnt matter
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
    
        
    #convert to numpy array for use in torch  -- padding with index 0 for padding.... Should change to a random word...
    totalpadlength = max(max(map(len, train_context_values)),max(map(len, test_context_values))) #the longest question 
    train_context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in train_context_values]) #needed because without padding we are lost 
    test_context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in test_context_values]) #needed because without padding we are lost 
    train_context_label_array = np.array(train_labels).reshape(-1,1)


    arrays_and_labels = defaultdict()
    arrays_and_labels = {"train_context_array":train_context_array,
                        "train_context_label_array":train_context_label_array,
                        "test_context_array":test_context_array}
    
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

def build_weights_matrix(vocab, embedding_file, wordindex, embed_type):
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
            embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    
    embedding_dim = embeddings_index[values[0]].shape[0]
    matrix_len = len(vocab)
    if embed_type == 'glove':
        embed_mean, embed_std = -0.00584, 0.48782
    elif embed_type == 'para':
        embed_mean, embed_std = -0.005325, 0.493465
        
    # Initializing the weights matrix as random normal values, so that any words not 
    # found will be placed in a random normal matrix
    weights_matrix = np.random.normal(embed_mean, embed_std, (matrix_len, embedding_dim))
    #weights_matrix = np.zeros((matrix_len, embedding_dim)) 
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
        words_not_found += 1
            
    print("{:.2f}% ({}/{}) of the vocabulary was in the pre-trained embedding.".format((words_found/len(vocab))*100,words_found,len(vocab)))
    return torch.from_numpy(weights_matrix)


def run_Attention_RNN(vectorized_data, test_ids, wordindex, vocablen, embedding_tensor, totalpadlength, num_epochs=3, 
     threshold=0.5, nsplits=5, hidden_dim=100, learning_rate=0.001,
     batch_size=500):
    '''
    This function uses pretrained embeddings loaded from a file to build an RNN of various types based on the parameters
    bidirectional will make the network bidirectional
    '''
    

    def create_emb_layer(weights_matrix):
        '''
        creates torch embeddings layer from matrix
        '''
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':weights_matrix})
        emb_layer.weight.requires_grad = False
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
            nn.init.xavier_uniform_(weight)
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
    
            a = a / torch.sum(a, 1, keepdim=True) + 1e-10
    
            weighted_input = x * torch.unsqueeze(a, -1)
            return torch.sum(weighted_input, 1)
      
    class Neural_Network(nn.Module):

        def __init__(self, hidden_size, weights_matrix,embedding_dim, context_size, vocablen, bidirectional_status=False, rnntype="RNN", pre_trained=True):
            super(Neural_Network, self).__init__()
            print('--- Using LSTM GRU with Attention Network ---')
            num_directions = 2
            if pre_trained:
                self.embedding, embedding_dim = create_emb_layer(weights_matrix)
            else:
                embedding_dim = embedding_dim
                self.embedding = nn.Embedding(vocablen, embedding_dim)
    
            self.embedding_dropout = nn.Dropout2d(0.1)
            self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True,
                                batch_first=True)
            self.gru = nn.GRU(hidden_size*num_directions, hidden_size, bidirectional=True,
                              batch_first=True)
    
            self.lstm_attention = Attention(hidden_size*num_directions, context_size)
            self.gru_attention = Attention(hidden_size*num_directions, context_size)
    
            self.attention = Attention(hidden_size, context_size)
            self.linear = nn.Linear(hidden_size*num_directions*4, 16)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.fc = nn.Linear(16, 1)
    
        def forward(self, inputs):            
            embeds = self.embedding(inputs) # [batch_size x totalpadlength x embedding_dim]
            h_lstm, _ = self.lstm(embeds) # [batch_size x totalpadlength x hidden_dim*num_directions]
            h_gru, _ = self.gru(h_lstm) # [batch_size x totalpadlength x hidden_dim*num_directions]
            h_lstm_attn = self.lstm_attention(h_lstm) # [batch_size x hidden_dim*num_directions]
            h_gru_attn = self.gru_attention(h_gru) # [batch_size x hidden_dim*num_directions]
            avg_pool = torch.mean(h_gru,1)
            max_pool, _ = torch.max(h_gru, 1)
            conc = torch.cat((h_lstm_attn, h_gru_attn, avg_pool, max_pool),1) #[batch_size x hidden_size*num_directions*4]
            conc = self.relu(self.linear(conc)) #[batch_size x 16]
            conc = self.dropout(conc) #[batch_size x 16]
            yhat = self.fc(conc) #[batch_size x 1]
            return yhat
            
        
    seed = 1234
    BATCH_SIZE = batch_size
    NUM_EPOCHS = num_epochs
    HIDDEN_DIM = hidden_dim
    EMBEDDING_DIM = 300
    sig_fn = nn.Sigmoid()
    N_SPLITS = nsplits
    THRESHOLD = threshold
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available
    splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(vectorized_data['train_context_array'], vectorized_data['train_context_label_array']))

    # using a numpy array because it's faster than a list
    predictionsfinal = torch.zeros((len(vectorized_data['test_context_array']),1), dtype=torch.float32)
    test_data = torch.tensor(vectorized_data['test_context_array'], dtype=torch.long).to(device)
    test = data_utils.TensorDataset(test_data)
    testloader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    print("--- Running Cross Validation ---")
    # Using K-Fold Cross Validation to train the model and predict the test set by averaging out the predictions across folds
    for i, (train_idx, valid_idx) in enumerate(splits):
        print("\n")
        print("--- Fold Number: {} ---".format(i+1))
        x_train_fold = torch.tensor(vectorized_data['train_context_array'][train_idx], dtype=torch.long).to(device)
        y_train_fold = torch.tensor(vectorized_data['train_context_label_array'][train_idx], dtype=torch.float32).to(device)
        x_val_fold = torch.tensor(vectorized_data['train_context_array'][valid_idx], dtype=torch.long).to(device)
        y_val_fold = torch.tensor(vectorized_data['train_context_label_array'][valid_idx], dtype=torch.float32).to(device)
        
        train = data_utils.TensorDataset(x_train_fold, y_train_fold)
        valid = data_utils.TensorDataset(x_val_fold, y_val_fold)
        
        trainloader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        validloader = data_utils.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
        
        model = Neural_Network(HIDDEN_DIM, embedding_tensor, EMBEDDING_DIM, totalpadlength, vocablen)
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        torch.backends.cudnn.benchmark = True #memory
        torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
        f1_list = []
        best_f1 = 0 

        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            iteration = 0
            running_loss = 0.0
            model.train()
            for i, (context, label) in enumerate(trainloader):
                iteration += 1
                # zero out the gradients from the old instance
                optimizer.zero_grad()
                # Run the forward pass and get predicted output
                yhat = model.forward(context) #required dimensions for batching
                # Compute Binary Cross-Entropy
                loss = criterion(yhat, label)
                loss.backward()
                optimizer.step()
                # Get the Python number from a 1-element Tensor by calling tensor.item()
                running_loss += float(loss.item())
    
                if not i%100:
                    print("Epoch: {:03d}/{:03d} | Batch: {:03d}/{:03d} | Cost: {:.4f}".format(
                            epoch+1,NUM_EPOCHS, i+1,len(trainloader),running_loss/iteration))
                    iteration = 0
                    running_loss = 0.0

            # Get the accuracy on the validation set for each epoch
            model.eval()
            with torch.no_grad():
                valid_predictions = torch.zeros((len(x_val_fold),1))
                valid_labels = torch.zeros((len(x_val_fold),1))
                for a, (context, label) in enumerate(validloader):
                    yhat = model.forward(context)
                    valid_predictions[a*BATCH_SIZE:(a+1)*BATCH_SIZE] = (sig_fn(yhat) > 0.5).int()
                    valid_labels[a*BATCH_SIZE:(a+1)*BATCH_SIZE] = label.int()
    
                f1score = f1_score(valid_labels,valid_predictions,average='macro') #not sure if they are using macro or micro in competition
                f1_list.append(f1score)
                
            print('--- Epoch: {} | Validation F1: {} ---'.format(epoch+1, f1_list[-1])) 
            running_loss = 0.0
            
            if f1_list[-1] > best_f1: #save if it improves validation accuracy 
                best_f1 = f1_list[-1]
                torch.save(model.state_dict(), 'train_valid_best.pth') #save best model
                
                
        kfold_test_predictions = torch.zeros((len(vectorized_data['test_context_array']),1))
        
        model.load_state_dict(torch.load('train_valid_best.pth'))
        model.eval()
        with torch.no_grad():
            for a, context in enumerate(testloader):
                yhat = model.forward(context[0])
                kfold_test_predictions[a*BATCH_SIZE:(a+1)*BATCH_SIZE] = (sig_fn(yhat) > 0.5).int()  #ranking instead of probs
            
            
            predictionsfinal += (kfold_test_predictions/N_SPLITS)
            
        # removing the file so that the next split can update it
        os.remove("train_valid_best.pth")    
        
    predictionsfinal = (predictionsfinal > THRESHOLD).int()
    output = pd.DataFrame(list(zip(test_ids.tolist(),predictionsfinal.numpy().flatten())))
    output.columns = ['qid', 'prediction']
    print(output.head())
    output.to_csv('submission.csv', index=False)

    
    print("Attention Model Completed --- %s seconds ---" % (round((time.time() - start_time),2)))


if __name__ == "__main__":
    main()