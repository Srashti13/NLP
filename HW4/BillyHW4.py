#%% Homework 4
import re
from collections import defaultdict
import numpy as np
import time
import torch.utils.data as data_utils
import torch
import torch.optim as optim
import torch.nn as nn 
import gc
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import math
from sklearn.model_selection import train_test_split

start_time = time.time()

def main():
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    train, test = preprocessdata()
    vectorized_data, indicies, revindicies, vocab = getvectors(train, test)
    # weights_matrix_torch = build_weights_matrix(vocab, "GoogleNews-vectors-negative300.txt", embedding_dim=300)

    run_RNN(vectorized_data, vocab, indicies, hidden_dim=256,
            bidirectional=True, pretrained_embeddings_status=False, RNNTYPE="LSTM")

    return

def preprocessdata():
    '''
    gets DF of data with BIO taggings
    '''
    def BIOconversion(df):
        '''
        convert to BIO taggings
        '''
        prop_values = []
        prop_value = "O"
        for i in range(len(df)):
            if df.iloc[i] == '<BREAK>':
                prop_values.append(df[i])
            elif df.iloc[i] == 'None':
                prop_values.append(df[i])
            elif df.iloc[i][0] == '(' and df.iloc[i][-1] == ')':
                prop_values.append(df.iloc[i][1:-2])
            elif df.iloc[i][0] == '(':
                prop_value = df.iloc[i][1:-1]
                prop_values.append("B-" + df.iloc[i][1:-1])
            elif df.iloc[i][-1] == ')':
                prop_values.append("I-" + prop_value)
                prop_value = ""
            elif df.iloc[i][-1] == '*' and prop_value == "":
                prop_values.append("O")
            elif df.iloc[i][-1] == '*' and (prop_value != "" and prop_value !="O"):
                prop_values.append("I-" + prop_value)
            else:
                prop_values.append(prop_value)     
        return prop_values
    print("--- Extracting Train --- %s seconds ---" % (round((time.time() - start_time),2)))
    train = r"data.wsj/train-set-small.txt"
    vocab = defaultdict(list)
    doc = []
    with open(train) as f:
        sentence = 0
        for line in f.read().splitlines():
            a = re.split(" +", line)
            if a[0] == " " or a[0] == "":
                doc.append(['<BREAK>']*15)
                sentence +=1
            else:
                doc.append([sentence] + a)

    train_df = pd.DataFrame(doc)
    train_df.columns = ["sentence", "word","pos","full_tree","ner","targetverb","prop1","prop2","prop3","prop4","prop5","prop6","prop7","prop8","prop9"]
    # print(train_df['prop1'].head(30))
    train_df.replace("", np.nan, inplace=True)
    train_df = train_df.fillna("None")
    for col in train_df.columns[6:]:
        train_df[col] = BIOconversion(train_df[col]) #all props to bio notation
    train_df['ner'] = BIOconversion(train_df['ner']) #all ner to bio notation
    # print(train_df.head(50))
    print("--- Extracting Test --- %s seconds ---" % (round((time.time() - start_time),2)))
    #preprocess test 
    test = r"data.wsj/test-set-small.txt"
    vocab = defaultdict(list)
    doc = []
    with open(test) as f:
        sentence = 0
        for line in f.read().splitlines():
            a = re.split(" +", line)
            if a[0] == " " or a[0] == "":
                doc.append(['<BREAK>']*15)
                sentence +=1
            else:
                doc.append([sentence] + a)

    test_df = pd.DataFrame(doc)
    test_df.columns = ["sentence","word","pos","full_tree","ner","targetverb","prop1","prop2","prop3","prop4","prop5","prop6","prop7","prop8","prop9"]
    test_df.replace("", np.nan, inplace=True)
    test_df = test_df.fillna("None")
    for col in test_df.columns[6:]:
        test_df[col] = BIOconversion(test_df[col])
    test_df['ner']= BIOconversion(test_df['ner'])
    # print(test_df.loc[test_df['prop1'] == 'None'])
    # print(test_df.iloc[12850:12890,:])
    # print(train_df.head(50))
    #drop training and testing sentence which have no arguments
    print("--- Splitting Train into Sentences --- %s seconds ---" % (round((time.time() - start_time),2)))
    #split into seperate sentences for each target verb
    trainsplit_df = pd.DataFrame()
    for s in train_df.sentence.unique():
        number = 0
        if s != '<BREAK>':
            df1 = train_df[train_df['sentence'] == s]
            for vrbloc in df1.index[df1['targetverb'] != '-'].tolist():
                verb = train_df.iloc[vrbloc,5]              
                for col in df1.columns:
                    if df1.at[vrbloc,col] == 'V':
                        df2 = pd.concat([df1.iloc[:,:6],df1.loc[:,col]],axis=1)
                        df2['targetverb'] = df2['targetverb'].apply(lambda x: 0 if x != verb else 1)
                        df2['verbnum'] = number
                        df2.columns = ['sentence','word','pos','full_tree','ner','targetverb','prop','verbnum']
                        df2=df2.reindex(columns=['sentence','verbnum','word','pos','full_tree','ner','targetverb','prop'])
                        trainsplit_df = pd.concat([trainsplit_df,df2],axis=0, sort=False)
                        number +=1
    # print(trainsplit_df)
    print("--- Splitting Test into Sentences --- %s seconds ---" % (round((time.time() - start_time),2)))
    testsplit_df = pd.DataFrame()
    for s in test_df.sentence.unique():
        number = 0
        if s != '<BREAK>':
            df1 = test_df[test_df['sentence'] == s]
            for vrbloc in df1.index[df1['targetverb'] != '-'].tolist():
                verb = test_df.iloc[vrbloc,5]              
                for col in df1.columns:
                    if df1.at[vrbloc,col] == 'V':
                        df2 = pd.concat([df1.iloc[:,:6],df1.loc[:,col]],axis=1)
                        df2['targetverb'] = df2['targetverb'].apply(lambda x: 0 if x != verb else 1)
                        df2['verbnum'] = number
                        df2.columns = ['sentence','word','pos','full_tree','ner','targetverb','prop','verbnum']
                        df2=df2.reindex(columns=['sentence','verbnum','word','pos','full_tree','ner','targetverb','prop'])
                        testsplit_df = pd.concat([testsplit_df,df2],axis=0, sort=False)
                        number +=1
    # print(testsplit_df.head(50))
    return trainsplit_df, testsplit_df

def getvectors(train,test):
    '''
    get numpy arrays for NN
    '''
    train =train.drop(['full_tree'],axis=1)
    test =test.drop(['full_tree'],axis=1)
    full = pd.concat([train,test])
    print("--- Vectorizing --- %s seconds ---" % (round((time.time() - start_time),2)))
    #index
    wordvocab = full['word'].unique().tolist() + ['<pad>']
    word_to_ix = {word: i for i, word in enumerate(set(wordvocab))} #index vocabulary
    vocab = full['pos'].unique().tolist() + ['<pad>']
    pos_to_ix = {pos: i for i, pos in enumerate(set(vocab))} #pos vocabulary
    vocab = full['ner'].unique().tolist() + ['<pad>']
    ner_to_ix = {ner: i for i, ner in enumerate(set(vocab))} #ner vocabulary
    vocab = full['prop'].unique().tolist() + ['<pad>']
    prop_to_ix = {prop: i for i, prop in enumerate(set(vocab))} #prop vocabulary
    
    #convert to numeric features
    train['word'] = train['word'].apply(lambda x: word_to_ix[x])
    test['word'] = test['word'].apply(lambda x: word_to_ix[x])
    
    train['pos'] = train['pos'].apply(lambda x: pos_to_ix[x])
    test['pos'] = test['pos'].apply(lambda x: pos_to_ix[x])
    
    train['ner'] = train['ner'].apply(lambda x: ner_to_ix[x])
    test['ner'] = test['ner'].apply(lambda x: ner_to_ix[x])
    
    train['prop'] = train['prop'].apply(lambda x: prop_to_ix[x])
    test['prop'] = test['prop'].apply(lambda x: prop_to_ix[x])

    combos = []
    train_sentences = []
    for word in train.values.tolist():
        # print(combos)
        # print((word[0],word[1]))
        # print(train_sentences)
        if (word[0],word[1]) not in set(combos):
            train_sentences.append([word])
            combos.append((word[0],word[1]))
        else:
            train_sentences[-1].extend([word])
    # y=np.array([np.array(xi) for xi in train_sentences])
    length = max(map(len, train_sentences))
    y=[xi+[[xi[0][0],xi[0][1],word_to_ix['<pad>'],word_to_ix['<pad>'],pos_to_ix['<pad>'],ner_to_ix['<pad>'],prop_to_ix['<pad>']]]*(length-len(xi)) for xi in train_sentences]
    y=np.array(y)
    train_sentences = y.transpose(0,2,1)
    train_labelsfull = train_sentences[:,-1,:]
    train_sentencesfull = train_sentences[:,:-2,:]
    splitpoint = int(round(train_sentencesfull.shape[0]*.8))
    print(train_labelsfull.shape)
    train_sentences = train_sentencesfull[:splitpoint,:,:]
    valid_sentences = train_sentencesfull[splitpoint:,:,:]
    train_labels = train_labelsfull[:splitpoint,:]
    valid_labels = train_labelsfull[splitpoint:,:]


    #same for test
    combos = []
    test_sentences = []
    for word in test.values.tolist():
        # print(combos)
        # print((word[0],word[1]))
        # print(train_sentences)
        if (word[0],word[1]) not in set(combos):
            test_sentences.append([word])
            combos.append((word[0],word[1]))
        else:
            test_sentences[-1].extend([word])
    # y=np.array([np.array(xi) for xi in train_sentences])
    length = max(map(len, test_sentences))
    y=[xi+[[xi[0][0],xi[0][1],word_to_ix['<pad>'],word_to_ix['<pad>'],pos_to_ix['<pad>'],ner_to_ix['<pad>'],prop_to_ix['<pad>']]]*(length-len(xi)) for xi in test_sentences]
    y=np.array(y)
    # print(y.shape)
    # print(y.transpose(0,2,1))
    test_sentences = y.transpose(0,2,1)
    test_labels = test_sentences[:,-1,:]
    test_sentences = test_sentences[:,:-2,:]
    print("--- Vectorizing Complete --- %s seconds ---" % (round((time.time() - start_time),2)))
    vectorizeddata = defaultdict()
    vectorizeddata = {'train_sents': train_sentences,
                      'train_lab': train_labels,
                      'valid_sents': valid_sentences,
                      'valid_lab': valid_labels,
                      'test_sents':test_sentences,
                      'test_lab':test_labels}

    indicies = defaultdict()
    indicies = {'word_to_ix': word_to_ix,
                'pos_to_ix':pos_to_ix,
                'ner_to_ix':ner_to_ix,
                'prop_to_ix':prop_to_ix}


    #back conversions
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

    #used to back convert to words from prop
    ix_to_prop = {} 
    for key, value in prop_to_ix.items(): 
        if value in ix_to_prop: 
            ix_to_prop[value].append(key) 
            print(value)
            print(key)
            print(ix_to_prop[value])
        else: 
            ix_to_prop[value]=[key] 

    #used to back convert to words from prop
    ix_to_ner = {} 
    for key, value in ner_to_ix.items(): 
        if value in ix_to_ner: 
            ix_to_ner[value].append(key) 
            print(value)
            print(key)
            print(ix_to_ner[value])
        else: 
            ix_to_ner[value]=[key] 

    #used to back convert to words from pos
    ix_to_pos = {} 
    for key, value in pos_to_ix.items(): 
        if value in ix_to_pos: 
            ix_to_pos[value].append(key) 
            print(value)
            print(key)
            print(ix_to_pos[value])
        else: 
            ix_to_pos[value]=[key] 

    revindicies = defaultdict()
    revindicies = {'ix_to_word': ix_to_word,
                'ix_to_prop':ix_to_prop,
                'ix_to_ner':ix_to_ner,
                'ix_to_pos':ix_to_pos}

    return vectorizeddata, indicies, revindicies, wordvocab

def build_weights_matrix(vocab, embedding_file, embedding_dim):
    """
    used to apply pretrained embeddigns to vocabulary
    """
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
            weights_matrix[i] = embeddings_index[word] #assign the pretrained embedding
            words_found += 1
        
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,)) #randomize out of vocabulary words
            words_not_found += 1
            
    print("{:.2f}% ({}/{}) of the vocabulary were in the pre-trained embedding.".format((words_found/len(vocab))*100,words_found,len(vocab)))
    return torch.from_numpy(weights_matrix)

def run_RNN(vectorized_data, vocab, revindicies, hidden_dim, weights_matrix_torch=[], bidirectional=False,pretrained_embeddings_status=False, 
            RNNTYPE="RNN"):
    '''
    This function uses pretrained embeddings loaded from a file to build an RNN of various types based on the parameters
    bidirectional will make the network bidirections, pretrained_embeddings_status=False will have the model learn its own embeddings
    '''

    def format_tensors(vectorized_data, dataset_type,num_mini_batches):
        '''
        helper function to format numpy vectors to the correct type, also determines the batch size for train, valid, and test sets
        based on minibatch size
        '''
        X = torch.from_numpy(vectorized_data[dataset_type+'_sents'])
        X = X.long()
        batch_size = math.ceil(X.size(0)/num_mini_batches) # 200 mini-batches per epoch
        y = torch.from_numpy(vectorized_data[dataset_type+'_lab'])
        y = y.long()
        tensordata = data_utils.TensorDataset(X,y)
        loader = data_utils.DataLoader(tensordata, batch_size=batch_size,shuffle=False)
        return loader

    # building data loaders
    NUM_MINI_BATCHES = 200 #not 2000 for time purposes
    trainloader = format_tensors(vectorized_data,'train',NUM_MINI_BATCHES)
    validloader = format_tensors(vectorized_data,'valid',NUM_MINI_BATCHES)
    testloader = format_tensors(vectorized_data,'test',NUM_MINI_BATCHES)
    
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
        def __init__(self, weights_matrix, hidden_size, bidirectional, pre_trained=True):
            super(RNNmodel, self).__init__()
            if bidirectional:
                num_directions = 2
            else:
                num_directions = 1
            self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, pre_trained)
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
            self.fc = nn.Linear(hidden_size*num_directions,10)
            
        def forward(self, inputs):
            embeds = self.embedding(inputs) # dim: batch_size x batch_max_len x embedding_dim
            out, _ = self.rnn(embeds) # dim: batch_size x batch_max_len x lstm_hidden_dim*directions 
            out = out.contiguous().view(-1, out.shape[2]) # dim: batch_size*batch_max_len x lstm_hidden_dim
            yhats = self.fc(out) # dim: batch_size*batch_max_len x num_tags                       #https://cs230-stanford.github.io/pytorch-nlp.html
            return yhats #CrossEntropy in pytorch takes care of softmax here


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
        for lab in range(1,10):
            weights.append(1-(flat_train_labels.count(lab)/(len(flat_train_labels)-flat_train_labels.count(0)))) #proportional to number without tags
        weights.insert(0,0) #zero padding values weight
        return weights
    weights = class_proportional_weights(vectorized_data['train_context_label_array'].tolist()) #zero out pads and reduce weights given to "O" objects in loss function
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    #initalize model parameters and variables
    model = RNNmodel(weights_matrix_torch, hidden_dim, bidirectional, pretrained_embeddings_status)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available...

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005) #learning rate set to 0.005 to converse faster -- change to 0.00001 if desired
    torch.backends.cudnn.benchmark = True #memory
    torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    
    metric_list = []
    best_metric = 0 
    print("Start Training --- %s seconds ---" % (round((time.time() - start_time),2)))
    for epoch in range(50): 
        iteration = 0
        running_loss = 0.0 
        for i, (context, label) in enumerate(trainloader):
            # zero out the gradients from the old instance
            optimizer.zero_grad()
            # Run the forward pass and get predicted output
            label = label.contiguous().view(-1) # convert to length batch_size*seq_len
            context = context.to(device)
            label = label.to(device)
            yhat = model.forward(context) #required dimensions for batching
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
                yhats = model.forward(context)
                index = yhats.max(1)[1] #index position of max value
                prediction = index.int().tolist()
                predictionsfull.extend(prediction)
                labelsfull.extend(label.int().tolist())
                del context, label, prediction #memory
            gc.collect()#memory
            torch.cuda.empty_cache()#memory
            # gpu_usage()

            # remove pads and "O" and do acc calculation:
            padindicies = [i for i, x in enumerate(labelsfull) if x == 0 or x==1] 
            for index in sorted(padindicies, reverse=True):
                del labelsfull[index]
                del predictionsfull[index]
            metricscore = accuracy_score(labelsfull, predictionsfull) #not sure if they are using macro or micro in competition
            metric_list.append(metricscore)
        print('--- Epoch: {} | Validation Accuracy (non-O): {} ---'.format(epoch+1, metric_list[-1])) 

        if metric_list[-1] > best_metric: #save if it improves validation accuracy 
            best_metric = metric_list[-1]
            bestmodelparams = torch.save(model.state_dict(), 'train_valid_best.pth') #save best model
        #early stopping condition
        if epoch+1 >= 5: #start looking to stop after this many epochs
            if metric_list[-1] < min(metric_list[-5:-1]): #if accuracy lower than lowest of last 10 values
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
            yhats = model.forward(context)
            index = yhats.max(1)[1] #index position of max value
            prediction = index.int().tolist()
            predictionsfull.extend(prediction) #saving for pad removal and pack conversion
            del context, label, prediction #memory
        gc.collect()#memory
        torch.cuda.empty_cache()#memory


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
    fname = 'results/{}--bidir={}--hidden_size={}--pretrain={}--results.txt'.format(RNNTYPE,bidirectional,hidden_dim,pretrained_embeddings_status)
    if os.path.exists(fname):
        os.remove(fname)
    f = open(fname,'w')
    for (i,element) in enumerate(labelsfull):
        f.write(formattedcontexts[i] + ' ' + formattedlabels[i] + ' ' + formattedpredictions[i] + '\n')
    f.close()
    print('--- {}--bidir={}--hidden_size={}--pretrain={}--results ---'.format(RNNTYPE,bidirectional,hidden_dim,pretrained_embeddings_status))
    evaluate_conll_file(open(fname,'r')) #evaluate using conll script

if __name__ == "__main__":
    main()