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

start_time = time.time()

def main():
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    train, test = preprocessdata()
    vectorizeddata, indicies, revindicies, vocab = getvectors(train, test)
    weights_matrix_torch = build_weights_matrix(vocab, "GoogleNews-vectors-negative300.txt", embedding_dim=300)
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
    train_labels = train_sentences[:,-1,:]
    train_sentences = train_sentences[:,:-2,:]

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

if __name__ == "__main__":
    main()