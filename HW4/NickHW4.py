### Homework 4
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
from sklearn.preprocessing import LabelEncoder


def main():
    train, test = preprocessdata()
    train_context_array, encoded_roles, verb_list = get_sentences(train, test)
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
    
    #index
    vocab = full['word'].unique().tolist() + ['<pad>']
    word_to_ix = {word: i for i, word in enumerate(set(vocab))} #index vocabulary
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
    # print(train.head(50))
    # print(train.transpose().head(50))
    # train = train.transpose()
    #convert to numpy arrays and pad
    # print(train.values.tolist())
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
    print(y.shape)
    print(y.transpose(0,2,1))

def get_sentences(train, test):
    # Getting a list of every sentence and every semantic role corresponding to the 
    # words in the sentence
    break_indices = []
    for i, row in train.iterrows():
        if row['word'] == '<BREAK>':
            break_indices.append(i)
    
    sentences = []
    roles = []
    for i in range(len(break_indices)-1):
        for col in train.columns[5:]:
            if train[col][break_indices[i] + 1] == "None":
                break
            else:
               sentences.append(train['word'][break_indices[i]+1:break_indices[i+1]].tolist()) 
               roles.append(train[col][break_indices[i]+1:break_indices[i+1]].tolist())
    
    ## Getting a list of one hot encoded lists where 1 signifies the corresponding word is the Verb
    verb_list = []
    
    for sentence in roles:
        temp_verb_list = []
        for word in sentence:
            if word == "V":
                temp_verb_list.append(1)
            else:
                temp_verb_list.append(0)
        verb_list.append(temp_verb_list)
        
    #padding the sentences so that they're all equal length
    max_sent_len = 67
    for sentence in sentences:
        sentence.extend('<pad>' for i in range(max_sent_len-len(sentence)))
    for role in roles:
        role.extend('<pad>' for i in range(max_sent_len-len(role)))
    # using 2 as a padding signifier in the verb_list
    for verb in verb_list:
        verb.extend(2 for i in range(max_sent_len-len(verb)))
        
    # There are 89 different roles, which means this is a dense multiclass classification problem 
    types_of_roles = []
    for role in roles:
        for word in role:
            types_of_roles.append(word)
    
    print("Number of Argument Roles in Train: ", len(np.unique(types_of_roles)))

    # Transforming the roles into numerical form and creating an array [num_sentences x max_sent_len]
    encoded_roles = []
    le = LabelEncoder()
    le.fit(types_of_roles)
    for role in roles:
        encoding = le.transform(role)
        encoded_roles.append(encoding)
    encoded_roles = np.array(encoded_roles)

    # Getting the entire vocab of the training set and creating a vocab_index dict
    
    total_vocab = []
    
    for sent in sentences:
        for word in sent:
            total_vocab.append(word)
    
    total_vocab = list(set(total_vocab))
    vocab_index = {v:i+1 for i,v in enumerate(total_vocab) if v != '<pad>'}
    vocab_index['<pad>'] = 0

    # Mapping the training sentence words to an array
    train_context_values = []
    for sent in sentences:
        train_context_values.append([vocab_index[w] for w in sent])
    
    train_context_array = np.array(train_context_values)
    
    return train_context_array, encoded_roles, np.array(verb_list)
    
if __name__ == "__main__":
    main()