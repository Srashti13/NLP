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
    def BIOconversion(df):
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

    train = r"data.wsj/train-set.txt"
    vocab = defaultdict(list)
    doc = []
    sentences = []
    arguments = []
    with open(train) as f:
        sentence = []
        for line in f.read().splitlines():
            a = re.split(" +", line)
            if a[0] == " " or a[0] == "":
                doc.append(['<BREAK>']*14)
            else:
                doc.append(a)

    train_df = pd.DataFrame(doc)
    train_df.columns = ["word","pos","full_tree","ner","targetverb","prop1","prop2","prop3","prop4","prop5","prop6","prop7","prop8","prop9"]
    # print(train_df['prop1'].head(30))
    train_df.replace("", np.nan, inplace=True)
    train_df = train_df.fillna("None")
    for col in train_df.columns[5:]:
        train_df[col] = BIOconversion(train_df[col])
    # print(train_df.head(50))

    #preprocess test 
    test = r"data.wsj/test-set.txt"
    vocab = defaultdict(list)
    doc = []
    sentences = []
    arguments = []
    with open(test) as f:
        sentence = []
        for line in f.read().splitlines():
            a = re.split(" +", line)
            if a[0] == " " or a[0] == "":
                doc.append(['<BREAK>']*14)
            else:
                doc.append(a)

    test_df = pd.DataFrame(doc)
    test_df.columns = ["word","pos","full_tree","ner","targetverb","prop1","prop2","prop3","prop4","prop5","prop6","prop7","prop8","prop9"]
    test_df.replace("", np.nan, inplace=True)
    test_df = test_df.fillna("None")
    for col in test_df.columns[5:]:
        test_df[col] = BIOconversion(test_df[col])
    comb = pd.concat([test_df.iloc[:,5],train_df.iloc[:,5]])
    print(set(comb)) #set of argument tags 

    return train_df, test_df

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