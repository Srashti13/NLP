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
#%%

train = r"C:\Users\nickn\Desktop\Grad School\AIT_726\Homework4\train-set.txt"
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
train_df.columns = ["word","pos","full_tree","ner","clauses","prop1","prop2","prop3","prop4","prop5","prop6","prop7","prop8","prop9"]
train_df = train_df.drop(columns=["full_tree","ner","clauses","pos"], axis=1)
         
            
#%%
total_train = []
for i in range(len(train_df.columns)-1):
    total_train.append((train_df[['word',train_df.columns[i+1]]].values.tolist()))

total_train = [word_pair for sentence in total_train for word_pair in sentence]
train_df = pd.DataFrame(total_train, columns=['word','prop'])

#%%

train_df = train_df[train_df['prop'].isna()==False]
train_df = train_df[train_df['prop']!=""]
train_df = train_df = train_df.reset_index(drop=True)
        
#%%
# removing rows where there are two duplicates in a row (usually <BREAK> items)
train_df = train_df.loc[(train_df.shift() != train_df).all(axis=1)]
train_df = train_df = train_df.reset_index(drop=True)


#%%
# labeling the dataframe with BIO structure
prop_values = []
for i in range(len(train_df)):
    if train_df['prop'].iloc[i] == '<BREAK>':
        prop_values.append(train_df['prop'][i])
    elif train_df['prop'].iloc[i][0] == '(' and train_df['prop'].iloc[i][-1] == ')':
        prop_values.append(train_df['prop'].iloc[i][1:-2])
    elif train_df['prop'].iloc[i][0] == '(':
        prop_value = train_df['prop'].iloc[i][1:-1]
        prop_values.append("B-" + train_df['prop'].iloc[i][1:-1])
    elif train_df['prop'].iloc[i][-1] == ')':
        prop_values.append("I-" + prop_value)
        prop_value = ""
    elif train_df['prop'].iloc[i][-1] == '*' and prop_value == "":
        prop_values.append("O")
    elif train_df['prop'].iloc[i][-1] == '*' and prop_value != "":
        prop_values.append("I-" + prop_value)
        
train_df['prop_values'] = prop_values
#%%
## making the target Verb (predicate) values apparent
train_df['target'] = train_df['prop_values'] == 'V'
train_df['target'] = train_df['target'].astype(int)