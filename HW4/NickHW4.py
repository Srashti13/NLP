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
def main():
    train, test = preprocessdata()
    
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

if __name__ == "__main__":
    main()