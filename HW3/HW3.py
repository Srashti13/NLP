"""
AIT726 HW 3 Due 11/07/2019
Named Entity Recognition using different types of recurrent neural networks.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW3.py 
"""
#%%

import re

#%% reading in the training data and creating the vocab
vocab = {}
train = []
with open(r"conll2003\train.txt") as f:
    for word in f.read().splitlines():
        a = word.split(" ")
        if len(a)>1:
            vocab[a[0]] = a[3]
            train.append([a[0],a[3]])
        else: 
            train.append(a[0])
#%%
train.insert(0,'')

#%%
def lower_repl(match):
    return match.group().lower()
def lowercase_text(txt):
    txt = re.sub('([A-Z]+[a-z]+)',lower_repl,txt) #lowercase words that start with captial    
    return txt        

for word in train:
    if word:
        word[0] = lowercase_text(word[0])
#%%
sentence_ends = []
for i, word in enumerate(train):
    if not word:
        sentence_ends.append(i)
sentence_ends.append(len(train)-1)
#%%
sentences = []
for i in range(len(sentence_ends)-1):
    sentences.append(train[sentence_ends[i]+1:sentence_ends[i+1]])
    
#%%
# getting the longest sentence
max(sentences, key=len)
# getting the length of the longest sentence
max_sent_len = len(max(sentences, key=len))

#%% padding all of the sentences to make them length 113

for sentence in sentences:
    sentence.extend(['0','<pad>'] for i in range(max_sent_len-len(sentence)))
    

#%% This is the code to read the embeddings
    
# from gensim.models.keyedvectors import KeyedVectors
# model = KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True)
# model.save_word2vec_format('D:\GoogleNews-vectors-negative300.txt', binary=False)
# model = KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True, limit=2000)