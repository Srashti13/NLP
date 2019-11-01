"""
AIT726 HW 3 Due 11/07/2019
Named Entity Recognition using different types of recurrent neural networks.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW3.py 
"""
#%%

import re
from collections import defaultdict
import numpy as np
import time

start_time = time.time()

def main():
    print("--- Start Program --- %s seconds ---" % (round((time.time() - start_time),2)))
    train_vocab, train_sentences, totalpadlength = get_sentences(r"conll2003\train.txt")
    valid_vocab, valid_sentences, _ = get_sentences(r"conll2003\valid.txt")
    test_vocab, test_sentences, _ = get_sentences(r"conll2003\test.txt")
    vocab = dict_combination(train_vocab, valid_vocab, test_vocab)
    vectorized_data = get_context_vectors(vocab, train_sentences, valid_sentences, test_sentences)

    return

def dict_combination(dictone,dicttwo,dictthree):
    '''
    Helper function to combine three dictionaries into one and keep the unique set of values for all keys
    '''
    combined = defaultdict()
    ds = [dictone, dicttwo,dictthree]
    for k in list(set(list(ds[0].keys()) + list(ds[1].keys())+list(ds[2].keys()))):
        value = list(d[k] for d in ds)
        combined[k] = list(set([item for sublist in value for item in sublist]))
    return combined

#%%
def get_sentences(docs):

    # tokenizing: lowercasing all words that have some, but not all, uppercase
    def lower_repl(match):
        return match.group().lower()
    def lowercase_text(txt):
        txt = re.sub('([A-Z]+[a-z]+)',lower_repl,txt) #lowercase words that start with captial    
        return txt   


    vocab = defaultdict(list)
    doc = []
    with open(docs) as f:
        for word in f.read().splitlines():
            a = word.split(" ")
            if len(a)>1:
                vocab[lowercase_text(a[0])].append(a[3])
                doc.append([lowercase_text(a[0]),a[3]])
            else: 
                doc.append(a[0])
    doc.insert(0,'')

    # retaining the unique tags for each vocab word
    for k,v in vocab.items():
        vocab[k] = (list(set(v)))

    # getting the indices of the end of each sentence
    sentence_ends = []
    for i, word in enumerate(doc):
        if not word:
            sentence_ends.append(i)
    sentence_ends.append(len(doc)-1)
    #%% creating a list of all the sentences 
    sentences = []
    for i in range(len(sentence_ends)-1):
        sentences.append(doc[sentence_ends[i]+1:sentence_ends[i+1]])
        
    # getting the longest sentence
    max(sentences, key=len)
    # getting the length of the longest sentence
    max_sent_len = len(max(sentences, key=len))

    ## padding all of the sentences to make them length 113
    for sentence in sentences:
        sentence.extend(['0','<pad>'] for i in range(max_sent_len-len(sentence)))
    # This is the code to read the embeddings
    vocab['0'] = '<pad>'
    print("--- Text Extracted --- %s seconds ---" % (round((time.time() - start_time),2)))
    return vocab, sentences, max_sent_len


def get_context_vectors(vocab, train_sentences, valid_sentences, test_sentences):
    '''
    convert to numpy vectors 
    '''
    word_to_ix = {word: i for i, word in enumerate(vocab)} #index vocabulary

    train_context_values = [] #array of word index for context
    train_label_values = [] 
    for sentence in train_sentences:
        train_context_values.append([word_to_ix[w[0]] for w in sentence])
        train_label_values.append([w[1] for w in sentence])

    valid_context_values = [] #array of word index for context
    valid_label_values = [] 
    for sentence in valid_sentences:
        valid_context_values.append([word_to_ix[w[0]] for w in sentence])
        valid_label_values.append([w[1] for w in sentence])

    test_context_values = [] #array of word index for context
    test_label_values = [] 
    for sentence in test_sentences:
        test_context_values.append([word_to_ix[w[0]] for w in sentence])
        test_label_values.append([w[1] for w in sentence])
        
    #convert to numpy array for use in torch  -- padding with index 0 for padding.... Should change to a random word...
    train_context_array = np.array(train_context_values)
    valid_context_array = np.array(valid_context_values) 
    test_context_array = np.array(test_context_values)
    train_context_label_array = np.array(train_label_values)
    valid_context_label_array = np.array(valid_label_values)
    test_context_label_array = np.array(test_label_values) 
    arrays_and_labels = {"train_context_array":train_context_array,
                        "train_context_label_array":train_context_label_array,
                        "valid_context_array":valid_context_array,
                        "valid_context_label_array":valid_context_label_array,
                        "test_context_array":test_context_array,
                        "test_context_array":test_context_array,
                        "test_context_label_array":test_context_label_array}

    print("--- Arrays Created --- %s seconds ---" % (round((time.time() - start_time),2)))
    return arrays_and_labels
# from gensim.models.keyedvectors import KeyedVectors
# model = KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True)
# model.save_word2vec_format('D:\GoogleNews-vectors-negative300.txt', binary=False)
# model = KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True, limit=2000)

if __name__ == "__main__":
    main()