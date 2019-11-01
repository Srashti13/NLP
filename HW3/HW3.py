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


def main():
    train_vocab, train_sentences = get_sentences(r"conll2003\train.txt")
    valid_vocab, valid_sentences = get_sentences(r"conll2003\valid.txt")
    test_vocab, test_sentences = get_sentences(r"conll2003\test.txt")
    vocab = dict_combination(train_vocab, valid_vocab, test_vocab)
    get_context_vectors(vocab, train_sentences, valid_sentences, test_sentences)

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
    vocab = defaultdict(list)
    doc = []
    with open(docs) as f:
        for word in f.read().splitlines():
            a = word.split(" ")
            if len(a)>1:
                vocab[a[0]].append(a[3])
                doc.append([a[0],a[3]])
            else: 
                doc.append(a[0])
    doc.insert(0,'')

    # retaining the unique tags for each vocab word
    for k,v in vocab.items():
        vocab[k] = (list(set(v)))
        
    # lowercasing all words that have some, but not all, uppercase
    def lower_repl(match):
        return match.group().lower()
    def lowercase_text(txt):
        txt = re.sub('([A-Z]+[a-z]+)',lower_repl,txt) #lowercase words that start with captial    
        return txt        

    for word in doc:
        if word:
            word[0] = lowercase_text(word[0])
            
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
    return vocab, sentences


def get_context_vectors(vocab, train_questions, train_labels, test_questions):
    '''
    Construct your n-grams: Create positive n-gram samples by collecting all pairs of adjacent
    tokens. Create 2 negative samples for each positive sample by keeping the first word the same
    as the positive sample, but randomly sampling the rest of the corpus for the second word. The
    second word can be any word in the corpus except for the first word itself. 
    
    This functions takes the docs and tokenized sentences and creates the numpyarrays needed for the neural network.
    --creates 2 fake grams for every real gram 
    '''
    import operator
    word_to_ix = {word: i+1 for i, word in enumerate(vocab)} #index vocabulary
    word_to_ix['XXPADXX'] = 0 #set up padding
    vocab.append('XXPADXX')

    train_context_values = [] #array of word index for context 
    for context in train_questions.values:
        train_context_values.append([word_to_ix[w] for w in context])

    test_context_values = [] #array of word index for context 
    for context in test_questions.values:
        test_context_values.append([word_to_ix[w] for w in context])
    
    train_context_labels = [] # list of labels for context
    for label in train_labels.values:
        train_context_labels.append([label])
        
    #convert to numpy array for use in torch  -- padding with index 0 for padding.... Should change to a random word...
    totalpadlength = max(max(map(len, train_context_values)),max(map(len, test_context_values))) #the longest question 
    train_context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in train_context_values]) #needed because without padding we are lost 
    test_context_array = np.array([xi+[0]*(totalpadlength-len(xi)) for xi in test_context_values]) #needed because without padding we are lost 
    train_context_label_array = np.array(train_context_labels) 

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
    return train_context_array, train_context_label_array, test_context_array, totalpadlength, ix_to_word, vocab
# from gensim.models.keyedvectors import KeyedVectors
# model = KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True)
# model.save_word2vec_format('D:\GoogleNews-vectors-negative300.txt', binary=False)
# model = KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True, limit=2000)

if __name__ == "__main__":
    main()