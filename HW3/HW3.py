"""
AIT726 HW 3 Due 11/07/2019
Named Entity Recognition using different types of recurrent neural networks.
Authors: Srashti Agrawal, Billy Ermlick, Nick Newman
Command to run the file: python HW3.py 
"""
#%%

import re
from collections import defaultdict



def main():
    train_vocab, train_sentences = get_sentences(r"conll2003\train.txt")
    valid_vocab, valid_sentences = get_sentences(r"conll2003\valid.txt")
    test_vocab, test_sentences = get_sentences(r"conll2003\test.txt")
    vocab =  dict(dict(train_vocab, **valid_vocab), **test_vocab)


    return
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

    return vocab, sentences

# from gensim.models.keyedvectors import KeyedVectors
# model = KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True)
# model.save_word2vec_format('D:\GoogleNews-vectors-negative300.txt', binary=False)
# model = KeyedVectors.load_word2vec_format('D:\GoogleNews-vectors-negative300.bin', binary=True, limit=2000)

if __name__ == "__main__":
    main()