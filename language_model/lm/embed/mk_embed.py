import random
import pandas as pd
import sys


def genRand(dim): #randomly initialize embedding, return string
    scale = 0.3 #equavilant to init_scale in LSTM RNN
    l = [str(round(random.gauss(0, scale), 8)) for i in range(dim)]
    s = ' '.join(l) 

    return s


def build(word_list, preembed):
    '''
    word_list is a list of top words from corpus
    preembed is pretrained-embedding (list of string)
    based on word_list build a embedding for language model
    '''
    length = len(word_list)
    preembed = [p.split(' ', 1) for p in preembed]#separate token embedding
    dim = len(preembed[0][1].split(' '))
    word_set = set([p[0] for p in preembed])

    word2vec = {}#build dictionary
    for p in preembed:
        word2vec[p[0]] = p[1]

    embed = [word2vec[w] if w in word_set else genRand(dim) for w in word_list]#add if exist, otherwise randomly initialize
    result = [word_list[i] + ' ' + embed[i] for i in range(length)]
    result = '\n'.join(result)

    return result


if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print("specify path and new name!")
        exit(0)
    
    word_path = '../../data/finance/word_list.txt'
    preembed_path = sys.argv[1]

    with open(word_path, 'r') as f:
        word_list = f.read().split('\n')
        
    with open(preembed_path, 'r') as f:
        preembed = f.read().split('\n')
        if len(preembed[-1]) < 2:
            del preembed[-1]
    
    google = build(word_list, preembed)
    with open(sys.argv[2], 'w') as f:
        f.write(google)
        del google
        print(sys.argv[2], "saved.")
    
    '''
    #names = ['GoogleNews-vectors-negative300.txt', 'wiki.en.vec', 'glove.840B.300d.txt']
    #new_names = ['w2v.txt', 'fasttext.txt', 'glove.txt']
    for i in range(len(names)):
        print("build ", names[i])
        temp = build(word_list, names[i])
        with open(new_names[i], 'w') as f:
            f.write(temp)
        del temp
        print("saved.")
    '''
