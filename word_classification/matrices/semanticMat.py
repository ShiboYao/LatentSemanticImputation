'''
Grab relevant company name embeddings. 
Shibo Yao, May 8 2019
'''
import sys
import numpy as np
import pandas as pd


aff = pd.read_csv("affMat.csv", index_col = 0)
tokenFreq = pd.read_csv("../word2vec/tokenFreq.csv")
stocks = aff.index.tolist()

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("specify full or part or other big embedding matrix")
        exit()

    fname = sys.argv[1]

    with open("../word2vec/"+fname+".txt", 'r') as f:
        preembed = f.read().split('\n')
        if len(preembed[-1]) < 2:
            del preembed[-1]
    
    preembed = [p.split(' ', 1) for p in preembed]
    word_set = set([p[0] for p in preembed])
    word2vec = {}
    for p in preembed:
        word2vec[p[0]] = p[1]
    embeddings = [word2vec[w.lower()] for w in stocks if w.lower() in word_set]
    embed = np.array([p.split(' ') for p in embeddings])
    stock_less = [S for S in stocks if S.lower() in word_set]

    labelind = [i for i in range(aff.shape[0]) if stocks[i] in stock_less]
    label = aff.iloc[labelind,-1].values #don't want to mixed up labels
    freqind = [tokenFreq.Token.values.tolist().index(s.lower()) for s in stock_less]
    freq = tokenFreq.Freq.values[freqind] #also record frequency

    print(embed.shape)
    print(freq.shape)
    print(label.shape)

    mat = np.hstack((embed, label.reshape(-1,1)))
    mat = np.hstack((mat, freq.reshape(-1,1)))
    df = pd.DataFrame(mat, index = stock_less)
    df.to_csv(fname+"Mat.csv")


