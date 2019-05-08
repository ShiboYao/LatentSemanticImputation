'''
Print out the Q matrix norm change against power iteration step.
Shibo Yao, May 8 2019
'''
import numpy as np
import pandas as pd
import sys
sys.path.append("../../model")
from lsi import *


def KNN(X, y, n):
    l = len(y)
    y_hat = []
    
    for i in range(l):
        X_train = np.delete(X, i, axis = 0) 
        y_train = np.delete(y, i, axis = 0) 
    
        neigh = KNeighborsClassifier(n_neighbors = n, weights = 'distance')
        neigh.fit(X_train, y_train) 
        y_hat.extend(neigh.predict(X[i].reshape(1,-1)))
        
    acc = sum(np.array(y_hat) == y) / l

    return acc



if __name__ == "__main__":
    semantic = pd.read_csv("../matrices/partMat.csv", index_col = 0)
    semantic = semantic.iloc[:,:-2] #see reconstruct for rfe
    aff = pd.read_csv("../matrices/affMat.csv", index_col = 0)
    aff,_,_ = permuteMat(aff, semantic)  
    aff = aff.iloc[:,:-1]

    delta = 8
    ita = 1e-4
    PQ = LSI(aff, semantic, delta, ita, lazyW=True, verbose=True, n_jobs=-1)
