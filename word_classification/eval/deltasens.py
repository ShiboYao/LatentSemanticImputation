'''
Look at the model sensitivity on delta, the minimum degree of the graph
Shibo Yao, May 8 2019
'''
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append("../../model")
from lsi import *
from hpc import *


def KNN(X, y, n):
    l = len(y)
    y_hat = []
    
    for i in range(l):
        X_train = np.delete(X, i, axis = 0) 
        y_train = np.delete(y, i, axis = 0) 
    
        neigh = KNeighborsClassifier(n_neighbors = n, weights = 'distance')
        neigh.fit(X_train, y_train) #'distance' better than 'uniform'
        y_hat.extend(neigh.predict(X[i].reshape(1,-1)))
        
    acc = sum(np.array(y_hat) == y) / l

    return acc



if __name__ == "__main__":
    deltaset = [3, 5, 8, 10, 15, 20]
    N = [1,2,3,4,5,6,7,8,9,10,15,20,30] 
    result = pd.DataFrame(np.zeros([len(deltaset), len(N)]), index=deltaset, columns=N)
    semantic = pd.read_csv("../matrices/partMat.csv", index_col = 0)
    semantic = semantic.iloc[:,:-2] #last 2 cols are label and freq
    aff = pd.read_csv("../matrices/affMat.csv", index_col = 0)
    aff,semantic,_ = permuteMat(aff, semantic) 
    y = aff.y.copy().values 
    aff = aff.iloc[:,:-1]
    Q_index = range(aff.shape[0])
    dis = multicore_dis(aff.values, Q_index, n_jobs=-1)
  
    for delta in deltaset:
        graph = MSTKNN(dis, Q_index, delta, n_jobs=-1, spanning=True)
        W = multicore_nnls(aff.values, graph, Q_index, n_jobs=-1, epsilon=1e-1)
        W = lazy(W)
        W = pd.DataFrame(W, index=aff.index)
        PQ = iterSolveQ(semantic, W, ita=1e-2, verbose=False)
        X = PQ.values
       
        for n in N: # classification n_neighbors = 5, n_components = 30 up
            result.loc[delta, n] = KNN(X, y, n)
    
    result = result.round(6)
    result.to_csv("deltasens.csv")
