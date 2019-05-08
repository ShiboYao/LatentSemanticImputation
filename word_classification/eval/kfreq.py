import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sys


def KNN(X, y, n):
    l = len(y)
    y_hat = []
    
    for i in range(l): 
        X_train = np.delete(X, i, axis = 0) 
        y_train = np.delete(y, i, axis = 0) 
    
        neigh = KNeighborsClassifier(n_neighbors = n)
        neigh.fit(X_train, y_train)
        y_hat.extend(neigh.predict(X[i].reshape(1,-1)))
        
    acc = sum(np.array(y_hat) == y) / l

    return acc



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("specify which PQ!")
        exit(0)

    aff = pd.read_csv("../matrices/"+sys.argv[1]+"_freq.csv", index_col = 0)

    N = [1,2,3,4,5,6,7,8,9,10,15,20,30]    
    span = 100
    bar = list(reversed(range(span)))    
    result = pd.DataFrame(np.zeros([len(N), span]), index=N, columns=bar)

    for b in bar:
        HQind = [] # contain non-low-frequency word indexes
        for i in range(aff.shape[0]):
            if aff.iloc[i,-1] > b:
                HQind.append(i)

        X = aff.iloc[HQind,:-2].values #drop low-freq word embeddings
        y = aff.iloc[HQind,-2].values #drop low-freq word label accordinglly

        for n in N: # classification n_neighbors = 5, n_components = 30 up
            result.loc[n, b] = KNN(X, y, n)
    
    result = result.round(6)
    result.to_csv("result_"+sys.argv[1]+"_freq.csv")
