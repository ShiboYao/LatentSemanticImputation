import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt



def KNN(X, y, n):
    l = len(y)
    y_hat = []
    
    for i in range(l): 
        X_train = np.delete(X, i, axis = 0) 
        y_train = np.delete(y, i, axis = 0) 
    
        neigh = KNeighborsClassifier(n_neighbors = n)
        neigh.fit(X_train, y_train)
        y_hat.extend(neigh.predict(X[i].reshape(1,-1)))
        
    print("%d-NN" %n)
    print(sum(np.array(y_hat) == y) / l)



if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("specify full or part, frequency bar")
        exit(0)
    
    fname = sys.argv[1]
    bar = int(sys.argv[2]) #word frequency threshold, 40 corresbonds to dictionary size 50004
    aff = pd.read_csv(os.path.abspath("../matrices/"+fname), index_col = 0)

    HQind = [] # contain non-low-frequency word indexes
    for i in range(aff.shape[0]):
        if aff.iloc[i,-1] > bar:
            HQind.append(i)

    X = aff.iloc[HQind,:-2].values #get rid of low-freq word embeddings
    y = aff.iloc[HQind,-2].values #get rid of low-frw word label accordinglly

    N = [2,5,8,10,15,20,30]
    for n in N: # classification n_neighbors = 5, n_components = 30 up
        KNN(X, y, n)

