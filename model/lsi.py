'''
Latent Semantic Imputation (S. Yao et.al, SIGKDD 2019)
Entity representation transfer from a geometry perspective. 
'''
import numpy as np
import pandas as pd
from hpc import *
import multiprocessing as mp


def permuteMat(aff, semantic): #permute affinity mat and semantic mat
    affInd = aff.index.tolist()
    semanticInd = semantic.index.tolist() #instead of index.values.tolist()
    Pind = [i for i in semanticInd if i in affInd]
    Qind = [i for i in affInd if i not in Pind]

    PMat = aff.loc[Pind].copy()
    QMat = aff.loc[Qind].copy()

    aff = pd.concat([PMat, QMat], axis=0)
    semanticInter = semantic.loc[Pind].copy()
    semanticOuter = semantic.drop(labels=Pind, axis=0)

    return aff, semanticInter, semanticOuter


def iterSolveQ(P, Weights, ita=1e-4, verbose=False):
    Pcolumns = P.columns
    P = P.values
    W = Weights.values
    mean = P.mean(axis = 0).reshape(1,-1) #mean along with row direction
    std = P.std(axis = 0).reshape(1,-1) #var along same direction
    N = W.shape[0]
    p = P.shape[0]
    q = N - p
    d = P.shape[1]
    W = W[-q:]
    Q = np.zeros((q,d))

    for i in range(q):
        noise = np.array([np.random.normal(0,s) for s in std])
        Q[i] = Q[i] + mean + noise

    PQ = np.vstack([P,Q])
    err = np.inf
    step = 0
    while err > ita: 
        dump = PQ[p:N].copy()
        PQ[p:N] = np.dot(W, PQ)
        err = sum(sum(abs(PQ[p:N]-dump))) / sum(sum(abs(dump)))
        if (verbose == True):
            print("%d %f" %(step, err))
            step += 1

    return pd.DataFrame(PQ, index=Weights.index, columns=Pcolumns)


def LSI(aff, semantic, delta, ita, spanning=True, lazyW=True, verbose=False, n_jobs=-1):
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    aff, semantic, semanticOuter = permuteMat(aff, semantic)
    N = aff.shape[0]
    P = semantic.shape[0]
    Q = N-P
    Q_index = range(N)#can change this to change the minimum spanning tree

    dis = multicore_dis(aff.values, Q_index, n_jobs)
    graph = MSTKNN(dis, Q_index, delta, n_jobs, spanning)
    W = multicore_nnls(aff.values, graph, Q_index, n_jobs, epsilon=1e-1)
    if lazyW:
        W = lazy(W)
    W = pd.DataFrame(W, index=aff.index)
    PQ = iterSolveQ(semantic, W, ita, verbose)

    return PQ



if __name__ == '__main__':
    aff = np.random.rand(1000, 500)
    aff = pd.DataFrame(aff)
    index = aff.index.values
    np.random.shuffle(index)
    aff.index = index

    semantic = np.random.rand(500, 200)
    semantic = pd.DataFrame(semantic)
    index = semantic.index.values
    np.random.shuffle(index)
    semantic.index = index

    aff,semantic,_ = permuteMat(aff, semantic)

    PQ = LSI(aff, semantic, 10, 1e-4, spanning=True, lazyW=True, n_jobs=-1)

