'''
Accelerate code running via CPU multiprocessing. 
Shibo Yao, May 8 2019
'''
import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import multiprocessing as mp



def dis_base(pid, index, return_dic, x):
    p = len(index)
    n = x.shape[0]
    small_dis = np.zeros([p,n])
    for i in range(p):
        vec = x[index[i]]
        small_dis[i] = [np.linalg.norm(vec-x[j]) for j in range(n)]

    return_dic[pid] = small_dis


def multicore_dis(x, Q_index, n_jobs, func=dis_base):
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    index_list = np.array_split(Q_index, n_jobs)
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(i,index_list[i],return_dic,x))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    dis_mat = [return_dic[i] for i in range(n_jobs)]
    dis_mat = np.concatenate(dis_mat, axis=0)
    
    return dis_mat


def MST(dis, Q_index):
    dis = dis[:, Q_index]
    d = csr_matrix(dis.astype('float'))
    Tcsr = minimum_spanning_tree(d)
    del d
    mpn = Tcsr.toarray().astype(float)
    del Tcsr
    mpn[mpn!=0] = 1
    n = dis.shape[0]

    for i in range(n):
        for j in range(n):
            if mpn[i,j] != 0:
                mpn[j,i] = 1

    return mpn


def knn_base(pid, index, dis, mpn, delta, return_dic):
    n = len(index)
    small_graph = mpn[index]
    
    for i in range(n):
        ind = index[i]
        nn_index = np.argsort(dis[ind])[1:(delta+1)]
        degree = small_graph[i].sum()
        j = 0
        while degree < delta:
            if small_graph[i, nn_index[j]] == 0:
                small_graph[i, nn_index[j]] = 1
                degree += 1
            j += 1

    return_dic[pid] = small_graph


def MSTKNN(dis, Q_index, delta, n_jobs, spanning=True, func=knn_base):
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu
    
    mpn = np.zeros(dis.shape)
    if spanning:
        mst = MST(dis, Q_index)
        mpn[:,Q_index[0]:] = mst
    
    index_list = np.array_split(range(len(Q_index)), n_jobs)
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(i,index_list[i],dis,mpn,delta,return_dic))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    graph = [return_dic[i] for i in range(n_jobs)]
    graph = np.concatenate(graph, axis=0)
    
    return graph

    
def nnlsw(aff, graph, pid, sub_list, return_dic, epsilon):
    nrows = len(sub_list)
    ncols = graph.shape[1]
    W = np.zeros((nrows, ncols))
    for i in range(nrows):
        ind_i = sub_list[i]
        vec = aff[ind_i]#b vector in scipy documentation
        gvec = graph[ind_i]
        indK = [j for j in range(ncols) if gvec[j] == 1]
        delta = len(indK) 
        mat = aff[indK]#A matrix in scipy documentation
        w = nnls(mat.T, vec)[0]#return both weights and residual
        if epsilon is not None:
            tmp = w[w!=0].copy()
            w = w + epsilon*min(tmp)#all neighbors nonzero
        if sum(w)==0: 
            w = np.ones(len(w))
        w = w/sum(w) #need to normalize, w bounded between 0 and 1
    
        for ii in range(delta):
            W[i, indK[ii]] = w[ii]
    
    return_dic[pid] = W


def multicore_nnls(aff, graph, Q_index, n_jobs, epsilon=1e-1, func=nnlsw):
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    graph_list = np.array_split(range(len(Q_index)), n_jobs)#default axis=0
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(aff,graph,i,graph_list[i],return_dic,epsilon))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    W_p = np.eye(aff.shape[0]-graph.shape[0], graph.shape[1])
    W_q = [return_dic[i] for i in range(n_jobs)]
    W_q = np.concatenate(W_q, axis=0)

    return np.concatenate([W_p,W_q], axis=0)


def lazy(W):

    return (W + np.eye(W.shape[0]))/2




if __name__ == '__main__':
    arr = np.random.rand(400,200)
    Q_index = range(300,400)
    dis = multicore_dis(arr, Q_index, 5)
    graph = MSTKNN(dis, Q_index, 20, 5)
    W = multicore_nnls(arr,graph,Q_index,5)
    W = lazy(W)

    print(W[-1])
    print(W.shape)
    print(W.sum())
    print(min(W.sum(axis=1)))
