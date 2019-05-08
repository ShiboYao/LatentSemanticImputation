'''
Latent Semantic Imputation, driver. 
Shibo Yao, May 8 2019
'''

import sys
sys.path.append("../../model/")
import numpy as np
import pandas as pd
from lsi import *


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Specify semanticMat, delta and ita!")
        exit(0)

    fname = sys.argv[1]
    delta = int(sys.argv[2])
    ita = float('1e-'+sys.argv[3])

    semantic = pd.read_csv("../matrices/"+fname+"Mat.csv", index_col=0)
    semantic = semantic.iloc[:,:-2]#last 2 cols are frequency and label
    aff = pd.read_csv("../matrices/affMat.csv", index_col=0)
    aff,semantic,_ = permuteMat(aff, semantic)
    y = aff.y.copy()
    aff = aff.iloc[:,:-1]#last col is label

    PQ = LSI(aff, semantic, delta, ita, spanning=True, lazyW=True, n_jobs=-1)
    PQ['y'] = y#append label, index auto align
    PQ.to_csv(fname+sys.argv[2]+'-'+sys.argv[3]+'.csv')
