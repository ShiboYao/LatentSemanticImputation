import numpy as np
import pandas as pd
import sys


if len(sys.argv) != 2:
    print("specicy which PQ!")
    exit(0)

PQ = pd.read_csv(sys.argv[1]+".csv", index_col=0)
aff = pd.read_csv("../matrices/fullMat.csv", index_col=0)

PQ['freq'] = aff.iloc[:,-1]

PQ.to_csv("../matrices/"+sys.argv[1]+"_freq.csv")


