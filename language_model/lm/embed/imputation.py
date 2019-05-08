import numpy as np
import pandas as pd
import sys
sys.path.append("../../../model")
from lsi import *
import time


if __name__ == '__main__':
    if (len(sys.argv) != 5):
        print("Specify aff, semantic, k and new name!")
        exit(0)

    affname = sys.argv[1]
    semanticname = sys.argv[2]
    k = int(sys.argv[3])
    newname = sys.argv[4]

    semantic = pd.read_csv(semanticname, sep=' ', index_col=0, header=None)
    print("semantic shape ", semantic.shape)
    aff = pd.read_csv(affname, sep =' ', index_col=0, header=None)
    aff, semantic, semanticOuter = permuteMat(aff, semantic)
    print("affinity shape ", aff.shape)

    start = time.time()
    PQ = LSI(aff, semantic, k, 1e-3, spanning=True, lazyW=True)
    PQ = pd.concat([PQ, semanticOuter], axis=0)
    print(PQ.shape)
    PQ.to_csv(newname, sep=' ', index=True, header=False)
    print("PQ saved.")
