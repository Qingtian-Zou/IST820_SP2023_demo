import os
import sys

import numpy as np
from sklearn.utils import shuffle

if __name__ == "__main__":
    # load original data
    X=np.load(os.path.join("..","Data","X.npy"),allow_pickle=True)
    Y=np.load(os.path.join("..","Data","Y.npy"),allow_pickle=True)

    # shuffle
    X, Y = shuffle(X, Y, random_state=1)

    # flatten
    X=X.reshape((X.shape[0],X.shape[1]*X.shape[2]*X.shape[3]))

    # save as CSV files
    np.savetxt(os.path.join("..","Data","flattened-X.csv"),X,delimiter=',')
    np.savetxt(os.path.join("..","Data","flattened-Y.csv"),Y,delimiter=',')
