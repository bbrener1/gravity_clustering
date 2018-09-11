import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def fit(mtx,metric="euclidean"):

    dist = squareform(pdist(mtx,metric=metric))
    neighbors = np.argsort(dist,axis=1)



    return neighbors


print(fit(test))
