#!/usr/bin/python
import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as datasets
import sklearn.metrics as metrics
import sklearn.utils as utils


import scipy.sparse.linalg as linalg
import scipy.cluster.hierarchy as hr
import sklearn.cluster as cluster


#classification
from sklearn.neighbors import KNeighborsClassifier


from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

import seaborn as sns
%matplotlib inline

print "fin import"
X, y = datasets.make_circles(noise=.1, factor=.5)
print "X.shape:", X.shape
print "unique labels: ", np.unique(y)

plt.prism()  # this sets a nice color map
plt.scatter(X[:, 0], X[:, 1], c=y)
