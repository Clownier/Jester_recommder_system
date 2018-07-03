#
# # -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data
L1 = [x[0] for x in X]
L2 = [x[1] for x in X]
L1 = np.array(L1).reshape(len(L1), 1)
L2 = np.array(L2).reshape(len(L2), 1)
X = np.hstack((L1, L2))
# print(X)
# plt.figure()
# plt.plot(X[:, 0], X[:, 1], 'k.')
# plt.show()
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    d = cdist(X, kmeans.cluster_centers_)
    temp = np.min(d,axis=1)
    meandistortions.append(sum(temp)/X.shape[0])

plt.plot(K, meandistortions,'bx-')
plt.show()  # 肘部法选最优K值
