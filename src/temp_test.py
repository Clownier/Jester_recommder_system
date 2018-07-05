import numpy as np
X = np.array([[-1, -1], [-2, -2], [-3, -3],[-4,-4],[-5,-5], [1, 1], [2, 2], [3, 3]])
# Y = np.array([1, 1, 1,1,1, 2, 2, 2])
Y = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [2, 1], [2, 1], [2, 1]])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
#拟合数据
clf.fit(X, Y)
print(clf.set_params(priors=[0.625, 0.375]))#设置priors参数
clf.priors#返回各类标记对应先验概率组成的列表
clf.class_prior_