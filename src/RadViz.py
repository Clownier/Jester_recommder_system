
import pandas as pd

data = pd.read_csv('https://raw.github.com/pydata/pandas/master/pandas/tests/data/iris.csv')

# print(data.head())
# print(type(data))
import matplotlib.pyplot as plot

plot.figure()

from sklearn import manifold

from sklearn.metrics import euclidean_distances

similarities = euclidean_distances(data.ix[:,:-1].values)
print(similarities)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)

X =  .fit(similarities).embedding_

pos = pd.DataFrame(X, columns=['X', 'Y'])
print(pos)
pos['Name'] = data['Name']
print(pos.ix[pos['Name']=='Iris-virginica'])
# ax = pos.ix[pos['Name']=='Iris-virginica'].plot(kind='scatter', x='X', y='Y', color='blue', label='Iris-virginica')
#
# ax = pos.ix[pos['Name']=='Iris-setosa'].plot(kind='scatter', x='X', y='Y', color='green', label='Iris-setosa', ax=ax)
#
# pos.ix[pos['Name']=='Iris-versicolor'].plot(kind='scatter', x='X', y='Y', color='red', label='Iris-versicolor', ax=ax)

plot.show()