import random
import xlrd
import numpy
import pandas as pd
import operator
from sklearn.cluster import KMeans

path = r"./../DATA/jester-data-1.xls"


def readFile(file_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheets()[0]
    nrows = table.nrows  # 获取总行数
    print("row : %d" % nrows)
    ncols = table.ncols  # 获取总列数
    print("col : %d" % ncols)
    data_martrix = numpy.zeros((nrows, ncols))
    for cur in range(ncols):
        cols = table.col_values(cur)
        cols = numpy.matrix(cols)
        if cur != 0:
            cols = (cols + 10.0) / 20.0
        data_martrix[:, cur] = cols

    # 无效数据修改
    for x in range(nrows):
        for y in range(ncols - 1):
            if data_martrix[x, y + 1] > 1.0:
                data_martrix[x, y + 1] = 0.5

    temp_martrix = data_martrix.copy()
    index = list(range(nrows))
    random.shuffle(index)
    for cur in range(nrows):
        temp_martrix[index[cur], :] = data_martrix[cur, :]
    data_martrix = temp_martrix.copy()
    count_martrix = data_martrix[:, 0]
    # print(count_martrix)
    # print(data_martrix[0])
    return data_martrix


# def get_data(data):
#     # train : verification : test = 8 : 1 : 1
#     row = data.shape[0]
#     print(row)
#     cardinal = int(row / 10)
#     test = data[0:cardinal, :]
#     verification = data[cardinal + 1:2 * cardinal, :]
#     train = data[2 * cardinal + 1:, :]
#     return test, verification, train


def get_data(data):
    """
    :param data: input data
    :return: train:verification = 9:1 (100% evaluation) o.w. test
    """
    train = []
    test = []
    for cur in range(data.shape[0]):
        if data[cur, 0] == 100:
            train.append(cur)
        else:
            test.append(cur)
    veri_size = int(len(train) / 10)
    train_set = numpy.zeros((len(train) - veri_size, data.shape[1]))
    verification_set = numpy.zeros((veri_size, data.shape[1]))
    test_set = numpy.zeros((len(test), data.shape[1]))
    for cur in range(veri_size):
        verification_set[cur, :] = data[train[cur], :]
    for cur in range(veri_size + 1, len(train)):
        train_set[cur - veri_size - 1, :] = data[train[cur], :]
    for cur in range(len(test)):
        test_set[cur, :] = data[test[cur], :]
    return test_set, verification_set, train_set



def training(data, n_clusters=8):
    outputfile = r"./../DATA/data_type.xlsx"
    model = KMeans(n_clusters=n_clusters)
    model.fit(data)
    eee = model.labels_
    r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    r = pd.concat([r2, r1], axis=1)  # 横向连接(0是纵向), 得到聚类中心对应的类别下的数目
    # r.columns = list(data.columns) + [u'类别数目']  # 重命名表头
    print(r)
    # 详细输出原始数据及其类别
    # r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)  # 详细
    # 输出每个样本对应的类别
    # r.columns = list(data.columns) + [u'聚类类别']  # 重命名表头
    r.to_excel(outputfile)  # 保存结果
    return model.labels_, model.cluster_centers_


def knn(point, train_set, k):
    """
    :param point: 测试样本点
    :param train_set: 训练样本集合
    :param k: top k nearest
    :return: top k nearest neighbor index
    """
    row = train_set.shape[0]
    assert point.shape[0] == train_set.shape[1]
    # 扩充样本点，使得可以运行矩阵运算
    # 计算欧式距离
    diffMat = numpy.tile(point, (row, 1)) - train_set
    diffMat = diffMat ** 2
    diffMat = diffMat.sum(axis=1)
    diffMat = diffMat ** 0.5
    # 对距离排序，返回排序数组下标
    sortedInd = diffMat.argsort()
    dist = []
    for cur in range(k):
        dist.append(1.0/diffMat[sortedInd[cur]])
    return sortedInd[0:k], dist
    # # 存放最终的分类结果及相应的结果投票数
    # classCount = {}
    # for i in range(k):
    #     label = train_labels[sortedInd[i]]
    #     classCount[label] = classCount.get(label, 0) + 1
    # sortedRes = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # return sortedRes[0][0]


def verify(verify, train, K):
    """
    :param verify: 验证数据组
    :param train: 训练数据
    :param K: top k nearest neighbor
    :return:  deviation: 误差绝对值
    """
    knn_res, dist = knn(verify[0:80], train[:, 0:80], K)
    res = train[knn_res[0], 80:]*dist[0]
    for t in range(1, K):
        res += train[knn_res[t], 80:]*dist[t]
    res = res / sum(dist)
    res = (res - verify[80:])
    res = numpy.fabs(res)
    res = numpy.mean(res)
    return res * 20


def verifyAll(verification, train, K):
    res = 0.0
    for cur in range(verification.shape[0]):
        res += verify(verification[cur, :], train, K)
    print("deviation : %f（绝对值）,verification count = %d" % (res, verification.shape[0]))
    print("mean = %f,percent = %f%%,current K = %d" % (res / verification.shape[0], res / verification.shape[0] * 5,K))


if __name__ == "__main__":
    data = readFile(path)
    test, verification, train = get_data(data)
    print("Number(test,verification,train):(%d,%d,%d)" % (test.shape[0], verification.shape[0], train.shape[0]))
    train = train[:, 1:]
    verification = verification[:, 1:]
    verifyAll(verification, train, 3)
    while 1:
        try:
            k = int(input())
            verifyAll(verification, train, k)
        except EOFError:
            break
