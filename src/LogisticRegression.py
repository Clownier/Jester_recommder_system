import random
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy
import xlrd


def read_file(file_path):
    """
    :param file_path: 读取文件路径
    :return: data_martrix
    """
    # 读数据
    file_table = xlrd.open_workbook(file_path).sheets()[0]
    data_martrix = numpy.zeros((file_table.nrows, file_table.ncols))
    for cur in range(file_table.ncols):
        data_martrix[:, cur] = numpy.matrix(file_table.col_values(cur))
    # 数据归一化

    # 无效数据处理

    # 数据随机化
    temp_martrix = data_martrix.copy()
    random_index = list(range(data_martrix.shape[0]))
    random.shuffle(random_index)
    for cur in range(data_martrix.shape[0]):
        temp_martrix[random_index[cur], :] = data_martrix[cur, :]
    data_martrix = temp_martrix.copy()
    # random_index = list(range(1, data_martrix.shape[1]))
    # random.shuffle(random_index)
    # for cur in range(1, data_martrix.shape[1]):
    #     data_martrix[:, random_index[cur-1]] = temp_martrix[:, cur]
    #     if random_index[cur - 1] == data_martrix.shape[1]-1:
    #         print("the last col = %d" % cur)
    return data_martrix


def get_data(data_martrix, verification_proportion=0.1):
    """
    :param data_martrix: 数据矩阵
    :param verification_proportion: 验证集比例
    :return: veri:(veri+train) = verification_proportion (100% evaluation) o.w. test
    """
    # get 100% evaluation data index
    eval_full = []
    eval_none = []
    for cur in range(data_martrix.shape[0]):
        if data_martrix[cur, 0] == 100:
            eval_full.append(cur)
        else:
            eval_none.append(cur)

    # get verification size and copy
    verification_size = int(len(eval_full) * verification_proportion)
    train_martrix = numpy.zeros((len(eval_full) - verification_size, data_martrix.shape[1]))
    verification_martrix = numpy.zeros((verification_size, data_martrix.shape[1]))
    test_martrix = numpy.zeros((len(eval_none), data_martrix.shape[1]))
    for cur in range(verification_size):
        verification_martrix[cur, :] = data_martrix[eval_full[cur], :]
    for cur in range(verification_size + 1, len(eval_full)):
        train_martrix[cur - verification_size - 1, :] = data_martrix[eval_full[cur], :]
    for cur in range(len(eval_none)):
        test_martrix[cur, :] = data_martrix[eval_none[cur], :]

    return test_martrix, verification_martrix, train_martrix


def naive_bayes_gaussian(train_martrix, label_train, predict_point, clf=None):
    """
    :param train_martrix: 训练样本
    :param label_train: 训练样本标签
    :param predict_point: 预测样本
    :return: 预测样本标签，proba 预测概率值
    """
    if clf == None:
        clf = GaussianNB()
        clf.fit(train_martrix, label_train)
    predict_label = clf.predict(predict_point)
    predict_proba = numpy.max(clf.predict_proba(predict_point))
    return predict_label, predict_proba


def verify(train_array, train_label, predecit_point, point_label, IOshow=False, clf=None):
    predict_label, predict_proba = naive_bayes_gaussian(train_array, train_label, predecit_point, clf)
    if IOshow:
        print("point label = ")
        print(point_label)
        print(" predict_label = ")
        print(predict_label[0])
        print("predict_proba = ")
        print(predict_proba)
    return point_label * predict_label[0] > 0


def verifyAll(train_array, verify_array, clf=GaussianNB()):
    train_martrix = train_array[:, 1:]
    verification_martrix = verify_array[:, 1:]
    train = train_martrix[:, :-1]
    label = train_martrix[:, -1] + 0.01
    label = label / (numpy.fabs(label))
    label = label.astype(int)
    print(label)
    label = label.reshape(-1, 1)
    count = 0
    sum = verification_martrix.shape[0]
    # clf = GaussianNB()
    clf.fit(train, label)
    for cur in range(sum):
        point = verification_martrix[cur, :-1]
        point = point.reshape(1, 99)
        if verify(train, label, point, verification_martrix[cur, -1], clf=clf):
            count += 1
    print("test num : %d, succ num : %d, succ precent: %f" % (sum, count, count * 1.0 / sum))


def read_all_file():
    data = read_file(file_path=r"./../DATA/jester-data-1.xls")
    data = numpy.vstack((data, read_file(file_path=r"./../DATA/jester-data-2.xls")))
    data = numpy.vstack((data, read_file(file_path=r"./../DATA/jester-data-3.xls")))
    return data

if __name__ == "__main__":
    data = read_all_file()
    test_martrix, verification_martrix, train_martrix = get_data(data, 0.1)
    print("Number(test,verification,train):(%d,%d,%d)" % (
        test_martrix.shape[0], verification_martrix.shape[0], train_martrix.shape[0]))
    verifyAll(train_martrix, verification_martrix, clf=LogisticRegression())
