import random
import numpy
import time
import xlrd
numpy.seterr(divide='ignore', invalid='ignore')

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
    return data_martrix


def read_all_file():
    print("reading file ...")
    data = read_file(file_path=r"./../DATA/jester-data-1.xls")
    data = numpy.vstack((data, read_file(file_path=r"./../DATA/jester-data-2.xls")))
    data = numpy.vstack((data, read_file(file_path=r"./../DATA/jester-data-3.xls")))
    return data


def split_data(data, verification_proportion=0.1, label_col=-1):
    """
    :param data: 数据
    :param verification_proportion: 验证集比例
    :param label_col: 作为标签的列
    :return:
    """    # get 100% evaluation data index
    eval_full = []
    eval_none = []
    for cur in range(data.shape[0]):
        if data[cur, 0] == 100:
            eval_full.append(cur)
        else:
            eval_none.append(cur)
    data = data[:, 1:]
    assert numpy.fabs(label_col) < data.shape[1]
    if label_col < 0:
        label_col = data.shape[1] + label_col
    X = data[:, :label_col]
    if label_col != data.shape[1]-1:
        X = numpy.vstack((X, data[:, label_col + 1:]))
    Y = (data[:, label_col]).reshape(-1, 1)
    Y = Y + 0.1
    Y = Y / numpy.fabs(Y)
    Y = Y.astype(int)

    print("current label col = %d" % label_col)


    # get verification size and copy
    verification_size = int(len(eval_full) * verification_proportion)
    X_train = numpy.zeros((len(eval_full) - verification_size, X.shape[1]))
    Y_train = numpy.zeros((len(eval_full) - verification_size, 1))
    X_veri = numpy.zeros((verification_size, X.shape[1]))
    Y_veri = numpy.zeros((verification_size, 1))
    X_test = numpy.zeros((len(eval_none), X.shape[1]))
    Y_test = numpy.zeros((len(eval_none), 1))
    for cur in range(verification_size):
        X_veri[cur, :] = X[eval_full[cur], :]
        Y_veri[cur, :] = Y[eval_full[cur], :]
    for cur in range(verification_size + 1, len(eval_full)):
        X_train[cur - verification_size - 1, :] = X[eval_full[cur], :]
        Y_train[cur - verification_size - 1, :] = Y[eval_full[cur], :]
    for cur in range(len(eval_none)):
        X_test[cur, :] = X[eval_none[cur], :]
        Y_test[cur, :] = Y[eval_none[cur], :]
    return X_test, Y_test, X_veri, Y_veri, X_train, Y_train


def get_model(model, X_train, Y_train):
    """
    :param model: 训练模型
    :param train: 训练样本
    :param label: 训练样本标签
    :return:
    """
    assert model is not None
    # print(model)
    model.fit(X_train, Y_train.ravel())
    return model


def prediction(model, X_point, Y_point=0, carry_Point_func=(lambda x: x.reshape(1, -1))):
    """
    :param model: 训练好的模型
    :param X_point: 测试点
    :param Y_point: 测试理想解
    :param carry_Point_func: 对验证坐标点处理
    :return:
    """
    assert model is not None
    predict_label = model.predict(carry_Point_func(X_point))
    return predict_label, (True if predict_label[0] == Y_point else False)


def verify(model, X_veri, Y_veri, carry_Point_func=(lambda x: x.reshape(1, -1))):
    """
    :param model: 训练好的样本
    :param X_veri: 验证集
    :param Y_veri: 验证标签
    :param carry_Point_func: 对验证坐标点处理
    :return:
    """
    assert model is not None
    # since = time.time()
    count = 0
    sum = X_veri.shape[0]
    for cur in range(sum):
        point = X_veri[cur, :]
        predict_label, predict_succ = prediction(model, point, Y_veri[cur][0], carry_Point_func)
        if predict_succ:
            count += 1
    print("test num : %d, succ num : %d, succ precent: %f\n" % (sum, count, count * 1.0 / sum))
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
