import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

import jester_lib as jester

model_dict = {'0': GaussianNB(),
              '1': MLPClassifier(),
              '2': LogisticRegression(),
              '3': LinearSVC(),
              '4': KNeighborsClassifier(n_neighbors=3),
              '5': GaussianNB()}
model_string_dict = {'0': 'GaussianNB() to cutdown 1/4 feature',
                     '1': 'MLPClassifier()',
                     '2': 'LogisticRegression()',
                     '3': 'LinearSVC()',
                     '4': 'KNeighborsClassifier(n_neighbors=3)',
                     '5': 'GaussianNB()'}


# carry_point_func_dict = {'0': (lambda x: x.reshape(1, -1)),
#                          '1': (lambda x: x.reshape(1, -1)),
#                          '2': (lambda x: x.reshape(1, -1)),
#                          '3': (lambda x: x.reshape(1, -1)),
#                          '4': (lambda x: x.reshape(1, -1))}


def test(data, k):
    X_test, Y_test, X_veri, Y_veri, X_train, Y_train = jester.split_data(data)
    print(model_string_dict.get(k))
    model = jester.get_model(model_dict.get(k), X_train, Y_train)
    if k == '0':
        jestersort = jester.sigma(model)
        data = jester.futureData(jestersort, data)
    a, b, per = jester.verify(model, X_veri, Y_veri)
    return data, per


def futureData(data):
    X_test, Y_test, X_veri, Y_veri, X_train, Y_train = jester.split_data(data)
    model = jester.get_model(model_dict.get('0'), X_train, Y_train)
    jestersort = jester.sigma(model)
    return jester.futureData(jestersort, data)


def testAll():
    data = jester.read_all_file()
    ind = numpy.array(['1', '2', '3', '4', '5'])
    ind = ind.reshape(-1, 1)
    ind = numpy.hstack((ind, ind, ind, ind, ind))
    ind = numpy.hstack((ind, ind))
    ind = ind.reshape(-1)
    # ind = numpy.vstack((ind, ind, ind, ind, ind))
    # # ind = numpy.hstack((ind, numpy.array(['0', '0', '0', '0', '0']).reshape(-1, 1)))
    # ind = ind.reshape(-1)
    # ind = ind[25:-1]
    res = ind.copy().astype(float)
    # print(ind)
    data = futureData(data)
    data = futureData(data)
    data = futureData(data)
    for cur in range(len(ind)):
        # print(cur)
        data, per = test(data, ind[cur])
        res[cur] = per
    res = res.reshape(1, -1)
    jester.write_to_file(res)


if __name__ == "__main__":
    # testAll()
    data = jester.read_all_file()
    while 1:
        try:
            print("please choose model")
            print(model_string_dict)
            k = input()
            X_test, Y_test, X_veri, Y_veri, X_train, Y_train = jester.split_data(data)
            print(model_string_dict.get(k))
            model = jester.get_model(model_dict.get(k), X_train, Y_train)
            if k == '0':
                jestersort = jester.sigma(model)
                data = jester.futureData(jestersort, data)
            jester.verify(model, X_veri, Y_veri)
            # while 1:
            #     try:
            #         print("please choose label col in data")
            #         ind = int(input())
            #         X_test, Y_test, X_veri, Y_veri, X_train, Y_train = jester.split_data(jester.read_all_file(), label_col=ind)
            #         model = jester.get_model(model_dict.get(k), X_train, Y_train)
            #         jester.verify(model, X_veri, Y_veri, carry_point_func_dict.get(k))
            #     except EOFError or ValueError:
            #         break
        except EOFError:
            break
