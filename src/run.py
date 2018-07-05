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
              '4': KNeighborsClassifier(n_neighbors=3)}
model_string_dict = {'0': 'GaussianNB()',
                     '1': 'MLPClassifier()',
                     '2': 'LogisticRegression()',
                     '3': 'LinearSVC()',
                     '4': 'KNeighborsClassifier(n_neighbors=3)'}
# carry_point_func_dict = {'0': (lambda x: x.reshape(1, -1)),
#                          '1': (lambda x: x.reshape(1, -1)),
#                          '2': (lambda x: x.reshape(1, -1)),
#                          '3': (lambda x: x.reshape(1, -1)),
#                          '4': (lambda x: x.reshape(1, -1))}


def naive_bayes():
    print("Naive Bayes model run...")


if __name__ == "__main__":
    data = jester.read_all_file()
    while 1:
        try:
            print("please choose model")
            print(model_string_dict)
            k = input()
            X_test, Y_test, X_veri, Y_veri, X_train, Y_train = jester.split_data(data)
            print(model_string_dict.get(k))
            model = jester.get_model(model_dict.get(k), X_train, Y_train)
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
