from mylib import  *
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    data = read_all_file()
    test_martrix, verification_martrix, train_martrix = get_data(data, 0.1)
    print("Number(test,verification,train):(%d,%d,%d)" % (
        test_martrix.shape[0], verification_martrix.shape[0], train_martrix.shape[0]))
    verifyAll(train_martrix, verification_martrix,clf= LogisticRegression())