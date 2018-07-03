import random
import xlrd
import numpy
import pandas as pd

path = r"./../DATA/jester-data-1.xls"


# data = xlrd.open_workbook(path)
# table = data.sheets()[0]
# print(table.row_values(0))
# print(table.col_values(0))
# print(table.cell(0,0).value)

def readFile(file_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheets()[0]
    print(type(table))
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
    temp_martrix = data_martrix.copy()
    index = list(range(nrows))
    random.shuffle(index)
    print(index)
    for cur in range(nrows):
        temp_martrix[index[cur], :] = data_martrix[cur, :]
    data_martrix = temp_martrix.copy()
    count_martrix = data_martrix[:, 0]
    print(count_martrix)
    print(data_martrix[0])
    return data_martrix

def get_data(data):
    # train : verification : test = 8 : 1 : 1
    row = data.shape[0]
    print(row)
    cardinal = int(row/10)
    test = data[0:cardinal, :]
    verification = data[cardinal+1:2*cardinal, :]
    train = data[2*cardinal+1:, :]
    return test, verification, train


if __name__ == "__main__":
    date = readFile(path)
    get_data(date)
