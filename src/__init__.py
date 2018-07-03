import xlrd
import sys
sys.path.append(".")
from . import readFile
path = r"./../DATA/jester-data-1.xls"
if __name__ == "__main__":
    date = readFile.readFile(path)
    readFile.get_data(date)