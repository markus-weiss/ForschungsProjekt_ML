import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = 0

def downloadData(link):
    data = pd.read_csv(link)
    return pd.read_csv(link)

def createColumnArray(data, column):
    columnarray = np.array(data)
    return np.array(columnarray[:, column])

def getroof(columndata):
    sum = 0
    for x1 in columndata:
        sum = (x1 + sum) / len(columndata)
    return sum

def gettehta1(X1, Y1, xRoof, yRoof):
    for x in X1:
        for y in Y1:
            return (x - xRoof) * (y - yRoof) / (x - xRoof) * (x - xRoof)


# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load CSV
# data = pd.read_csv("C:/Users/schnu/Desktop/PyCharm/test_data_rand.csv")

# data.set_index("Y", inplace=True)


# data.plot()
# plt.show()


# print(data.head())
#
#
# TestData = np.array(data)

# print(TestData[:, 0])


# xQuerArray = np.array(TestData[:, 0])
# yQuerArray = np.array(TestData[:, 1])


#
# for x in QuerArray:
#     additioner = additioner + x
#
# def GetQuer(QuerArray):
#
#         return additioner
#
#
# GetQuer(xQuerArray)
#
#
# def createTheta0(yRoof, theta1, xQuer ):
#    return