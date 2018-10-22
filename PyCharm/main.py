import LinearRegression as lr
import numpy as np

data = lr.downloadData("C:/Users/schnu/Desktop/PyCharm/test_data_rand.csv")

X1 = lr.createColumnArray(data, 0)
Y1 = lr.createColumnArray(data, 1)

xRoof = lr.getroof(X1)
yRoof = lr.getroof(Y1)


theta1 = lr.gettehta1(X1, Y1, xRoof, yRoof)

theta0 = yRoof - theta1*xRoof

predictor = 10

h = theta0 + theta1 * predictor

print(h)
