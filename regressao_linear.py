import numpy as np
xdata = np.array([1,2,3])
print("Dados de entrada", xdata)
ydata = np.array([1,4,8])
print("Dados de saida", ydata)
#calcular as medias
xbar = np.mean(xdata)
print("A media de X é", xbar)
ybar = np.mean(ydata)
print("A media de Y é", ybar)
b_1 = np.sum((ydata-ybar) * (xdata-xbar))/np.sum((xdata-xbar)**2)
print("O valor do coefiente angular", b_1)
b_0 = ybar - b_1 * xbar
print("O intercepto é", b_0)
#Funcao estimada


# 