import numpy as np

xdata = np.array([1, 2, 3])
print("Dados de entrada:", xdata)
ydata = np.array([1,4,8])
print("Dados de saída:", ydata)
# calcular as medias
xbar = np.mean(xdata)
print("Média de x:", xbar)
ybar = np.mean(ydata)
print("Média de y:", ybar)
b_1 = np.sum((xdata - xbar) * (ydata - ybar)) / np.sum((xdata - xbar) ** 2)
print("Coeficiente angular (b1):", b_1)
b_0 = ybar - b_1 * xbar
print("Coeficiente linear (b0):", b_0)
# funcao estimada
def f(x):
    return b_0 + b_1 * x