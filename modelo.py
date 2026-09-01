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
y_pred = b_0 + b_1 * xdata
print(y_pred)



#Avaliação do Modelo 
r_score = 1 - (np.sum((ydata - y_pred)**2) / np.sum((ydata - ybar)**2))
print(r_score)


class LinearRegression():
    def __init__(self, x, y):
        self.x = x
        self.y = y  
        self.b0 = None #coeficiente intercepto
        self.b1 = None #coeficiente angular
    def fit(self): #treinamento
        xbar = np.mean(self.x)
        ybar = np.mean(self.y)

        self.b1 = np.sum((self.y - ybar) * (self.x - xbar)) / np.sum((self.x - xbar)**2)
        self.b0 = ybar - self.b1 * xbar
        return self #representação do objeto

    def predict(self, x_new): #predição
        return self.b0 + self.b1 * np.array(x_new)

    def summary(self):
        print(f"Modelo: y = {self.b0} + {self.b1} * x")
        print(f"Interecepto (b0): {self.b0}")
        print(f"Coeficiente angular (b1): {self.b1}")
        return self


print("--------------------------------------------------")
LinearRegression(xdata, ydata).fit().summary()

# Estimar e prever o custo financeiro total (Y) que a seguradora terá
# com base em uma projeção da quantidade de acidentes ou de sinistros (X).
# Usando os dados reais do arquivo CSV.

dados = np.loadtxt('slr06 (1) (1).csv', delimiter=',', skiprows=1)

x_seg = dados[:, 0]  # quantidade de acidentes / sinistros
y_seg = dados[:, 1]  # custo financeiro total

modelo_seg = LinearRegression(x_seg, y_seg).fit()
modelo_seg.summary()

x_previsto = np.array([20, 50, 100, 150])
y_previsto = modelo_seg.predict(x_previsto)
print("Previsão de custo para valores de X:", x_previsto)
print("Valores previstos de Y:", y_previsto)

print("\nInterpretação:")
for x, y in zip(x_previsto, y_previsto):
    print(f"Se houver {x} acidentes/sinistros, o custo previsto será aproximadamente: {y:.2f}")

# Estimar e prever o custo financeiro total (Y) que a seguradora terá com base em uma projeção da quantidade de acidentes ou de sinistros (X).



