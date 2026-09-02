import numpy as np
import math
random_state = 42
np.random.seed(random_state)
x = np.array([34,5,6,4])
print(np.random.permutation(x)) #embaralhar os dados


def train_test_split(X, y, test_size = 0.3, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho!")
    n_samples = len(X)
    print("Amostra Quantidade", n_samples)
    indices = np.random.permutation(n_samples)
    print("Indices embaralhados", indices)
    n_test = math.ceil(n_samples * test_size)
    print("Tamanho da amostra de teste", n_test)
    test_indices = indices[:n_test]
    print("Indices de Teste", test_indices)
    train_indices = indices[n_test:]
    print("Indices de treino", train_indices) 
    if X.ndim == 1:
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X[train_indices,:], X[test_indices,:]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

class MRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.b0 = None
        self.b1 = None

    def fit(self):
        Xbar = np.mean(self.X, axis=0)
        ybar = np.mean(self.y)
        self.b1 = np.linalg.inv(self.X.T @ self.X) @ (self.X.T @ self.y)
        self.b0 = ybar - Xbar @ self.b1
        return self

    def predict(self, X_new):
        return self.b0 + X_new @ self.b1

    def summary(self):
        print(f"Modelo: y = {self.b0} + {self.b1} * X")
        print(f"Intercepto (b0): {self.b0}")
        print(f"Coeficientes (b1): {self.b1}")
        return self

x1 = np.array([2,8,11,10,8,4,2,2,9,8])
x2 = np.array([50, 110, 120,550,295, 200, 375,52, 100, 300])
y = np.array([9.95,24.45,31.75, 35, 25.02, 16.86, 14.38,9.6,24.35,27.5])
X = np.column_stack((x1,x2))
print(len(X))
train_test_split(X, y, test_size=0.3, random_state=42)