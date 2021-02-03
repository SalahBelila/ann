import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def activate(self, x):
        self.x = x
        return 1/(1 + np.exp(-x))
    def differentiate(self):
        return self.activate(self.x) * (1 - self.activate(self.x))

class Relu:
    def __init(self):
        pass

    def activate(self, x):
        self.x = x
        return np.maximum(x, 0)
    def differentiate(self):
        x = self.x
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

class MeanSquare:
    def __init__(self):
        pass

    def activate(self, y_hat, y):
        self.y_hat = y_hat
        self.y = y
        return ((y - y_hat)**2)/2
    def differentiate(self):
        return (self.y - self.y_hat) * -1