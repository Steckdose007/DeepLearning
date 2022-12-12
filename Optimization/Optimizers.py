import numpy as np

class Sgd:
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - (self.learning_rate*gradient_tensor)
        return np.array(weight_tensor)
class SgdWithMomentum():
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.last_update = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        update = self.momentum_rate * self.last_update - self.learning_rate * gradient_tensor
        self.last_update = update
        return weight_tensor + update


class Adam():
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.lastV = 0
        self.lastR = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.mu * self.lastV + (1 - self.mu) * gradient_tensor
        self.lastV = v
        r = self.rho * self.lastR + (1 - self.rho) * (gradient_tensor ** 2)
        self.lastR = r
        vHat = v / (1 - self.mu ** self.k)
        rHat = r / (1 - self.rho ** self.k)
        self.k = self.k + 1
        return weight_tensor - self.learning_rate*(vHat / (np.sqrt(rHat) + np.finfo(float).eps))



