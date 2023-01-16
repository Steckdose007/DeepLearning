import numpy as np


class Sigmoid():
    def __init__(self, ):
        super().__init__()
        self.trainable = False
        self.activations = 0

    def forward(self, input_tensor):
        self.activations = 1/(1+np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        return error_tensor*self.activations*(1-self.activations)