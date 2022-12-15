import numpy as np
from . import Base
class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_shape = 0

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return np.reshape(input_tensor, (input_tensor.shape[0], int(input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3])))

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.input_shape)