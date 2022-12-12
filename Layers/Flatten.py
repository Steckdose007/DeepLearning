import numpy as np
from . import Base
class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_shape = 0

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return np.reshape(input_tensor, (input_tensor.shape[0], int(np.prod(self.input_shape)/self.input_shape[0])))

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.input_shape)