import numpy as np
from . import Base


class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.trainable = False
        self.probability = probability
        self.dropout_array = 0
        self.testing_phase = False

    def forward(self, input_tensor):
        if not self.testing_phase:
            self.dropout_array = np.random.rand(*input_tensor.shape) < self.probability
            return np.multiply(input_tensor, self.dropout_array)/self.probability
        return input_tensor

    def backward(self, error_tensor):
        return np.multiply(error_tensor, self.dropout_array)/self.probability

