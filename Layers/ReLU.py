from .Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self, ):
        super().__init__()
        self.trainable = False



    def backward(self, error_tensor):
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor[input_tensor < 0] = 0
        return input_tensor