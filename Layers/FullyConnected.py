
from .Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):

    def __init__(self,input_size, output_size):
        super().__init__()
        self.trainable = True
        self. weights = np.random.rand(input_size+1,output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = None
        self._optimizer = None
        self.last_input = None
        self._gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @optimizer.setter
    def optimizer(self,setter):
        self._optimizer = setter
        return

    def forward(self,input_tensor):
        self.batch_size = input_tensor.shape[0]
        input_copy = np.ones((input_tensor.shape[0],input_tensor.shape[1]+1))
        for row in range(input_tensor.shape[0]):
            for col in range(input_tensor.shape[1]):
                input_copy[row][col] = input_tensor[row][col]
        input = np.array(input_copy)
        # berechne Forward pass
        output= (np.dot(self.weights.T,input.T)).T

        self.last_input = input
        return np.array(output)


    def backward(self,error_tensor):
        #gradient with respect to input
        grad_input = np.dot( error_tensor,self.weights[:-1,:].T)
        mat_mult = np.dot(self.last_input.T,error_tensor)
        if(self._optimizer is not None):
            self.weights = self._optimizer.calculate_update(self.weights,mat_mult)
        self._gradient_weights = mat_mult
        return grad_input

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        self.weights[-1] = bias_initializer.initialize(self.weights[-1].shape, 1, self.output_size)

