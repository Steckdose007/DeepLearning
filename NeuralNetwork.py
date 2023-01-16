import copy

import numpy as np
from Layers import *
from Optimization import *

class NeuralNetwork():
    def __init__(self,optim,weights_initializer,bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optim
        #loss value for each iteration
        self.loss = []
        #holds architekture
        self.layers = []
        #provides input layer and labels
        self.data_layer = None
        #layer providing loss and prediction
        self.loss_layer = None
        self.input_tensor=None
        self.label_tensor=None
        #property phase because for training f.e. some connections drop out zero
        self.phase = False

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        input_tensor = copy.deepcopy(self.input_tensor)
        loss = 0
        for layer in self.layers:
            if not self.phase and layer.optimizer is not None:
                if type(layer.optimizer) == tuple:
                    if layer.optimizer[0].regularizer is not None:
                        loss = loss + layer.optimizer[0].regularizer.norm(layer.weights)
                elif layer.optimizer.regularizer is not None:
                    loss = loss + layer.optimizer.regularizer.norm(layer.weights)
            input_tensor = layer.forward(input_tensor)
        loss = self.loss_layer.forward(input_tensor, self.label_tensor) + loss
        return loss

    def backward(self):
        #first backpropagating with label for current input starting with loss_layer
        error_tensor = self.loss_layer.backward(self.label_tensor)
        #iterate from back
        for i in self.layers[::-1]:
            error_tensor = i.backward(error_tensor)

    def append_layer(self, layer):
        if(layer.trainable == True):
            #deepcopy creates new object without modifying old one
            optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        for i in self.layers:
            input_tensor = i.forward(input_tensor)
        return input_tensor

    @property
    def phase(self):
        return self.__phase

    @phase.setter
    def phase(self, testing_phase):
        for layer in self.layers:
            layer.testing_phase = testing_phase
        self.__phase = testing_phase