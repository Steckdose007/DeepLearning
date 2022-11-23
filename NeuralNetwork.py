import copy

import numpy as np
from Layers import *
from Optimization import *

class NeuralNetwork():
    def __init__(self,optim):
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

    def forward(self):
        #get input and label and store it for later use in backward
        self.input_tensor,self.label_tensor = self.data_layer.next()
        last = self.input_tensor
        #input durch alle durchschleusen und input von einer layer als input für neue
        for i in self.layers:
            last = i.forward(last)
        #calculate loss and return it
        loss = self.loss_layer.forward(last,self.label_tensor)
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
            layer._optimizer = optimizer
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        for i in self.layers:
            input_tensor = i.forward(input_tensor)
        return input_tensor