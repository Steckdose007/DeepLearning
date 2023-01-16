import numpy as np
from abc import ABC, abstractmethod
class BaseLayer(ABC):
    def __init__(self):
        self.trainable = False
        self.testing_phase = False
        self.optimizer = None
        self. weights = np.ones(5)

    @abstractmethod
    def forward(self, input_tensor):
        pass

    @abstractmethod
    def backward(self, error_tensor):
        pass

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer