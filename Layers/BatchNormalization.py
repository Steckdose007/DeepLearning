import numpy as np
from . import Helpers


class BatchNormalization:
    def __init__(self, channels):
        self.trainable = True
        self.testing_phase = False
        self.channels = channels
        self.bias, self.weights = self.initialize()
        self.moving_mean = 0
        self.moving_variance = 0
        self.decay = 0
        self.normalized_input = 0
        self.input_tensor = 0
        self.optimizer = None
        self.gradient_bias = 0
        self.gradient_weights = 0
        self.conv = False

    def initialize(self, weights_init=None, bias_init=None):
        return np.zeros(self.channels), np.ones(self.channels)

    def forward(self, input_tensor):
        # saving for shape
        self.orig_input = input_tensor
        if len(input_tensor.shape) > 2:
            self.conv = True
        if self.conv:
            input_tensor = self.reformat(input_tensor)
        # save for backwards pass
        self.input_tensor = input_tensor

        # check for testing phase and use moving targets if detected
        if self.testing_phase:
            normalized_input = (input_tensor - self.moving_mean) / np.sqrt(self.moving_variance + np.finfo(float).eps)
            if self.conv:
                res = self.weights * normalized_input + self.bias
                return self.reformat(res)
            return self.weights * normalized_input + self.bias

        # normal forward pass
        self.variance = np.var(input_tensor, axis=0)
        self.mean = np.mean(input_tensor, axis=0)
        self.normalized_input = (input_tensor - self.mean) / np.sqrt(self.variance + np.finfo(float).eps)

        # update moving targets
        self.moving_mean = self.decay * self.moving_mean + (1-self.decay)*self.mean
        self.moving_variance = self.decay * self.moving_variance + (1 - self.decay) * self.variance
        if self.conv:
            res = self.weights * self.normalized_input + self.bias
            return self.reformat(res)
        return self.weights*self.normalized_input + self.bias

    def backward(self, error_tensor):
        if self.conv:
            error_tensor = self.reformat(error_tensor)
        self.gradient_weights = np.sum(np.multiply(error_tensor, self.normalized_input), axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        if self.conv:
            res = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.variance)
            return self.reformat(res)
        return Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.variance)

    def reformat(self, tensor):
        if len(tensor.shape) > 2:
            self.orig_input = tensor

            b, h, m, n = self.orig_input.shape
            tensor = np.reshape(tensor, (b, h, m * n))
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = np.reshape(tensor, (b * m * n, h))
            return tensor

        b, h, m, n = self.orig_input.shape
        tensor = np.reshape(tensor, (b, m * n, h))
        tensor = np.transpose(tensor, (0, 2, 1))

        tensor = np.reshape(tensor, (b, h, m, n))
        return tensor

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer

    @property
    def mean(self):
        if self.testing_phase:
            return self.moving_mean
        return self.__mean

    @mean.setter
    def mean(self, mean):
        self.__mean = mean

    @property
    def variance(self):
        if self.testing_phase:
            return self.moving_variance
        return self.__variance

    @variance.setter
    def variance(self, variance):
        self.__variance = variance


