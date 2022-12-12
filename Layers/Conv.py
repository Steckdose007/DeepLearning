import math

import numpy as np
from scipy import signal
from scipy.ndimage import zoom
from scipy import interpolate
from PIL import Image


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.optimizer_exist = False
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.is_two_d = False
        self.is_one_d = False
        if len(self.convolution_shape) > 2:
            self.is_two_d = True
        else:
            self.is_one_d = True
        self.num_kernels = num_kernels
        self.optimizer = None
        self.weights = np.random.uniform(0.0, 1.0, (self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0.0, 1.0, self.num_kernels)
        self.input_tensor = []
        self.input_size = np.prod(convolution_shape)
        self.output_size = np.prod(self.convolution_shape[1:]) * self.num_kernels

    def initialize(self, weights_initializer, bias_initializer):
        if self.is_two_d:
            self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
            self.bias = bias_initializer.initialize((self.num_kernels), self.num_kernels, 1)
        else:
            unref = weights_initializer.initialize(self.weights.shape, self.input_size, 1)
            self.weights = np.reshape(unref, (self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]))
            self.bias = bias_initializer.initialize((self.num_kernels), self.num_kernels, 1)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if self.is_two_d:
            # Input Tensor Dims  b, c, y, x
            self.output_height = math.ceil(input_tensor.shape[2] / self.stride_shape[0])
            self.output_width = math.ceil(input_tensor.shape[3] / self.stride_shape[1])
            self.output_size = (input_tensor.shape[0], self.num_kernels, self.output_height, self.output_width)
            output = np.zeros(self.output_size)

            for image in range(self.output_size[0]):
                for kernel in range(self.num_kernels):
                    res = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
                    for channel in range(input_tensor.shape[1]):
                        res = res + signal.correlate(input_tensor[image, channel, :, :],
                                                     self.weights[kernel, channel, :, :], mode='same')
                    someres = res[0::self.stride_shape[0], 0::self.stride_shape[1]]
                    someres = someres + self.bias[kernel]
                    output[image, kernel, :, :] = someres

            return output
        else:
            # Input of 1D is b, c, y
            self.output_width = math.ceil(input_tensor.shape[2] / self.stride_shape[0])
            self.output_size = (input_tensor.shape[0], self.num_kernels, self.output_width)
            output = np.zeros(self.output_size)

            for image in range(self.output_size[0]):
                for kernel in range(self.num_kernels):
                    res = np.zeros((input_tensor.shape[2]))
                    for channel in range(input_tensor.shape[1]):
                        res = res + signal.correlate(input_tensor[image, channel, :],
                                                     self.weights[kernel, channel, :], mode='same')
                    someres = res[0::self.stride_shape[0]]
                    output[image, kernel, :] = someres + self.bias[kernel]

            return output

    def backward(self, error_tensor):

        if self.is_two_d:
            # Gradient with respect to weights
            self.output_size = self.weights.shape

            self.gradient_weights = np.zeros(self.weights.shape)
            self.gradient_bias = np.zeros(self.num_kernels)

            for image in range(error_tensor.shape[0]):
                for kernel in range(self.gradient_weights.shape[0]):
                    for kernel_channel in range(self.gradient_weights.shape[1]):
                        padded_img = np.pad(self.input_tensor[image, kernel_channel, :, :],
                                            math.ceil(self.weights.shape[3] / 2))
                        err = error_tensor[image, kernel, :, :]
                        if self.stride_shape[0] > 1 or self.stride_shape[1] > 1:
                            right_size = np.zeros((error_tensor.shape[2] * self.stride_shape[0],
                                                   error_tensor.shape[3] * self.stride_shape[1]))
                            right_size[::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[image, kernel, :,
                                                                                         :]
                            err = right_size

                        x = signal.correlate(
                            padded_img, err,
                            mode='valid')

                        half_h = (x.shape[0] - self.weights.shape[2]) / 2
                        half_v = (x.shape[1] - self.weights.shape[3]) / 2
                        x = x[math.ceil(half_h):x.shape[0] - math.floor(half_h),
                            math.ceil(half_v):x.shape[1] - math.floor(half_v)]
                        self.gradient_weights[kernel, kernel_channel] = self.gradient_weights[
                                                                            kernel, kernel_channel] + x

            res = np.zeros(self.num_kernels)
            for image in range(error_tensor.shape[0]):
                for kernel in range(self.output_size[0]):
                    res[kernel] = res[kernel] + np.sum(error_tensor[image, kernel])

            self.gradient_bias = res

            # Gradient with respect to lower layers
            self.output_size = self.input_tensor.shape
            gradient_layer = np.zeros(self.output_size)
            for image in range(self.output_size[0]):
                for channel in range(self.output_size[1]):
                    new_kernel = np.zeros((error_tensor.shape[1], self.weights.shape[2], self.weights.shape[3]))
                    for kernel_channel in range(new_kernel.shape[0]):
                        new_kernel[kernel_channel] = self.weights[kernel_channel, channel]
                        empty_right_size = np.zeros((self.output_size[2], self.output_size[3]))
                        empty_right_size[::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[image,
                                                                                           kernel_channel, :, :]

                        x = signal.convolve(
                            empty_right_size, new_kernel[kernel_channel, :, :],
                            mode='same')
                        gradient_layer[image, channel] = gradient_layer[image, channel] + x

            if self.optimizer_exist:
                opt_w, opt_b = self.optimizer
                self.weights = opt_w.calculate_update(self.weights, self.gradient_weights)
                self.bias = opt_b.calculate_update(self.bias, self.gradient_bias)

            return gradient_layer
        else:
            # Gradient with respect to weights
            self.output_size = self.weights.shape

            self.gradient_weights = np.zeros(self.weights.shape)
            self.gradient_bias = np.zeros(self.num_kernels)

            for image in range(error_tensor.shape[0]):
                for kernel in range(self.gradient_weights.shape[0]):
                    for kernel_channel in range(self.gradient_weights.shape[1]):
                        padded_img = np.pad(self.input_tensor[image, kernel_channel, :],
                                            math.ceil(self.weights.shape[2] / 2))
                        err = error_tensor[image, kernel, :]
                        if self.stride_shape[0] > 1 or self.stride_shape[1] > 1:
                            right_size = np.zeros((error_tensor.shape[2] * self.stride_shape[0]))
                            right_size[::self.stride_shape[0]] = error_tensor[image, kernel, :]
                            err = right_size

                        x = signal.correlate(
                            padded_img, err,
                            mode='valid')

                        half_h = (x.shape[0] - self.weights.shape[2]) / 2
                        x = x[math.ceil(half_h):x.shape[0] - math.floor(half_h)]
                        self.gradient_weights[kernel, kernel_channel] = self.gradient_weights[
                                                                            kernel, kernel_channel] + x

            res = np.zeros(self.num_kernels)
            for image in range(error_tensor.shape[0]):
                for kernel in range(self.output_size[0]):
                    res[kernel] = res[kernel] + np.sum(error_tensor[image, kernel])

            self.gradient_bias = res

            # Gradient with respect to lower layers
            self.output_size = self.input_tensor.shape
            gradient_layer = np.zeros(self.output_size)
            for image in range(self.output_size[0]):
                for channel in range(self.output_size[1]):
                    new_kernel = np.zeros((error_tensor.shape[1], self.weights.shape[2]))
                    for kernel_channel in range(new_kernel.shape[0]):
                        new_kernel[kernel_channel] = self.weights[kernel_channel, channel]
                        empty_right_size = np.zeros(self.output_size[2])
                        empty_right_size[::self.stride_shape[0]] = error_tensor[image, kernel_channel, :]

                        x = signal.convolve(
                            empty_right_size, new_kernel[kernel_channel, :],
                            mode='same')
                        gradient_layer[image, channel] = gradient_layer[image, channel] + x

            if self.optimizer_exist:
                opt_w, opt_b = self.optimizer
                self.weights = opt_w.calculate_update(self.weights, self.gradient_weights)
                self.bias = opt_b.calculate_update(self.bias, self.gradient_bias)

            return gradient_layer

    @property
    def optimizer(self):
        return self.__weights_optimizer, self.__bias_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if optimizer is not None:
            self.optimizer_exist = True
            import copy
            self.__weights_optimizer = copy.deepcopy(optimizer)
            self.__bias_optimizer = optimizer
        else:
            self.__weights_optimizer = optimizer
            self.__bias_optimizer = optimizer

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.__gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self.__gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self.__gradient_bias = gradient_bias
