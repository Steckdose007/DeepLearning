import math

import numpy as np
from scipy import signal
from scipy.ndimage import zoom
from scipy import interpolate
from PIL import Image


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.HaveOptimizer = False
        self.trainable = True
        self.strideShape = stride_shape
        self.convolutionShape = convolution_shape
        self.twoDim = False
        self.oneDim = False
        if len(self.convolutionShape) > 2:
            self.twoDim = True
        else:
            self.oneDim = True
        self.numberKernels = num_kernels
        self.optimizer = None
        self.weights = np.random.uniform(0.0, 1.0, (self.numberKernels, *self.convolutionShape))
        self.bias = np.random.uniform(0.0, 1.0, self.numberKernels)
        self.input_tensor = []
        self.input_size = np.prod(convolution_shape)
        self.output_size = np.prod(self.convolutionShape[1:]) * self.numberKernels


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if self.twoDim:
            # Input Tensor  b, c, y, x num_images,channel,height,width
            self.outputHeight = math.ceil(input_tensor.shape[2] / self.strideShape[0])
            self.outputWidth = math.ceil(input_tensor.shape[3] / self.strideShape[1])
            self.output_size = (input_tensor.shape[0], self.numberKernels, self.outputHeight, self.outputWidth)
            output = np.zeros(self.output_size)

            for img in range(self.output_size[0]):
                for kernel in range(self.numberKernels):
                    res = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
                    for channel in range(input_tensor.shape[1]):
                        #flip kernel here because weights are random anyway
                        res = res + signal.correlate(input_tensor[img, channel, :, :],
                                                     self.weights[kernel, channel, :, :], mode='same')
                    som = res[0::self.strideShape[0], 0::self.strideShape[1]]
                    som = som + self.bias[kernel]
                    output[img, kernel, :, :] = som

            return output
        else:
            # Input Tensor b, c, y num_images,channel,length
            self.outputWidth = math.ceil(input_tensor.shape[2] / self.strideShape[0])
            self.output_size = (input_tensor.shape[0], self.numberKernels, self.outputWidth)
            output = np.zeros(self.output_size)

            for img in range(self.output_size[0]):
                for kernel in range(self.numberKernels):
                    res = np.zeros((input_tensor.shape[2]))
                    for channel in range(input_tensor.shape[1]):
                        res = res + signal.correlate(input_tensor[img, channel, :],
                                                     self.weights[kernel, channel, :], mode='same')
                    som = res[0::self.strideShape[0]]
                    output[img, kernel, :] = som + self.bias[kernel]

            return output
    def initialize(self, weights_initializer, bias_initializer):
        if self.twoDim:
            self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
            self.bias = bias_initializer.initialize((self.numberKernels), self.numberKernels, 1)
        else:
            unref = weights_initializer.initialize(self.weights.shape, self.input_size, 1)
            self.weights = np.reshape(unref, (self.numberKernels, self.convolutionShape[0], self.convolutionShape[1]))
            self.bias = bias_initializer.initialize((self.numberKernels), self.numberKernels, 1)

    def backward(self, error_tensor):

        if self.twoDim:
            # Gradient with respect to weights
            self.output_size = self.weights.shape

            self.gradient_weights = np.zeros(self.weights.shape)
            self.gradient_bias = np.zeros(self.numberKernels)

            for image in range(error_tensor.shape[0]):
                for kernel in range(self.gradient_weights.shape[0]):
                    for kernel_channel in range(self.gradient_weights.shape[1]):
                        padded_img = np.pad(self.input_tensor[image, kernel_channel, :, :],
                                            math.ceil(self.weights.shape[3] / 2))
                        err = error_tensor[image, kernel, :, :]
                        if self.strideShape[0] > 1 or self.strideShape[1] > 1:
                            right_size = np.zeros((error_tensor.shape[2] * self.strideShape[0],
                                                   error_tensor.shape[3] * self.strideShape[1]))
                            right_size[::self.strideShape[0], ::self.strideShape[1]] = error_tensor[image, kernel, :,
                                                                                         :]
                            err = right_size

                            #
                        newKernel = signal.correlate(
                            padded_img, err,
                            mode='valid')

                        half_h = (newKernel.shape[0] - self.weights.shape[2]) / 2
                        half_v = (newKernel.shape[1] - self.weights.shape[3]) / 2
                        # take the middle element of the corresponding kernal layer
                        newKernel = newKernel[math.ceil(half_h):newKernel.shape[0] - math.floor(half_h),
                                    math.ceil(half_v):newKernel.shape[1] - math.floor(half_v)]
                        self.gradient_weights[kernel, kernel_channel] = self.gradient_weights[
                                                                            kernel, kernel_channel] + newKernel

            #gradient with respect to bias
            res = np.zeros(self.numberKernels)
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
                        empty_right_size[::self.strideShape[0], ::self.strideShape[1]] = error_tensor[image,
                                                                                           kernel_channel, :, :]
                        newKernel = signal.convolve(
                            empty_right_size, new_kernel[kernel_channel, :, :],
                            mode='same')
                        gradient_layer[image, channel] = gradient_layer[image, channel] + newKernel
            if self.HaveOptimizer:
                 opt_w, opt_b = self.optimizer
                 self.weights = opt_w.calculate_update(self.weights, self.gradient_weights)
                 self.bias = opt_b.calculate_update(self.bias, self.gradient_bias)

            return gradient_layer
        else:
            # Gradient with respect to weights
            self.output_size = self.weights.shape
            self.gradient_weights = np.zeros(self.weights.shape)
            self.gradient_bias = np.zeros(self.numberKernels)
            for image in range(error_tensor.shape[0]):
                for kernel in range(self.gradient_weights.shape[0]):
                    for kernel_channel in range(self.gradient_weights.shape[1]):
                        padded_img = np.pad(self.input_tensor[image, kernel_channel, :],
                                            math.ceil(self.weights.shape[2] / 2))
                        err = error_tensor[image, kernel, :]
                        if self.strideShape[0] > 1 or self.strideShape[1] > 1:
                            right_size = np.zeros((error_tensor.shape[2] * self.strideShape[0]))
                            right_size[::self.strideShape[0]] = error_tensor[image, kernel, :]
                            err = right_size

                        newKernel = signal.correlate(
                            padded_img, err,
                            mode='valid')

                        half_h = (newKernel.shape[0] - self.weights.shape[2]) / 2
                        newKernel = newKernel[math.ceil(half_h):newKernel.shape[0] - math.floor(half_h)]
                        self.gradient_weights[kernel, kernel_channel] = self.gradient_weights[
                                                                            kernel, kernel_channel] + newKernel
            res = np.zeros(self.numberKernels)
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
                        empty_right_size[::self.strideShape[0]] = error_tensor[image, kernel_channel, :]

                        newKernel = signal.convolve(
                            empty_right_size, new_kernel[kernel_channel, :],
                            mode='same')
                        gradient_layer[image, channel] = gradient_layer[image, channel] + newKernel

            if self.HaveOptimizer:
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
            self.HaveOptimizer = True
            import copy
            self.__weights_optimizer = copy.deepcopy(optimizer)
            self.__bias_optimizer = optimizer
        else:
            self.__weights_optimizer = optimizer
            self.__bias_optimizer = optimizer

    @property
    def gradient_weights(self):
        return self.__gradient_weights



    @property
    def gradient_bias(self):
        return self.__gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self.__gradient_bias = gradient_bias

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.__gradient_weights = gradient_weights
