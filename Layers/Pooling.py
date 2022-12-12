import numpy as np
import math

class Pooling():
    def __init__(self,stride_shape,pooling_shape):
        super().__init__()
        self.trainable = False
        self.input_shape = 0
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        S = self.stride_shape
        x_pooling = self.pooling_shape[0]
        y_pooling = self.pooling_shape[1]
        y_size = 1 + math.floor((self.input_tensor.shape[2]-x_pooling)/S[0])
        x_size = 1 + math.floor((self.input_tensor.shape[3] - y_pooling) / S[1])
        out= np.zeros((self.input_tensor.shape[0],self.input_tensor.shape[1],y_size,x_size))

        for image in range(self.input_tensor.shape[0]):
            for channel in range(self.input_tensor.shape[1]):
                for y in range(y_size):
                    for x in range(x_size):
                        out[image,channel,y,x] = np.max(input_tensor[image,channel,y*S[0]:y*S[0]+x_pooling,x*S[1]:x*S[1]+y_pooling])
        return out

    def backward(self, error_tensor):
        output_size = self.input_tensor.shape
        output = np.zeros(output_size)
        S = self.stride_shape
        x_pooling = self.pooling_shape[0]
        y_pooling = self.pooling_shape[1]
        y_size = 1 + math.floor((self.input_tensor.shape[2]-x_pooling)/S[0])
        x_size = 1 + math.floor((self.input_tensor.shape[3] - y_pooling) / S[1])

        for image in range(self.input_tensor.shape[0]):
            for channel in range(self.input_tensor.shape[1]):
                for y in range(y_size):
                    for x in range(x_size):
                        curr_slice=self.input_tensor[image, channel, y * S[0]:y * S[0] + x_pooling, x * S[1]:x * S[1] + y_pooling]
                        index = np.unravel_index(curr_slice.argmax(), curr_slice.shape)
                        output[image,channel,y*S[0]+index[0],x*S[1]+index[1]] =output[image,channel,y*S[0]+index[0],x*S[1]+index[1]]+error_tensor[image,channel,y,x]

        return output
