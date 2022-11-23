from .Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.last_forward = None



    def forward(self,input_tensor):
        res = np.zeros((input_tensor.shape[0],input_tensor.shape[1]))

        for row in range (input_tensor.shape[0]):
           for col in range(input_tensor.shape[1]):
               if(input_tensor[row][col] > 0):
                   res[row][col] = input_tensor[row][col]
        self.last_forward = res
        return res

    def backward(self,error_tensor):
        res = np.zeros((error_tensor.shape[0],error_tensor.shape[1]))
        print(np.shape(error_tensor))
        print(np.shape(self.last_forward))
        for row in range (error_tensor.shape[0]):
           for col in range(error_tensor.shape[1]):
               if(self.last_forward[row][col] > 0):
                   res[row][col] = error_tensor[row][col]
        return res