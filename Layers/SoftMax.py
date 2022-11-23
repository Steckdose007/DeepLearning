from .Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.last_forward = None

    def forward(self,input_tensor):
        res = []
        for row in range (input_tensor.shape[0]):
            max_value = max(input_tensor[row])
            x_k = input_tensor[row] - max_value
            sum = np.sum(np.exp(x_k))
            if(sum == 0):
                res.append(np.zeros(input_tensor.shape[1]))
            else:
                res.append(np.divide(np.exp(x_k),sum))
        self.last_forward = res
        return np.array(res)

    def backward(self,error_tensor):
        res = []
        for row in range (error_tensor.shape[0]):
            res.append(self.last_forward[row] * (error_tensor[row] - np.sum(error_tensor[row]*self.last_forward[row]))) # Frage ist y hat j das gleiche wie y hat
        return np.array(res)
