import numpy as np
class CrossEntropyLoss():
    def __init__(self):
        self.preds = None
    def forward(self,prediction_tensor,label_tensor):
        self.preds = prediction_tensor

        return np.sum(-np.log(prediction_tensor[label_tensor == 1] + np.finfo(float).eps))

    def backward(self,label_tensor):
        return- np.divide(label_tensor,(self.preds + np.finfo(float).eps))