import numpy as np

class Sgd:
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - (self.learning_rate*gradient_tensor)
        #slope, intercept = np.polyfit(weight_tensor,gradient_tensor,1)
        #slope = -self.learning_rate * slope
        return np.array(weight_tensor)

