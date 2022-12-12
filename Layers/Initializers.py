import numpy as np
import math

class Constant:
    def __init__(self, const_weight:float=0.1):
        self.fan_in = None
        self.fan_out = None
        self.const_weight = const_weight

    def initialize(self, dimension_weight, fan_in, fan_out):
        return np.reshape(np.full((dimension_weight), self.const_weight), dimension_weight)



class UniformRandom:
    def __init__(self):
        self.fan_in = None
        self.fan_out = None

    def initialize(self, dimension_weight, fan_in, fan_out):
        return np.reshape(np.random.uniform(0.0, 1.0, (dimension_weight)), dimension_weight)

class Xavier:
    def __init__(self):
        self.fan_in = None
        self.fan_out = None

    def initialize(self, dimension_weight, fan_in, fan_out):
        return np.reshape(np.random.normal(0.0, math.sqrt(2/(fan_in+fan_out)), dimension_weight), weights_shape)

class He:
    def __init__(self):
        self.fan_in = None
        self.fan_out = None

    def initialize(self, dimension_weight, fan_in, fan_out):
        return np.reshape(np.random.normal(0.0, math.sqrt(2/fan_in), dimension_weight), dimension_weight)
