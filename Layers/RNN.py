import numpy as np

from .Base import BaseLayer
from . import FullyConnected
from . import TanH
from . import Sigmoid

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):

        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memorize = False
        self.hidden_state = np.zeros((1, self.hidden_size))
        self.hidden_error = np.zeros((1, self.hidden_size))
        self.layers = []
        self.layers_initialized = False
        self.initalizerw = None
        self.initalizerb = None
        self.optimizer = None
        self.input_dense_one = []
        self.activations_tanh = []
        self.input_dense_two = []
        self.activations_sigmoid = []
        self.gradient_weights = 0
        self.gradient_weights_two = 0
        # Creating internal layers
        self.layers.append(FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size))
        self.layers.append(TanH.TanH())
        self.layers.append(FullyConnected.FullyConnected(self.hidden_size, self.output_size))
        self.layers.append(Sigmoid.Sigmoid())
        self.hidden_error = np.zeros((1, self.hidden_size))
        super().__init__()

    def initialize(self, weights_initializer, bias_initializer):
        self.initalizerw = weights_initializer
        self.initalizerb = bias_initializer
        self.initialize_layers()

    def initialize_layers(self):
        if self.initalizerw is not None or self.initalizerb is not None:
            self.layers[0].initialize(self.initalizerw,self.initalizerb)
            self.layers[2].initialize(self.initalizerw, self.initalizerb)

    def forward(self, input_tensor):
        if not self.memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))
        self.input_tensor = input_tensor
        result_array = np.zeros((input_tensor.shape[0], self.output_size))
        self.input_dense_one = []
        self.activations_tanh = []
        self.input_dense_two = []
        self.activations_sigmoid = []
        for batch_element in range(input_tensor.shape[0]):
            inp = np.reshape(input_tensor[batch_element], (1, input_tensor[batch_element].shape[0]))
            conc_input = np.concatenate((inp, self.hidden_state),axis=1)
            self.hidden_state = self.layers[1].forward(self.layers[0].forward(conc_input))
            result_array[batch_element] = self.layers[3].forward(self.layers[2].forward(self.hidden_state))

            self.input_dense_one.append(self.layers[0].input_tensor)
            self.activations_tanh.append(self.layers[1].activations)
            self.input_dense_two.append(self.layers[2].input_tensor)
            self.activations_sigmoid.append(self.layers[3].activations)

        return result_array

    def backward(self, error_tensor):
        new_error_tensor = np.zeros(self.input_tensor.shape)
        self.gradient_weights = 0
        self.gradient_weights_two = 0
        self.hidden_error = np.zeros((1, self.hidden_size))

        for batch_element in reversed(range(new_error_tensor.shape[0])):
            inp = np.reshape(error_tensor[batch_element], (1, error_tensor[batch_element].shape[0]))
            # Adding correct layer activations to the internal layers
            self.layers[3].activations = self.activations_sigmoid[batch_element]
            self.layers[2].input_tensor = self.input_dense_two[batch_element]
            self.layers[1].activations = self.activations_tanh[batch_element]
            self.layers[0].input_tensor = self.input_dense_one[batch_element]

            # Calculating internal error -> copy in forward becomes sum in backwards
            internal_gradient = self.layers[2].backward(self.layers[3].backward(inp)) + self.hidden_error
            internal_lower_gradient = self.layers[0].backward(self.layers[1].backward(internal_gradient))

            # Splitting the internal error in two parts to undo conc
            new_error_tensor[batch_element] = internal_lower_gradient[0, :self.input_tensor.shape[1]]
            self.hidden_error = internal_lower_gradient[0, self.input_tensor.shape[1]:]

            self.gradient_weights = self.gradient_weights + self.layers[0].gradient_weights
            self.gradient_weights_two = self.gradient_weights_two + self.layers[2].gradient_weights

        # Final update step with sum of gradient weights
        if self.optimizer is not None:
            self.layers[0].weights = self.optimizer.calculate_update(self.layers[0].weights, self.gradient_weights)
            self.layers[2].weights = self.optimizer.calculate_update(self.layers[2].weights, self.gradient_weights_two)

        return new_error_tensor

    @property
    def memorize(self):
        return self.__memorize

    @memorize.setter
    def memorize(self, memorize):
        self.__memorize = memorize

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer

    @property
    def weights(self):
        return self.layers[0].weights

    @weights.setter
    def weights(self, weights):
        self.layers[0].weights = weights


