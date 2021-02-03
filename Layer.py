import numpy as np

class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.rand(output_dim, input_dim) - 0.5

        self.activation = activation()

        self.freeze = False

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, input)
        self.activated_output = self.activation.activate(self.output)
        return self.activated_output

    def backward(self, next_error, lr):
        previous_layer_error = np.dot(self.weights.T, next_error)
        if(not self.freeze):
            self.weights -= lr * np.dot(self.activation.differentiate() * next_error, self.input.T)
        return previous_layer_error