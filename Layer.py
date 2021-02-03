import numpy as np

class Layer:
    """
        The Layer class. Defines a Layer given the input and output dimensions and the activation function.

        Args:
            input_dim: the dim the input.
            output_dim: the dimension of the output.
            activation: the activation function (a class not an object).
    """
    def __init__(self, input_dim, output_dim, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.rand(output_dim, input_dim) - 0.5

        self.activation = activation()

        self.freeze = False

    def forward(self, input):
        """
            The forward method. Does one forward pass for the layer and returns the output.

            Args:
                input: the input vector.

            Returns:
                returns a vector of dimension output_dim (the output dimension given to the Layer constructor).
        """
        self.input = input
        self.output = np.dot(self.weights, input)
        self.activated_output = self.activation.activate(self.output)
        return self.activated_output

    def backward(self, next_error, lr):
        """
            The backward method. Does one backward pass for the layer and returns the output.

            Args:
                next_error: the error of the next layer.
                lr: the learning rate.

            Returns:
                returns the error of the layer.
        """
        previous_layer_error = np.dot(self.weights.T, next_error)
        if(not self.freeze):
            self.weights -= lr * np.dot(self.activation.differentiate() * next_error, self.input.T)
        return previous_layer_error