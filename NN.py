from Function import Sigmoid, MeanSquare
import numpy as np

class NN:
    """
        The NN class. Defines an Artificial neural network.

        Args:
            layers_list: a list of Layer objects.
    """
    def __init__(self, layers_list):
        self.layers = layers_list
        self.error = 1
    
    def forward(self, input):
        """
            The forward method. Does a forward pass throughout all the ANN layers by iteratively calling the forward method in each element the layers_list.

            Args:
                input: the input vectors.

            Returns:
                returns the ANN output which is the same as the output of the last layer.
        """
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        self.output = output
        return output
    
    def backward(self, target, cost_function, lr, freeze=False):
        """
            The backward method. Does a full backward pass by iteratively calling the backward method in the layers_list.
            
            Args:
                target: the target vector. (not an index)
                cost_function: the cost function to be used to calculate the ANN cost at each training iteration.
                lr: the learning rate.
                freeze: a boolean value whether to freeze the network or not (be freezing the network we mean no weight update should happen). Default: False
            
            Returns:
                returns the error of the whole network.
        """
        self.error = cost_function.activate(self.output, target)
        error = cost_function.differentiate()
        for layer in reversed(self.layers):
            layer.freeze = freeze
            error = layer.backward(error, lr=lr)
        return self.error

    def one_sample(self, input, target, cost_function=MeanSquare(), lr=0.1, freeze=False):
        """
            the one_sample method. Does one training iteration by calling the forward then the backward methods.

            Args:
                input: the input vector.
                target: the target vector.
                cost_function: the cost function to be used by the ANN (an object no a class). Default: MeanSquare().
                lr: the learning rate. Default: 0.1.
                freeze: whether to freeze the network or not, will passed backward(). Default: False
            
            Returns:
                returns a tuple of the ANN output and error respectively.

        """
        output = self.forward(input)
        error = self.backward(target, cost_function, lr, freeze=freeze)
        return (output, error)
    
    def fit(self, train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix, cost_function=MeanSquare(), lr=0.1, epochs=1):
        """
            The fit method. Fits the network given the number of epochs.

            Args:
                train_input_matrix: a list (or iterable) of training input vectors.
                train_target_matrix: a list (or iterable) of training target vectors.
                test_input_matrix: a list (or iterable) of test input vectors.
                test_target_matrix: a list (or iterable) of test target vectors.
                cost_function: the cost function to be use in training. Default: MeanSquare().
                lr: the learning rate. Default: 0.1.
                epoch: the number of epochs. Default: 1.
            
            Returns:
                returns the results of the training and test. a dictionary with two keys: 'train', 'test'.
                Each key maps to a list of dictionairies, each dictionairy has the keys: 'lr', 'epoch', 'error', 'accuracy', 'output', 'target'
        """
        last_test_accuracy = 0
        for epoch in range(epochs):
            print('-------------------- Epoch: ', epoch + 1, '--------------------')
            hits = 0
            results = {'train': [], 'test': []}
            for i in range(len(train_input_matrix)):
                input = train_input_matrix[i]
                target = train_target_matrix[i]
                output_error = self.one_sample(input, target, cost_function, lr)
                if(np.argmax(output_error[0]) == np.argmax(target)):
                    hits += 1
                if(i % 600 == 0):
                    print('------TRAINING:------ \nEpoch: ', epoch + 1, ', Accuracy: ', hits/(i + 1), ', Last Test Accuracy: ', last_test_accuracy, ', Error: ', np.sum(output_error[1])/10)
                    print('Output: ', np.argmax(output_error[0]), 'Target: ', np.argmax(target))
                    results['train'].append({
                        'lr': lr,
                        'epoch': epoch + 1,
                        'accuracy': hits/(i + 1),
                        'error': np.sum(output_error[1]/10),
                        'output': np.argmax(output_error[0]),
                        'target': np.argmax(target)
                    })
            
            hits = 0
            for i in range(len(test_input_matrix)):
                input = test_input_matrix[i]
                target = test_target_matrix[i]
                output, error = self.one_sample(input, target, freeze=True)
                if(np.argmax(output) == np.argmax(target)):
                    hits += 1
                if(i % 100 == 0):
                    print('------TEST:------ \nEpoch: ', epoch + 1, ', Accuracy: ', hits/(i + 1))
                    print('Output: ', np.argmax(output), 'Target: ', np.argmax(target))
                    results['test'].append({
                        'lr': lr,
                        'epoch': epoch + 1,
                        'accuracy': hits/(i + 1),
                        'error': np.sum(error/10),
                        'output': np.argmax(output),
                        'target': np.argmax(target)
                    })
                last_test_accuracy = hits/(i + 1)
        return results