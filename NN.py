from Function import Sigmoid, MeanSquare
import numpy as np

class NN:
    def __init__(self, layers_list):
        self.layers = layers_list
        self.error = 1
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        self.output = output
        return output
    
    def backward(self, target, cost_function, lr, freeze=False):
        self.error = cost_function.activate(self.output, target)
        error = cost_function.differentiate()
        for layer in reversed(self.layers):
            layer.freeze = freeze
            error = layer.backward(error, lr=lr)
        return self.error

    def one_sample(self, input, target, cost_function=MeanSquare(), lr=0.1, freeze=False):
        output = self.forward(input)
        error = self.backward(target, cost_function, lr, freeze=freeze)
        return (output, error)
    
    def fit(self, train_input_matrix, train_target_vector, test_input_matrix, test_target_vector, cost_function=MeanSquare(), lr=0.1, epochs=1):
        last_test_accuracy = 0
        for epoch in range(epochs):
            print('-------------------- Epoch: ', epoch + 1, '--------------------')
            hits = 0
            results = {'train': [], 'test': []}
            for i in range(len(train_input_matrix)):
                input = train_input_matrix[i]
                target = train_target_vector[i]
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
                target = test_target_vector[i]
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