from preprocessing import normalize, loadCSV
from Layer import Layer
from Function import Sigmoid, Relu
from NN import NN

INPUT_DIM = 784
HIDDEN_DIM = 400
OUTPUT_DIM = 10
TRAIN_PATH = 'D:\datasets\mnist\large_dataset\mnist_train.csv'
TEST_PATH = 'D:\datasets\mnist\large_dataset\mnist_test.csv'
LR = 0.1

print('Initialize the model')
layers = [
    Layer(INPUT_DIM, HIDDEN_DIM, Sigmoid),
    Layer(HIDDEN_DIM, OUTPUT_DIM, Sigmoid)
]
ann = NN(layers)

print('Load and preprocess the dataset')
train_inputs, train_targets = loadCSV(TRAIN_PATH)
train_inputs = normalize(train_inputs)
test_inputs, test_targets = loadCSV(TEST_PATH)
test_inputs = normalize(test_inputs)

print('Start the training')
ann.fit(train_inputs, train_targets, test_inputs, test_targets, lr=LR, epochs=6)
