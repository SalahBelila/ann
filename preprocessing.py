import numpy as np

def normalize(vector, norm_factor=255):
    return (vector/norm_factor)

def loadCSV(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    inputs = []
    targets = []
    for line in lines:
        line = line.split(',')
        target_index = int(line[0])
        input = np.asfarray(line[1:])
        input = input.reshape((len(input), 1))
        target = np.zeros((10, 1))
        target[target_index][0] = 1.00
        targets.append(target)

        inputs.append(input)
    
    return (np.asfarray(inputs), np.asfarray(targets))
