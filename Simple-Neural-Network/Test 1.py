#https://www.youtube.com/watch?v=kft1AJ9WVDk
#https://www.3blue1brown.com/lessons/neural-networks
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
H=.995
L=.005
training_inputs = np.array([[0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1]])
print('training inputs: ')
print(training_inputs)
##output is 1 if 2nd digit is 1
training_outputs = np.array([[0,0,1,1,0,0,1,1]]).T 
print('training outputs: ')
print(training_outputs)

np.random.seed(5)

synaptic_weights = 2 * np.random.random((3,1)) - 1
bias = 2 * np.random.random((1,1)) - 1
print('bias:')
print(bias)
print('Random Synaptic Starting Weight: ')
print(synaptic_weights)

y = 1
print(y)

'''

for iterration in range(100):  
    
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights)+bias)
    
    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)
    bias += np.sum(adjustments, axis=0, keepdims=True)

'''
input_layer = training_inputs
outputs = sigmoid(np.dot(input_layer, synaptic_weights)+bias)
    
error = training_outputs - outputs
adjustments = error * sigmoid_derivative(outputs)
synaptic_weights += np.dot(input_layer.T, adjustments)
bias += np.sum(adjustments, axis=0, keepdims=True)
while  (H > outputs[0] > L) or (H > outputs[1] > L) or (H > outputs[2] > L) or (H > outputs[3] > L) or (H > outputs[4] > L) or (H > outputs[5] > L) or (H > outputs[6] > L) or (H > outputs[7] > L) : 
    
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights)+bias)
    
    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)
    bias += np.sum(adjustments, axis=0, keepdims=True)
    y +=1

print('Synaptic weights after training: ')
print(synaptic_weights)    
print('Outputs after training: ')
roundedoutputs = np.around(outputs,5)
print(roundedoutputs)

print('iterations:')
print(y)
