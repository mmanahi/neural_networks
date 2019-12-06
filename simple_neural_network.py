# simple neural network just for the sack of explaination
import numpy as np

def sigmiod(x):
    return 1.0/ (1 + np.exp(-x))

def sigmiod_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        print('inputs \n', self.input)
        print()
        self.weights1 = np.random.rand(self.input.shape[1],4)
        print('weights1 \n', self.weights1)
        print()
        self.weights2 = np.random.rand(4,1)
        print('weights2 \n', self.weights2)
        print()
        self.y = y
        print('y \n', self.y)
        print()
        self.output = np.zeros(self.y.shape)
        print('output \n', self.output)
        print()

    def feedforward(self):
        self.layer1 = sigmiod(np.dot(self.input, self.weights1)) 
        self.output = sigmiod(np.dot(self.layer1, self.weights2))

    def backpropagation(self):
        d_weights1 = np.dot(self.layer1.T,(2*(self.y - self.output) * sigmiod_derivative(self.output)))
        d_weights2 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmiod_derivative(self.output), self.weights2.T) * sigmiod_derivative(self.layer1)))
        self.weights1 += d_weights1
        self.weights2 += d_weights2

X = np.array([[0, 0, 1],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1]])

y = np.array([[0],
             [1],
             [1],
             [0]])
nn = NeuralNetwork(X, y)

for i in range(100):
    nn.feedforward()
    nn.backpropagation()
print(nn.output)
