# neural network to recognize handwritten numbers
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

# number_values = data_list[0].split(',') # each number refers to label i.e. here number 0 refers to label 5 (number 5)
# image_array = np.asfarray(number_values[1:]).reshape((28,28)) # ignore the first value (number lable) and split the others and convert image to an 28*28 array
# plt.imshow(image_array , cmap='Greys', interpolation='None')
# plt.show()

# scaled_input = (np.asfarray(number_values[1:]) / 255.0 * 0.99) + 0.01 # to get all numbers 0.01 to 1.00 
# print(scaled_input)

# Y = np.zeros(output_layer) + 0.01 # array of zeros to avoid zeros effect with weights and its len is 10
# Y[int(number_values[0])] = 0.99 # number values here refers to the label in which the represented number is written. 
# print("predicted output:", Y)

class NeuralNetwork:
    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.learning_rate = learning_rate
        # weights for propagation from input to hidden layer
        self.Wih = np.random.normal(0.0, pow(self.hidden_layer, -0.5), (self.hidden_layer, self.input_layer))
        # weights for propagation from  hidden to output layer
        self.Who = np.random.normal(0.0, pow(self.output_layer, -0.5), (self.output_layer, self.hidden_layer))

        self.activation_function = lambda x: scipy.special.expit(x) # sigmoid activation function

    def train(self, X, Y):
        # convert input and desired output to 2d arrays
        X_array = np.array(X, ndmin = 2).T
        Y_array = np.array(Y, ndmin = 2).T
        # feedforwared propagation from input to hidden layer
        Zih = np.dot(self.Wih, X_array)
        Aih = self.activation_function(Zih)
        # feedforwared propagation from hidden to output layer
        Zho = np.dot(self.Who, Aih)
        Aho = self.activation_function(Zho)
        # calculate error rate with delta 
        delta_ho = Y_array - Aho  
        delta_ih = np.dot(self.Who.T, delta_ho)
        # backward propagation to update weights
        self.Who += self.learning_rate * np.dot((delta_ho * Aho * (1.0 - Aho)), np.transpose(Aih))
        self.Wih += self.learning_rate * np.dot((delta_ih * Aih * (1.0 -  Aih)), np.transpose(X_array))

    def query(self, X):
        X_array = np.array(X, ndmin=2).T

        Zih = np.dot(self.Wih, X_array)
        Aih = self.activation_function(Zih)

        Zho = np.dot(self.Who, Aih)
        Aho = self.activation_function(Zho)
        return Aho


# design of neural network is for classification, 10 output neurons represent 10 possible numbers, firing first neuron means the label is zero and so on.
input_layer = 784 # multiplication of 28*28 array that represents the lables
hidden_layer = 100 # 100 neurons for the hidden layer
output_layer = 10 # 10 possible outputs, thus 10 neuron  represent labels (0-9)
learning_rate = 0.3 # learning rate is set to 0.3

dataset = open("dataset/mnist_train_100.csv", 'r')
data_list = dataset.readlines()
dataset .close()
print(len (data_list))
print(data_list[0])

n = NeuralNetwork(input_layer, hidden_layer, output_layer, learning_rate)
for record in data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # scale and shift the inputs
    X = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create the target output values (all 0.01, except the desired label which is 0.99)
    Y = np.zeros(output_layer) + 0.01
    Y[int(all_values[0])] = 0.99
    n.train(X, Y)


test_dataset = open("dataset/mnist_test_10.csv", 'r')
test_data_list = test_dataset.readlines()
test_dataset.close()

all_values = record.split(',')
print(all_values[0])
image_array = np.asfarray(all_values[1:]).reshape((28,28)) # ignore the first value (number lable) and split the others and convert image to an 28*28 array
plt.imshow(image_array , cmap='Greys', interpolation='None')
plt.show()
print(n.query((np.asfarray(all_values[1:])/255.0 *0.99)))
