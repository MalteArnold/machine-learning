import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, no_input_nodes, no_hidden_nodes, no_output_nodes, L, bias):
        """
        Initialize the neural network with random weights.

        Parameters:
        - no_input_nodes (int): Number of input nodes.
        - no_hidden_nodes (int): Number of hidden nodes.
        - no_output_nodes (int): Number of output nodes.
        - L (float): Learning rate.
        - bias (float): Bias term.
        """
        self.no_input_nodes = no_input_nodes
        self.no_hidden_nodes = no_hidden_nodes
        self.no_output_nodes = no_output_nodes
        self.L = L
        self.bias = bias
        self.wih = 2 * np.random.random((no_hidden_nodes, no_input_nodes)) - 1
        self.who = 2 * np.random.random((no_output_nodes, no_hidden_nodes)) - 1

    def train(self, training_images, training_labels):
        """
        Train the neural network using backpropagation.

        Parameters:
        - training_images (numpy.ndarray): Input training data.
        - training_labels (numpy.ndarray): Target labels for training data.
        """
        for i in range(len(training_images)):
            input = np.array(training_images[i], ndmin=2).T
            output_expected = np.array(training_labels[i], ndmin=2).T

            # Forwardpropagation
            weighted_sum_hidden = np.dot(self.wih, input) + self.bias
            output_hidden = sigmoid(weighted_sum_hidden)

            weighted_sum_output = np.dot(self.who, output_hidden)
            output_output = sigmoid(weighted_sum_output)

            error = (1 / 2) * (np.power((output_expected - output_output), 2))

            # Backpropagation
            d_error_d_output_output = output_output - output_expected
            d_output_output_d_weightedsum_output = sigmoid_der(output_output)
            d_weightedsum_output_d_who = np.array(output_hidden, ndmin=2).T
            delta_who = np.dot(
                d_error_d_output_output * d_output_output_d_weightedsum_output,
                d_weightedsum_output_d_who,
            )

            d_error_d_output_hidden = np.dot(
                np.array(self.who, ndmin=2).T,
                d_error_d_output_output * d_output_output_d_weightedsum_output,
            )
            d_output_hidden_d_weightedsum_hidden = sigmoid_der(output_hidden)
            d_weightedsum_hidden_d_wih = np.array(input, ndmin=2).T
            delta_wih = np.dot(
                d_error_d_output_hidden * d_output_hidden_d_weightedsum_hidden,
                d_weightedsum_hidden_d_wih,
            )

            # Update weights
            self.who = self.who - self.L * delta_who
            self.wih = self.wih - self.L * delta_wih

    def evaluate(self, data, labels):
        """
        Generate the confusion matrix and print it.

        Parameters:
        - data (numpy.ndarray): Input data.
        - labels (numpy.ndarray): True labels for the input data.

        Returns:
        - numpy.ndarray: Confusion matrix.
        """
        confusion_matrix = np.zeros((self.no_output_nodes, self.no_output_nodes), int)

        for i in range(len(data)):
            test_result = self.test(data[i])
            test_result_max = test_result.argmax()
            target_label = labels[i].argmax()
            confusion_matrix[test_result_max, target_label] += 1

        print("Confusion matrix:\n", confusion_matrix)
        return confusion_matrix

    def test(self, input):
        """
        Forward propagate the input and return the computed output.

        Parameters:
        - input (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Computed output.
        """
        input = np.array(input, ndmin=2).T

        weighted_sum_hidden = np.dot(self.wih, input) + self.bias
        output_hidden = sigmoid(weighted_sum_hidden)

        weighted_sum_output = np.dot(self.who, output_hidden)
        output_output = sigmoid(weighted_sum_output)

        return output_output

    def measurements(self, confusion_matrix):
        """
        Print common neural network measures (accuracy, recall, precision).

        Parameters:
        - confusion_matrix (numpy.ndarray): Confusion matrix.
        """
        no_correct, no_incorrect, accuracy = 0, 0, 0
        for i in range(self.no_output_nodes):
            no_correct += confusion_matrix[i][i]
            for j in range(self.no_output_nodes):
                if i != j and confusion_matrix[i][j] != 0:
                    no_incorrect += confusion_matrix[i][j]

        for i in range(self.no_output_nodes):
            print(
                "digit: ",
                i,
                "precision: ",
                self.precision(i, confusion_matrix),
                "recall: ",
                self.recall(i, confusion_matrix),
            )

        accuracy = no_correct / (no_correct + no_incorrect)

        return no_correct, no_incorrect, accuracy

    def recall(self, label, confusion_matrix):
        """
        Calculate recall for a specific label.

        Parameters:
        - label (int): Label index.
        - confusion_matrix (numpy.ndarray): Confusion matrix.

        Returns:
        - float: Recall value.
        """
        column = confusion_matrix[:, label]
        sum = column.sum()
        if sum != 0:
            return confusion_matrix[label, label] / sum
        else:
            return 0

    def precision(self, label, confusion_matrix):
        """
        Calculate precision for a specific label.

        Parameters:
        - label (int): Label index.
        - confusion_matrix (numpy.ndarray): Confusion matrix.

        Returns:
        - float: Precision value.
        """
        row = confusion_matrix[label, :]
        sum = row.sum()
        if sum != 0:
            return confusion_matrix[label, label] / sum
        else:
            return 0

def main():
    """
    Main function to execute the neural network on the MNIST dataset.
    """
    # Number of inputs and output neurons
    no_inputs = 28 * 28
    no_outputs = 10

    training_inputs = np.loadtxt("mnist_train_1000.csv", delimiter=",")
    test_inputs = np.loadtxt("mnist_test_100.csv", delimiter=",")

    # Extract the labels from the input
    training_outputs_tmp = np.asfarray(training_inputs[:, 0]).astype(int)
    test_outputs_tmp = np.asfarray(test_inputs[:, 0]).astype(int)

    # Transform the label into a representation
    training_outputs = np.zeros((len(training_outputs_tmp), no_outputs)) + 0.01
    for row in range(training_outputs_tmp.size):
        value = training_outputs_tmp[row]
        training_outputs[row][value] = 0.99

    test_outputs = np.zeros((len(test_outputs_tmp), no_outputs)) + 0.01
    for row in range(test_outputs_tmp.size):
        value = test_outputs_tmp[row]
        test_outputs[row][value] = 0.99

    training_inputs = np.asfarray(training_inputs[:, 1:]) / 255 * 0.98 + 0.01
    test_inputs = np.asfarray(test_inputs[:, 1:]) / 255 * 0.98 + 0.01

    epochs = 100
    hidden_nodes = 100
    L = 0.1
    bias = 0.1

    neural_network = NeuralNetwork(no_inputs, hidden_nodes, no_outputs, L, bias)

    for epoch in range(epochs):
        neural_network.train(training_inputs, training_outputs)

    print("TRAINING SET")
    confusion_matrix_training = neural_network.evaluate(training_inputs, training_outputs)
    correct, incorrect, accuracy = neural_network.measurements(confusion_matrix_training)
    print("Accuracy: ", accuracy)
    print("Number of correct instances: ", correct)
    print("Number of incorrect instances: ", incorrect)

    print("TESTING SET")
    confusion_matrix = neural_network.evaluate(test_inputs, test_outputs)
    correct, incorrect, accuracy = neural_network.measurements(confusion_matrix)
    print("Accuracy: ", accuracy)
    print("Number of correct instances: ", correct)
    print("Number of incorrect instances: ", incorrect)

if __name__ == "__main__":
    main()
