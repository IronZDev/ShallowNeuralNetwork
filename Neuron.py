import numpy as np


# noinspection PyMethodMayBeStatic
class Neuron:
    def __init__(self, activation='heaviside', learning_rate=0.01, inputs_number=3):
        # seeding for random number generation
        np.random.seed(1)
        self.weights = np.random.uniform(0, 1, inputs_number)
        self.learning_rate = learning_rate
        self.activation_function = getattr(self, activation)
        self.input = []
        self.adjustment = []
        self.error = []
        self.calculated = []

    def sigmoid(self, x, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-x))
        else:
            return self.sigmoid(x) * (1 - self.sigmoid(x))

    def heaviside(self, x, derivative=False):
        if not derivative:
            return np.heaviside(x, 1)
        else:  # Derivative mode
            return 1

    def sin(self, x, derivative=False):
        if not derivative:
            return np.sin(x)
        else:
            return np.cos(x)

    def cos(self, x, derivative=False):
        if not derivative:
            return np.cos(x)
        else:
            return -np.sin(x)

    def tanh(self, x, derivative=False):
        if not derivative:
            return np.tanh(x)
        else:
            return 1 - np.tanh(x)**2

    def sign(self, x, derivative=False):
        if not derivative:
            if x < 0:
                return -1
            elif x > 0:
                return 1
            else:
                return 0
        else:
            return 1

    def relu(self, x, derivative=False):
        if not derivative:
            if x > 0:
                return x
            else:
                return 0
        else:
            if x > 0:
                return 1
            else:
                return 0

    def leaky_relu(self, x, derivative=False):
        if not derivative:
            if x > 0:
                return x
            else:
                return 0.01 * x
        else:
            if x > 0:
                return 1
            else:
                return 0.01

    def train(self, training_inputs, training_outputs):
        for input_val, output in zip(training_inputs, training_outputs):
            calculated = self.predict(self.weights @ input_val)
            error = output - calculated
            adjustments = self.learning_rate * error * self.activation_function(self.weights @ input_val, True) * input_val
            self.weights += adjustments

    def predict(self, ins):
        self.input = ins
        ins = self.weights @ ins
        # passing the inputs via the neuron to get output
        ins = ins.astype(float)
        self.calculated = self.activation_function(ins)
        return self.calculated

    def calculate_basic_adjustment(self, target):  # For the last layer
        self.error = target - self.calculated
        self.adjustment = self.learning_rate * self.error * self.activation_function(self.weights @ self.input, True) * self.input

    def calculate_adjustment(self, upper_neurons, neuron_number):  # For hidden layers
        self.error = 0
        for neuron in upper_neurons:
            self.error += neuron.weights[neuron_number] * neuron.activation_function(neuron.weights @ neuron.input) * neuron.error
        self.adjustment = self.learning_rate * self.error * self.activation_function(self.weights @ self.input, True) * self.input

    def apply_changes(self):
        self.weights += self.adjustment
