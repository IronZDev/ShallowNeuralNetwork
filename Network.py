from Neuron import Neuron
import numpy as np


class Layer:
    def __init__(self, activation='heaviside', learning_rate=0.01, neurons_in_hidden_layer=2, inputs_number=3):
        self.neurons = []
        self.output = []
        for neuron in range(neurons_in_hidden_layer):
            self.neurons.append(Neuron(activation, learning_rate, inputs_number))

    def predict(self, input_data):
        self.output = []
        for neuron in self.neurons:
            self.output.append(neuron.predict(input_data))
        return np.asarray(self.output)

    def calculate_errors_deltas(self, upper_layer):
        for neuron_num, neuron in enumerate(self.neurons, start=0):
            neuron.calculate_adjustment(upper_layer.neurons, neuron_num)

    def apply_changes(self):
        for neuron in self.neurons:
            neuron.apply_changes()


class Network:
    def __init__(self, activation='heaviside', learning_rate=0.01, hidden_layers_size=2, neurons_in_hidden_layer=2):
        self.layers = []
        for hidden_layer in range(hidden_layers_size):
            if hidden_layer == 0:  # For first layer the input number is fixed
                self.layers.append(Layer(activation, learning_rate, neurons_in_hidden_layer, 3))
            else:
                self.layers.append(Layer(activation, learning_rate, neurons_in_hidden_layer, neurons_in_hidden_layer))
        self.layers.append(Layer(activation, learning_rate, 1, neurons_in_hidden_layer))  # Add one output neuron

    def train(self, training_inputs, training_outputs, training_iterations):
        for _ in range(training_iterations):
            for input_val, output in zip(training_inputs, training_outputs):
                current_input = input_val
                for layer in self.layers:
                    current_input = layer.predict(current_input)

                #  Calculate errors for last layer
                self.layers[len(self.layers) - 1].neurons[0].calculate_basic_adjustment(output)

                #  Calculate errors fot the rest
                for i in range(len(self.layers) - 2, -1, -1):
                    self.layers[i].calculate_errors_deltas(self.layers[i + 1])

                #  Apply changes
                for layer in self.layers:
                    layer.apply_changes()

    def predict(self, input):
        current_input = input
        for layer in self.layers:
            current_input = layer.predict(current_input)
        return current_input
