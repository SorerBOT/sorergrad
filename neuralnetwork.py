import random

class Neuron:
    def __init__(self, num_of_inputs):
        self.weights = [Single(random.uniform(-1, 1)) for i in range(num_of_inputs)]
        self.bias = Single(random.uniform(-1, 1))
    def __call__(self, x):
        activision = sum((weight * input for weight, input in zip(self.weights, x)), self.bias)
        # passing activision through non-linearity, chose tanh because sigmoid was too slow ??
        output = activision.tanh()
        
        return output
    def parameters(self):
        return self.weights + [self.bias]
class Layer:
    def __init__(self, num_of_inputs, num_of_neurons):
        self.neurons = [Neuron(num_of_inputs) for i in range(num_of_neurons)]
    def __call__(self, x):
        output = [n(x) for n in self.neurons]
        return output[0] if len(output) == 1 else output
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
class MLP:
    def __init__(self, num_of_inputs, num_of_neurons_list):
        sizes = [num_of_inputs] + num_of_neurons_list
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(num_of_neurons_list))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]