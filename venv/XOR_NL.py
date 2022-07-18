import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)
l = 3
bias = 1
graph_set = {
    "x_axis" : [],
    "y_axis" : []
}
NL = {
    "layer" : [None] * l,
    "l_error" : [None] * l,
    "l_delta" : [None] * l,
    "synapse" : [None] * (l - 1)
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def train(inputs, outputs, generations=4096):
    inputs = np.column_stack((inputs, [bias for i in range(len(inputs))]))
    NL["synapse"][0] = 2 * np.random.random((inputs.T).shape) - 1
    NL["synapse"][1] = 2 * np.random.random(outputs.shape) - 1
    NL["layer"][0] = inputs

    for i in range(generations):
        # Layers initialization
        for j in range(1, l):
            NL["layer"][j] = sigmoid(np.dot(NL["layer"][j - 1], NL["synapse"][j - 1]))

        # Calculation of layer error and deviation level
        NL["l_error"][l - 1] = outputs - NL["layer"][l - 1]
        NL["l_delta"][l - 1] = NL["l_error"][l - 1] * d_sigmoid(NL["layer"][l - 1])
        for j in range(1, l - 1):
            NL["l_error"][l - 1 - j] = np.dot(NL["l_delta"][l - j], NL["synapse"][l - 1 - j].T)
            NL["l_delta"][l - 1 - j] = NL["l_error"][l - 1 - j] * d_sigmoid(NL["layer"][l - 1 - j])

        # Adjusting synopses based on deviation level
        for j in range(l - 1):
            NL["synapse"][l - 2 - j] += np.dot(NL["layer"][l - 2 - j].T, NL["l_delta"][l - 1 - j])

        if i % 1000 == 0:
            graph_set["x_axis"].append(i / 1000)
            graph_set["y_axis"].append(np.mean(abs(NL["l_error"][l - 1])))

    return NL["synapse"]

def show_neural_network():
    print("L0:\n", NL["layer"][0])
    print("SYNAPCES_01:\n", NL["synapse"][0])
    print("L1:\n", NL["layer"][1])
    print("SYNAPCES_12:\n", NL["synapse"][1])
    print("L2:\n", NL["layer"][2])

def run(layer_input, synapses):
    layer_input = np.column_stack((layer_input, [bias for i in range(len(layer_input))]))
    for data in layer_input:
        layer_hidden = sigmoid(np.dot(data, synapses[0]))
        layer_out = sigmoid(np.dot(layer_hidden, synapses[1]))
        print("XOR: ", data, " = ", layer_out)

def MSE_grapf():
    plt.plot(graph_set["x_axis"], graph_set["y_axis"], c="black", linewidth=4, marker="o", markersize=6, label="MSE")
    plt.legend()
    plt.show()
