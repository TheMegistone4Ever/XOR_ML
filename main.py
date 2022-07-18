from XOR_NL import *
import time
start_time = time.time()
training_inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

training_outputs = np.array([[0, 1, 1, 0, 1, 0, 0, 1]]).T
generations = 20000

train(training_inputs, training_outputs, generations)
run([[0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]])
show_neural_network()
print("time elapsed: {:.2f}s".format(time.time() - start_time))
MSE_grapf()
