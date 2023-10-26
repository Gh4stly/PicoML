import random
import math

#Largely copied over from Data Science from Scratch

def add(v,w):
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v, w):
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors):
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

def scalar_multiply(c,v):
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v, w):
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"
    #print(v)
    #print(w)
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
def partial_difference_quotient(f,v,i,h):
        """Returns the i-th partial difference quotient of f at v"""
        w = [v_j + (h if j == i else 0)    # add h to just the ith element of v
             for j, v_j in enumerate(v)]
    
        return (f(w) - f(v)) / h
    
def estimate_gradient(f,v,h = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]

def gradient_step(v, gradient, step_size):
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    # weights includes the bias term, inputs includes a 1
    #print(inputs)
    #print(weights)
    return sigmoid(dot(weights, inputs))
 
def sqerror_gradients(network,
                      input_vector,
                      target_vector):
    """
    Given a neural network, an input vector, and a target vector,
    make a prediction and compute the gradient of the squared error
    loss with respect to the neuron weights.
    """
    # forward pass
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # gradients with respect to output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]

    # gradients with respect to output neuron weights
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    # gradients with respect to hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                         dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]

def create_layers(size,num_layers):
    """
    Create an array of random weights
    size = number of weights in each layer
    """
    network = []
    for i in range(num_layers):
        network.append([random.random() for _ in range(size + 1)])
    return network  

def training_network(xs,ys,network,rounds = 20000,learning_rate = 1.0):    
    for epoch in range(rounds):
            for x, y in zip(xs, ys):
                gradients = sqerror_gradients(network, x, y)
    
            # Take a gradient step for each neuron in each layer
                network = [[gradient_step(neuron, grad, -learning_rate)
                            for neuron, grad in zip(layer, layer_grad)]
                           for layer, layer_grad in zip(network, gradients)]
    return network

def data_split(data,percent):
    """
    Split data based on percentage for testing/training networks
    """
    training = data[:int(len(data) * percent)]
    test = data[-int(len(data) * (1 - percent)):]
    return training,test 

def feed_forward(neural_network,
                 input_vector):
    """
    Feeds the input vector through the neural network.
    Returns the outputs of all layers (not just the last one).
    """
    outputs = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]              # Add a constant.
        output = [neuron_output(neuron, input_with_bias)  # Compute the output
                  for neuron in layer]                    # for each neuron.
        outputs.append(output)                            # Add to results.

        # Then the input to the next layer is the output of this one
        input_vector = output

    return outputs

def test_neuralnet():
    """
    A quick test to make sure we can train the NN
    We're training it on learning XOR since its fast, simple,
    and if something goes wrong we can troubleshoot easy
    """
    full_network = []
    hidden_layer = create_layers(2,2)
    output_layer = create_layers(2,1)
    full_network.append(hidden_layer)
    full_network.append(output_layer)
    #test data
    xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
    ys = [[0.], [1.], [1.], [0.]]
    network = training_network(xs,ys,full_network)
    #lets make sure everything is working
    print("Running assert block")
    assert feed_forward(network, [0, 0])[-1][0] < 0.01
    assert feed_forward(network, [0, 1])[-1][0] > 0.99
    assert feed_forward(network, [1, 0])[-1][0] > 0.99
    assert feed_forward(network, [1, 1])[-1][0] < 0.01
    print("Test complete!")

    