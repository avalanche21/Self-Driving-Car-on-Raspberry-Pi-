import numpy as np
import sys


def sigmoid(Z):
    """
    Compute the sigmoid of Z.

    Arguments:
    Z -- Output of the linear layer, of any shape.

    Return:
    A -- Post-activation parameter, of the same shape as Z.
    """
    A = 1.0 / (1.0 + np.exp(-Z))
    assert(A.shape == Z.shape)

    return A


def relu(Z):
    """
    Implement the Relu function.

    Arguments:
    Z -- Output of the linear layer, of any shape.

    Returns:
    A -- Post-activation parameter, of the same shape as Z.
    """
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)

    return A


def tanh(Z):
    """
    Implement the tanh function.

    Arguments:
    Z -- Output of the linear layer, of any shape.

    Returns:
    A -- Post-activation parameter, of the same shape as Z.
    """
    A = np.tanh(Z)
    assert(A.shape == Z.shape)

    return A


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(2)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from ***previous*** layer (or input data).
    W -- weights matrix.
    b -- bias vector.

    A.shape == (size of previous layer, number of examples).
    W.shape == (size of current layer, size of previous layer).
    b.shape == (size of the current layer, 1).

    Returns:
    Z -- the input of the activation function, the "pre-activation parameter".
    """

    Z = np.dot(W, A_prev) + b
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))

    return Z


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    A_prev -- activations from previous layer (or input data):
    W -- weights matrix.
    b -- bias vector.
    activation -- the activation function: "sigmoid", "tanh" or "relu".

    A_prev.shape == (size of previous layer, number of examples).
    W.shape == (size of current layer, size of previous layer).
    b.shape == (size of the current layer, 1).

    Returns:
    A -- the post-activation value, which is the output of activation function.
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing backward propagation.

    # A = []
    # linear_cache = []
    # activation_cache = []

    # Inputs: "A[l-1]", "W[l]", b[l]".
    # Outputs: "A[l]", "activation_cache".
    # activation_cache == (linear_cache, activation_cache).
    # linear_cache == (A[l-1], W[l], b[l]).
    # activation_cache == Z[l].
    """

    assert(isinstance(activation, str))

    if activation == "sigmoid":
        Z = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
    elif activation == "relu":
        Z = linear_forward(A_prev, W, b)
        A = relu(Z)
    elif activation == "tanh":
        Z = linear_forward(A_prev, W, b)
        A = tanh(Z)
    else:
        print("invalid activation function!")
        sys.exit(1)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    return A


def L_model_forward(X, parameters):
    """
    Implement forward propagation.
    Our L-layer model is: [LINEAR->RELU/TANH]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data.
    X.shape == (input size, number of examples).
    parameters -- output of initialize_parameters_deep().

    Returns:
    AL -- the post-activation value from output(Lth) layer.
    """

    A = X
    L = len(parameters) // 2

    # Propagate through the first L-1 layers.
    for l in range(1, L):
        A_prev = A
        A = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "sigmoid")

    # Propagate through the output layer.
    AL = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")

    return AL


def predict(X, parameters):
    """
    Using the learned parameters, to predict a class for each $x\in X$.

    Arguments:
    parameters -- python dictionary containing your parameters.
    X -- input data of size (n_x, m).

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict based on probabilities obtained by forward propagation.
    AL = L_model_forward(X, parameters)

    return AL


def compute_cost(AL, Y):
    """
    Implement the cost function.

    Arguments:
    AL -- probability vector corresponding to your label predictions.
    Y -- true "label" vector.

    AL.shape == (1, number of examples).

    Returns:
    cost -- cross-entropy cost.
    """

    assert(AL.shape == Y.shape)
    m = Y.shape[1]

    # logprobs = loss matrix.
    logprobs = Y * np.log(AL) + (1.0 - Y) * np.log(1.0 - AL)
    cost = - (1.0 / m) * np.sum(logprobs, axis=None)
    cost = np.squeeze(cost)
    assert(isinstance(cost, float))

    return cost


def evaluate(AL, test_data):
    predict_data = np.argmax(AL, axis=0)
    assert(predict_data.shape == (AL.shape[1], ))
    compare_results = list(zip(predict_data, test_data))

    return sum(int(x == y) for (x, y) in compare_results)
