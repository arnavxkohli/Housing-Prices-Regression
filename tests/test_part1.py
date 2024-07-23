import pytest
from part1_nn_lib import *


# ------------------------ SIGMOID CLASS ---------------------------

@pytest.fixture
def sigmoid_layer():
    return SigmoidLayer()


@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 4)])
def test_sigmoid_forward(sigmoid_layer, shape):
    x = np.random.rand(*shape)
    result = sigmoid_layer.forward(x)
    expected_result = 1 / (1 + np.exp(-x))
    np.testing.assert_array_almost_equal(result, expected_result)


@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 4)])
def test_sigmoid_backward(sigmoid_layer, shape):
    x = np.random.rand(*shape)
    grad_z = np.random.rand(*shape)
    expected_result = 1 / (1 + np.exp(-x))
    sigmoid_layer.forward(x)
    grad = sigmoid_layer.backward(grad_z)
    expected_grad = expected_result * (1 - expected_result) * grad_z
    np.testing.assert_array_almost_equal(grad, expected_grad)


# -------------------------- RELU CLASS ----------------------------

@pytest.fixture
def relu_layer():
    return ReluLayer()


@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 4)])
def test_relu_forward(relu_layer, shape):
    x = np.random.rand(*shape)
    result = relu_layer.forward(x)
    expected_result = np.maximum(0, x)
    np.testing.assert_array_almost_equal(result, expected_result)


@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 4)])
def test_relu_backward(relu_layer, shape):
    grad_z = np.random.rand(*shape)
    relu_layer.forward(np.random.rand(*shape))
    grad = relu_layer.backward(grad_z)
    expected_grad = np.where(relu_layer._cache_current > 0, grad_z, 0)
    np.testing.assert_array_almost_equal(grad, expected_grad)


# ------------------------- LINEAR CLASS ---------------------------

@pytest.mark.parametrize("n_in, n_out, batch_size", [(5, 3, 20), (10, 5, 100), (3, 2, 10)])
def test_linear_forward(n_in, n_out, batch_size):
    linear_layer = LinearLayer(n_in, n_out)

    x = np.random.rand(batch_size, n_in)

    expected_z = (x @ linear_layer._W) + linear_layer._b

    z = linear_layer.forward(x)

    np.testing.assert_array_equal(z, expected_z)
    np.testing.assert_array_equal(x, linear_layer._cache_current)


@pytest.mark.parametrize("n_in, n_out, batch_size", [(5, 3, 20), (10, 5, 100), (3, 2, 10)])
def test_linear_backward(n_in, n_out, batch_size):
    linear_layer = LinearLayer(n_in, n_out)
    x = np.random.rand(batch_size, n_in)
    linear_layer.forward(x)

    grad_z = np.random.rand(batch_size, n_out)

    expected_grad_x = grad_z @ np.transpose(linear_layer._W)
    expected_grad_W = np.transpose(linear_layer._cache_current) @ grad_z
    expected_grad_b = np.mean(grad_z, axis=0)

    grad_x = linear_layer.backward(grad_z)
    grad_W = linear_layer._grad_W_current
    grad_b = linear_layer._grad_b_current

    np.testing.assert_array_equal(grad_x, expected_grad_x)
    np.testing.assert_array_equal(grad_W, expected_grad_W)
    np.testing.assert_array_equal(grad_b, expected_grad_b.reshape(*grad_b.shape))


@pytest.mark.parametrize("n_in, n_out, batch_size, learning_rate", [(5, 3, 20, 0.1), (10, 5, 100, 0.5),
                                                                    (3, 2, 10, 0.3)])
def test_linear_update_params(n_in, n_out, batch_size, learning_rate):
    linear_layer = LinearLayer(n_in, n_out)
    x = np.random.rand(batch_size, n_in)
    grad_z = np.random.rand(batch_size, n_out)
    linear_layer.forward(x)
    linear_layer.backward(grad_z)

    expected_W = linear_layer._W - learning_rate * linear_layer._grad_W_current
    expected_b = linear_layer._b - learning_rate * linear_layer._grad_b_current

    linear_layer.update_params(learning_rate)

    np.testing.assert_array_equal(linear_layer._W, expected_W)
    np.testing.assert_array_equal(linear_layer._b, expected_b)


# --------------------- PRE-PROCESSOR CLASS ------------------------

@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 4)])
def test_pre_processor(shape):
    data = np.random.rand(*shape)
    pre_processor = Preprocessor(data)
    applied = pre_processor.apply(data)
    np.testing.assert_array_almost_equal(pre_processor.revert(applied), data)

# --------------------- MULTILAYER-NETWORK CLASS ------------------------
@pytest.fixture
def activations():
    return ['relu', 'sigmoid']

@pytest.fixture
def initialized_network(input_dim, neurons, activations):
    network = MultiLayerNetwork(input_dim, neurons, activations)
    return network

@pytest.mark.parametrize("input_dim, neurons, batch_size", [(10, [5, 4], 100), (5, [3, 2], 20), (3, [2, 3], 10)])
def test_multi_forward(input_dim, neurons, batch_size, initialized_network):
    x = np.random.rand(batch_size, input_dim)
    output = initialized_network.forward(x)
    assert output.shape == (batch_size, neurons[-1])

@pytest.mark.parametrize("input_dim, neurons, batch_size", [(10, [5, 4], 100), (5, [3, 2], 20), (3, [2, 3], 10)])
def test_multi_backward(input_dim, neurons, batch_size, initialized_network):
    x = np.random.rand(batch_size, input_dim)
    output = initialized_network.forward(x)
    grad_z = np.random.rand(batch_size, neurons[-1])
    grad_input = initialized_network.backward(grad_z)
    assert grad_input.shape == (batch_size, input_dim)

@pytest.mark.parametrize("input_dim, neurons, batch_size, loss, learning_rate", [(10, [5, 4], 100, MSELossLayer(), 0.03), 
                                                                (5, [3, 2], 20, MSELossLayer(), 0.1), 
                                                                (3, [2, 3], 10, CrossEntropyLossLayer(), 0.5)])
def test_multi_update_params(input_dim, neurons, batch_size, loss, learning_rate, initialized_network):
    x = np.random.rand(batch_size, input_dim)
    prediction = initialized_network.forward(x)
    target = np.zeros(prediction.shape)
    initial_loss = loss.forward(prediction, target)
    grad_z = loss.backward()
    grad_input = initialized_network.backward(grad_z)

    initialized_network.update_params(learning_rate)

    prediction = initialized_network.forward(x)
    final_loss = loss.forward(prediction, target)
    assert final_loss <= initial_loss
