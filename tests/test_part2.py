import pytest
from part2_house_value_regression import *

# --------------------------------- NETWORK ------------------------------------

@pytest.fixture
def simple_network():
    # Create an instance of the Network class with a simple configuration
    return Network(input_dim=10, neurons=[5,3], activations=[F.relu, F.relu])

def test_forward_pass(simple_network):
    # Input data for testing
    input_data = torch.randn((1, 10))

    # Ensure that the forward pass runs without errors
    output = simple_network(input_data)
    assert isinstance(output, torch.Tensor)

def test_activation_functions(simple_network):
    # Input data for testing
    input_data = torch.randn((1, 10))

    # Check if the correct activation functions are applied
    output = simple_network(input_data)
    
    assert torch.all(output >= 0)  # Check ReLU activation

def test_module_list_length(simple_network):
    # Check if the length of the module list matches the number of layers
    assert len(simple_network.module_list) == len(simple_network.activations) == len(simple_network.neurons)



# --------------------------------- PREPROCESSOR  ------------------------------------

@pytest.mark.parametrize("input_x, input_y, expected_output_x, expected_output_y", [
    # Test case 1
    (pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]}),
     pd.DataFrame({'Target': [0.1, 0.2, 0.3]}),
     torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
     torch.tensor([[0.0], [0.5], [1.0]])),

    # Test case 2
    (pd.DataFrame({'Feature1': [4, 5, 6], 'Feature2': [7, 8, 9]}),
     pd.DataFrame({'Target': [0.4, 0.5, 0.6]}),
     torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
     torch.tensor([[0.4], [0.5], [0.6]])),
])
def test_regressor_preprocessor(input_x, input_y, expected_output_x, expected_output_y):
    regressor = Regressor(input_x)

    # Test retention of normalized values
    training_x, training_y = regressor._preprocessor(input_x, input_y, training=True)
    testing_x, testing_y = regressor._preprocessor(input_x, input_y, training=False)

    # Tolerance for floating-point comparisons
    tol = 1e-6

    assert torch.allclose(training_x, testing_x, atol=tol)
    assert torch.allclose(training_y, testing_y, atol=tol)


# -------------------------------------- Fit ----------------------------------------

@pytest.fixture
def sample_data():
    # Create sample data for testing
    x = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.DataFrame({'target': [10, 20, 30]})
    return x, y

def test_fit_training(sample_data):
    x, y = sample_data
    regressor = Regressor(x, nb_epoch=3)
    trained_regressor = regressor.fit(x, y)

    # Add assertions based on your requirements
    assert isinstance(trained_regressor, Regressor)


# ------------------------- RegressorHyperParameterSearch  --------------------------

from sklearn.datasets import make_regression

@pytest.fixture
def read_data():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    return x_train, y_train

def test_regressor_hyperparameter_search(read_data):
    X_train, y_train = read_data

    # Call the function with synthetic data
    best_params = RegressorHyperParameterSearch(X_train, y_train)

    # Ensure that the returned parameters are of the expected types
    assert isinstance(best_params, dict)
