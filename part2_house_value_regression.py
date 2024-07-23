import torch
import pickle
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from numpy.random import default_rng
from sklearn.metrics import d2_pinball_score, d2_tweedie_score, explained_variance_score, mean_squared_error, r2_score, mean_absolute_percentage_error


class Network(nn.Module):
    def __init__(self, input_dim, neurons, activations):
        super(Network, self).__init__()
        self.module_list = nn.ModuleList()
        self.neurons = neurons
        self.activations = activations

        for layer in neurons:
            net = nn.Linear(input_dim, layer)
            self.module_list.append(net)
            input_dim = layer

    def forward(self, x):
        for layer, activation in zip(self.module_list, self.activations):
            # Module_list contain the layers of nnet of type torch.nn.Linear
            # Activations contain the function object of type torch.nn.functional or None indicating identity
            if(x.dtype != torch.float32):
                x = x.to(torch.float32)
            if (activation==None):
                x = layer(x)
            else:
                x = activation(layer(x))
        return x


class Regressor:

    def __init__(self, x, neurons=[64,16,1], activations=[torch.tanh, torch.relu, None], nb_epoch=100, learning_rate=0.001, optimiser=torch.optim.Adam, loss=nn.MSELoss, batch_size = 20, normalization='min-max'):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        # Label binarizer for each column in the dataframe, store for validation and testing
        self._lb = {}
        # Store parameters to normalize the data, apply the same scale for validation and testing
        # as training
        self._normalization_params = {}
        self._normalization = normalization
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        #check last two lines added new
        
        # self.evaulation_method = ['explained_variance', 'max_error', 'neg_mean_absolute_error',\
        # 'neg_mean_squared_log_error', 'r2', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance',\
        # 'neg_mean_absolute_percentage_error', 'd2_pinball_score', 'd2_tweedie_score']#
        self.evaluation_method = {
            'd2_pinball_score': d2_pinball_score,
            'd2_tweedie_score': d2_tweedie_score,
            'explained_variance': explained_variance_score,
            'root_mean_squared_error': mean_squared_error,
            'r2_score': r2_score,
            'mean_absolute_percentage_error': mean_absolute_percentage_error
        }

        # Creating network and its optimiser and loss function
        self.network = Network(self.input_size, neurons, activations)
        self.optimiser = optimiser(self.network.parameters(), lr=learning_rate)
        self.loss = loss()


    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """
        # Implement mode imputation, replace N/As
        x = x.fillna(value={column: x[column].mode()[0] for column in x.columns})
        if y is not None:
            y = y.fillna(value={column: y[column].mode()[0] for column in y.columns})

        def min_max_normalization(pre_x, pre_y):
            post_x, post_y = None, None

            if training:
                self._normalization_params['min_values_x'] = torch.min(pre_x, dim=0).values
                self._normalization_params['max_values_x'] = torch.max(pre_x, dim=0).values

                if pre_y is not None:  # For fit method
                    pre_y = torch.tensor(pre_y.values)
                    self._normalization_params['min_values_y'] = torch.min(pre_y, dim=0).values
                    self._normalization_params['max_values_y'] = torch.max(pre_y, dim=0).values

            # Perform normalization of input and output data using min-max scaling
            post_x = (pre_x - self._normalization_params['min_values_x']) / \
                     (self._normalization_params['max_values_x'] - self._normalization_params['min_values_x'])

            if pre_y is not None:  # For fit and score
                if isinstance(pre_y, pd.DataFrame):  # For the predict and score methods only
                    pre_y = torch.tensor(pre_y.values)
                if isinstance(pre_y, torch.Tensor):  # Sanitization of input parameters, might be tested?
                    post_y = (pre_y - self._normalization_params['min_values_y']) / \
                             (self._normalization_params['max_values_y'] - self._normalization_params['min_values_y'])

            return post_x, post_y

        def z_score_normalization(pre_x, pre_y):
            post_x, post_y = None, None

            if training:
                self._normalization_params['mean_x'] = torch.mean(pre_x, dim=0)
                self._normalization_params['std_x'] = torch.std(pre_x, dim=0)

                if pre_y is not None:
                    pre_y = torch.tensor(pre_y.values)
                    self._normalization_params['mean_y'] = torch.mean(pre_y, dim=0)
                    self._normalization_params['std_y'] = torch.std(pre_y, dim=0)

            post_x = (pre_x - self._normalization_params['mean_x']) / self._normalization_params['std_x']

            if pre_y is not None:
                if isinstance(pre_y, pd.DataFrame):
                    pre_y = torch.tensor(pre_y.values)
                if isinstance(pre_y, torch.Tensor):
                    post_y = (pre_y - self._normalization_params['mean_y']) / \
                             self._normalization_params['std_y']

            return post_x, post_y

        if training:
            # Check if each column is entirely text and if so, one-hot encode it
            for i, column in enumerate(x.columns):
                # Check if any value in the column is text
                if any(x[column].apply(lambda val: isinstance(val, str))):
                    # Initialize the label binarizer if not already created
                    if column not in self._lb:
                        # Store the label binarizers based on columns for later
                        self._lb[(i, column)] = LabelBinarizer()

        # General code for training, testing, validation
        for key in self._lb:  # Need separate loop so that x is not mutated mid-loop
            i, column = key
            # One-hot encode the text column
            one_hot_encoded = self._lb[key].fit_transform(x[column])

            # Replace the original column with one-hot encoded values while preserving column order
            one_hot_columns = [f"{column}_{j}" for j in range(one_hot_encoded.shape[1])]
            x.reset_index(drop=True, inplace=True)
            x = pd.concat([x.iloc[:, :i], pd.DataFrame(one_hot_encoded, columns=one_hot_columns),
                           x.iloc[:, i + 1:]], axis=1, ignore_index=True)
        # Perform preferred normalization
        if self._normalization == 'z-score':
            x, y = z_score_normalization(torch.tensor(x.values), y)
        else:
            x, y = min_max_normalization(torch.tensor(x.values), y)
        # Return preprocessed x and y, return None for y if it was None
        return x, (y if isinstance(y, torch.Tensor) else None)
    
    def _reverse_preprocessing(self, y):
        """
        Reverses preprocessing for output - this is only reversing the min-max normalization
        because the one-hot encoding is on the input.

        Arguments: 
            - y {torch.tensor} -- Tensor containing the initial value of y 
            to be reversed

        Returns:
            - {torch.tensor} -- Tensor containing the original value of y 
        """ 
       # Reverse min-max normalization
        if self._normalization == 'z-score':
            return (y * self._normalization_params['std_y']) + self._normalization_params['mean_y']

        return y * (self._normalization_params['max_values_y'] - self._normalization_params['min_values_y']) + self._normalization_params['min_values_y']

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns:
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """

        random_generator = default_rng()
        n_instances = len(input_dataset)
        shuffled_indices = random_generator.permutation(n_instances)
        shuffled_input = input_dataset[shuffled_indices]
        shuffled_target = target_dataset[shuffled_indices]

        return shuffled_input, shuffled_target

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
        
        X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)

        callback_queue = []
        valid_out = self.network(X_valid)
        valid_loss = self.loss(valid_out, y_valid.to(torch.float32))
        prev_loss = valid_loss.item()
        min_delta = 0.0001   # Default 0
        patience = 5    # Default 5

        for epoch in range(self.nb_epoch):
            X, Y = self.shuffle(X, Y)

            for input_batch, target_batch in zip(torch.split(X, self.batch_size), torch.split(Y, self.batch_size)):

                # set model into training
                self.network.train()

                # Zero the gradients
                self.optimiser.zero_grad()

                # Forward pass
                outputs = self.network(input_batch)
                
                # Compute loss
                loss_obj = self.loss(outputs, target_batch.to(torch.float32))
                
                # Backward pass
                loss_obj.backward()

                # Update network parameters
                self.optimiser.step()

            valid_out = self.network(X_valid)
            valid_loss = self.loss(valid_out, y_valid.to(torch.float32))
                
            if prev_loss + min_delta < valid_loss.item():
                callback_queue.append(1)
            else:
                callback_queue.append(0)
            if len(callback_queue) > patience:
                callback_queue.pop(0)
            if 0 not in callback_queue:
                print("Early Stopping")
                break
            prev_loss = valid_loss.item()

            # Print the current loss after each epoch
            print(f'Epoch {epoch+1}/{self.nb_epoch}, Loss: {valid_loss.item()}')

        return self

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        X, _ = self._preprocessor(x, training=False)  
        predict = self.network(X)
        predict = self._reverse_preprocessing(predict)  # Reverse pre-processing to reduce RMSE
        output = predict.detach().numpy()

        return output

    def score(self, x, y, metrics='root_mean_squared_error'):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
    
        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget

        if metrics in self.evaluation_method:
            evaluation_method = self.evaluation_method[metrics]
        else:
            raise ValueError(f"Unsupported evaluation metric: {metrics}")
        Y = self._reverse_preprocessing(Y)  # Reversed preprocessing in fit, so need to reverse here too.
        return evaluation_method(Y, self.predict(x))
    

def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(X, y):
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    from skorch import NeuralNetRegressor
    from skorch.callbacks import EarlyStopping

    regressor = Regressor(X)
    X, y = regressor._preprocessor(X, y, training=True)
    
    input_dim = len(X[0])

    if(X.dtype != torch.float32):
        X = X.to(torch.float32)
    if(y.dtype != torch.float32):
        y = y.to(torch.float32)

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # Use skorch's NeuralNetRegressor with GridSearchCV
    net = NeuralNetRegressor(
        Network,
        module__input_dim=input_dim,
        module__neurons=[64, 1],
        module__activations=[torch.relu, torch.relu],
        max_epochs=10,
        lr=0.001,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss,
        iterator_train__shuffle=True,
        callbacks=[('early_stopping', EarlyStopping())]
    )

    # Need to set param_grid to right parameters
    param_grid_1 = {
        'module__neurons': [[64,16,8,1],[64,32,16,1],[64,32,8,1]],
        'module__activations': [[torch.relu, torch.tanh, torch.relu, None]],
        'optimizer': [torch.optim.Adam],
        'lr': [0.001],
        'max_epochs': [100],
        'batch_size': [10,20,30],
    }

    param_grid_2 = {
        'module__neurons': [[64,16,1]],
        'module__activations': [[torch.tanh, torch.relu, None]],
        'optimizer': [torch.optim.Adam],
        'lr': [0.001],
        'max_epochs': [100],
        'batch_size': [30],
    }

    grid_search_regressor_1 = GridSearchCV(net, param_grid_1, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_search_regressor_2 = GridSearchCV(net, param_grid_2, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_search_regressor_1.fit(X_train, y_train)
    grid_search_regressor_2.fit(X_train, y_train)

    # Print the best parameters and the corresponding mean squared error
    print("Best parameters found: ", grid_search_regressor_1.best_params_, grid_search_regressor_2.best_params_)
    print("Best cross-validation negative mean squared error: {},{}".format(grid_search_regressor_1.best_score_, grid_search_regressor_2.best_score_))

    # Evaluate on the validation set using the best model
    y_pred_1 = torch.tensor(grid_search_regressor_1.best_estimator_.predict(X_valid))
    y_pred_2 = torch.tensor(grid_search_regressor_2.best_estimator_.predict(X_valid))

    mse_1 = F.mse_loss(regressor._reverse_preprocessing(y_pred_1), y_valid)
    mse_2 = F.mse_loss(regressor._reverse_preprocessing(y_pred_2), y_valid)
    print("Validation mean squared error using the best model: {},{}".format(mse_1, mse_2))

    return grid_search_regressor_2.best_params_



def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    evaluation_metric = 'root_mean_squared_error'
    # Error
    error = regressor.score(x_train, y_train, evaluation_metric)
    print("\nRegressor error: {}\n".format(error))

    # RegressorHyperParameterSearch(x_train, y_train)


if __name__ == "__main__":
    example_main()
