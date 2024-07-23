# Artificial Neural Networks

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Part 1: Neural Network Library](#part-1-neural-network-library)
    - [Running instructions](#running-instructions)
3. [Part 2: House Value Regression](#part-2-house-value-regression)
    - [Running instructions](#running-instructions-1)

## Repository Structure

```
Neural_Networks_60012_001
├── README.md
├── housing.csv
├── images
│   ├── households.png
│   ├── housing_median_age.png
│   ├── latitude.png
│   ├── latitude_vs_Value.png
│   ├── longitude.png
│   ├── longitude_vs_Value.png
│   ├── longitude_vs_latitude.png
│   ├── median_house_value.png
│   ├── median_income.png
│   ├── population.png
│   ├── total_bedrooms.png
│   └── total_rooms.png
├── iris.dat
├── part1_nn_lib.py
├── part2_house_value_regression.py
├── part2_model.pickle
├── requirements.txt
└── tests
    ├── test_part1.py
    └── test_part2.py
```

- `README.md`: This file, describes the build process and repository structure.
- `housing.csv`: Contains the dataset for testing of part 2's implementation. Processed by `pandas`.
- `images`: Some images used to find correlations between the datapoints in the dataset.
- `iris.dat`: Contains the dataset for testing of part 1's implementation.
- `part1_nn_lib.py`: Part 1, neural network library written using `NumPy`.
- `part2_house_value_regression.py`: Part 2, regression model constructed using `PyTorch` and `SciKit Learn`.
- `part2_model.pickle`: Pickle file containing the model for Part 2's implementation.
- `requirements.txt`: Libraries used to set-up the virtual environment.
- `tests`: Tests folder, integrated into the gitlab pipeline for CI/CD.

## Part 1: Neural Network Library

Part 1 focused on implementing a neural network library using `NumPy`. 

### Running instructions

`example_main()` contains an implementation of a test program that can be ran to verify the functioning of the code. It uses the `iris.dat` dataset.
```Python
python3 part1_nn_lib.py
```

## Part 2: House Value Regression

Part 2 focused on implementing a regression model on the `housing.csv` dataset with `pandas` for predicting the median house value of a block group.

### Running instructions

`example_main()` contains an implementation of a test program that can be ran to verify the functioning of the code.
```Python
python3 part2_house_value_regression.py
```
