# Module_21_Deep-Learning-Challenge

This repository contains Jupyter Notebook files and Python code for analyzing and predicting the success of Alphabet Soup-funded organizations using neural networks.

## Step 1: Preprocess the Data
1) Start by running the preprocess_data.ipynb notebook, which contains the following steps:
* Read the charity_data.csv file into a Pandas DataFrame and identify the target variable(s) and feature(s) for the model.
* Drop the EIN and NAME columns as they are not relevant for the model.
* Determine the number of unique values for each column and, for columns with more than 10 unique values, analyze the number of data points for each unique value.
* Bin "rare" categorical variables together into a new value, Other, based on the number of data points.
* Encode categorical variables using pd.get_dummies().
* Split the preprocessed data into a features array, X, and a target array, y, and further divide the data into training and testing datasets using train_test_split.
* Scale the training and testing features datasets using StandardScaler from scikit-learn.

## Step 2: Compile, Train, and Evaluate the Model
Open the neural_network_model.ipynb notebook to design and train a neural network model for binary classification.
Create a neural network model using TensorFlow and Keras by specifying the number of input features and nodes for each layer.
Add hidden layers with appropriate activation functions to improve the model's performance.
Create an output layer with an appropriate activation function for binary classification.
Compile and train the model using the training data and evaluate its performance on the testing data.
Implement a callback that saves the model's weights every five epochs for future reference.
Calculate the model's loss and accuracy on the test data.
Save the trained model to an HDF5 file named AlphabetSoupCharity.h5.

## Step 3: Optimize the Model
Create a new Jupyter Notebook named AlphabetSoupCharity_Optimisation.ipynb.
Import the required dependencies and read the charity_data.csv file into a Pandas DataFrame.
Preprocess the dataset as done in Step 1, making adjustments based on any modifications needed from the optimization process.
Design a neural network model with adjustments to achieve a target accuracy higher than 75%.
Train and evaluate the optimized model.
Save the optimized model to an HDF5 file named AlphabetSoupCharity_Optimisation.h5.

## Step 4: Write a Report on the Neural Network Model
