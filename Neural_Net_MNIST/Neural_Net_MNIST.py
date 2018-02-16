import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

from sklearn.datasets import load_digits                # Import MNIST dataset using sklearn

from sklearn.preprocessing import StandardScaler        # For scaling the data
from sklearn.model_selection import train_test_split    # For splititng data into test and train sets
from numpy import random

n_hidden_l1 = 25                                    # Number of nodes in the first hidden layer - layer 1
n_hidden_l2 = 12                                    # Number of nodes in the second hidden layer - layer 2
n_input = 64                                        # Number of input nodes
n_output = 10                                       # Number of output nodes


def convert_to_one_hot(y, l):                       # Convert output values to one hot encoded vectors
    y_vect = np.zeros((l, len(y)))                  # Since we have ten digits so we'll have one hot encoded vectors of length 10
    for i in range(len(y)):                         # Traverse over the data in y
        y_vect[y[i], i] = 1                         # For the given data row, assign the 'target value'-th column in the row, the value '1'
    return y_vect


def random_mini_batches(X, Y, batch_size, seed):                            # Create random batches of the data of the size 'batch_size'
    random.seed(seed)                                                       # Create seed
    random_idxs = random.choice(Y.shape[1], Y.shape[1], replace=False)      # create an array of random ids of the size equal to number of samples in X,Y - this will be equal to Y.shape[1]
    X_shuffled = X[:, random_idxs]                                          # create shuffled X by passing the random ids
    y_shuffled = Y[:, random_idxs]                                          # create shuffled y by passing the random ids

    mini_batches = [(X_shuffled[:,i:i+batch_size], y_shuffled[:,i:i+batch_size])    # create mini batches by dividing X_shuffled and y_shuffled into batches of size 'batch_size'
                   for i in range(0, Y.shape[1], batch_size)]
    return mini_batches                                                             # return the group of batches created


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 8 * 8 = 64)
    n_y -- scalar, number of classes (from 0 to 9, so -> 10)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 64]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [10, 12]
                        b3 : [10, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)                   # set seed for the random number

    # initialize parameters weights with random values and biases with zeros
    W1 = tf.get_variable("W1", [n_hidden_l1, n_input], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [n_hidden_l1, 1], initializer=tf.constant_initializer(0.0))
    W2 = tf.get_variable("W2", [n_hidden_l2, n_hidden_l1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [n_hidden_l2, 1], initializer=tf.constant_initializer(0.0))
    W3 = tf.get_variable("W3", [n_output, n_hidden_l2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [n_output, 1], initializer=tf.constant_initializer(0.0))

    #store the pwightes and biases in the dictionary 'parameter'
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    We'll be using 'relu' as the activation function for the hidden layers

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3

    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=300, minibatch_size=22, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 64, number of training examples = 1078)
    Y_train -- test set, of shape (output size = 10, number of training examples = 1078)
    X_test -- training set, of shape (input size = 64, number of training examples = 719)
    Y_test -- test set, of shape (output size = 64, number of test examples = 719)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """


    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)              # create placeholder for X(size = n_input) and Y(size = n_output) - this is where we will feed our training input and output data

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):                 # run epochs - going through the entire cycle of data for gradient descent

            epoch_cost = 0.                                                                 # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)                                       # number of minibatches of size 'minibatch_size' in the train set
            seed = seed + 1                                                                 # increment seed to get different combination of mini btaches in each epoch
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)       # get mini batches

            for minibatch in minibatches:                                                   # traverse over each of the batches in the minibatches
                (minibatch_X, minibatch_Y) = minibatch                                      # separate X(input) and Y(output) from the selected minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # Calculate epoch cost -- average cost of minibatches
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every 100 epochs
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))      # print epoch number and epoch cost after every 100 epochs
            if print_cost == True and epoch % 5 == 0:                       # append the epoch_cost to the list of costs after every 5 epochs
                costs.append(epoch_cost)

        print(costs)
        print(np.squeeze(costs))
        print(np.squeeze(costs).shape)
        # plot the cost
        # plt.plot(np.squeeze(costs))
        plt.plot(costs)                                         # plot costs data
        plt.ylabel('cost')                                      # label the y-axis
        plt.xlabel('iterations (per tens)')                     # label the x-axis
        plt.title("Learning rate =" + str(learning_rate))       # add the title
        plt.show()                                              # display the plot

        # lets save the parameters in a variable
        parameters = sess.run(parameters)                       # run the sess to retrieve trained values of parameters
        print("Parameters have been trained!")
        # print("parameters: ", parameters)

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3, axis=0), tf.argmax(Y, axis=0))
        # tf.argmax gives the index of the highest value in the column(axis=0) or row(axis=1)
        # tf.equal will return an array of 'true' or 'false' for where the values are equal or different
        # resultant correct_predcition will be an array of 'True' and 'False' values
        print(" correct_prediction" , correct_prediction.eval({X: X_test, Y: Y_test}, session = sess))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # tf.cast - casts the tensor 'correct_predictions' to a new type 'float'  - converted from bool type to float type
        # tf.reduce_mean - calculates the mean
        # tf.multiply - multiplies mean by 100 to get the accuracy as a percentage

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))           # accuracy.eval will evaluate the value for accuracy - and we'll feed to the graph X_train and Y_train to get accuracy on the training data
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))              # accuracy.eval will evaluate the value for accuracy - and we'll feed to the graph X_test and Y_test to get accuracy on the test data

        # Let's calculate the confusion matrix
        y = tf.argmax(Z3, axis=0)                                                                   # get the predicted output 'values' as y by reducing Z3(predicted output vectors) along the column axis - You can imagine this as an inverse operation to creating one-hot encoded vectors
        classification = y.eval(feed_dict={X: X_test}, session=sess)                                # evaluate y by passing the X_test data and store in variable classification
        print(classification)                                                                       # print the predicted digits
        right_predictions = tf.argmax(Y_test,axis=0)                                                # get the actual output values - here we are just reducing Y_test to get values for the corresponding one hot encoded vectors
        right_predictions = right_predictions.eval(feed_dict={X: X_test}, session=sess)             # evaluate the actual values and store as right_predictions
        print(right_predictions)                                                                    # print values for the actual digits outputs
        confusion_matrix = tf.contrib.metrics.confusion_matrix(right_predictions, classification)   # calculate the confusion matrix using predicted and actual output values
        print("Confusion matrix:  \n", confusion_matrix)                                            # since we have not yet evaluated the confusion matrix by running throug the graph, we'll not get the 'values' displayed for the confusion matrix here
        confusion_matrix =  tf.Tensor.eval(confusion_matrix, feed_dict=None, session=None)          # evaluate the confusion matrix
        print('Confusion Matrix: \n\n',confusion_matrix)                                            # print the confusion matrix

        return parameters                                                                           # return the trained parameters



# MAIN:
if __name__ == "__main__":
    digits = load_digits()                                    # Load the digits from the dataset

    X = digits.data                                           # get the inputs in the variable 'X'
    y = digits.target                                         # get the output in the variable 'y'

    # Normalize image vectors - this will help reach gradient minimum faster
    X_scale = StandardScaler()                              # Scale the data to improve convergence of neural network
    X = X_scale.fit_transform(X)                  # This will scale the arrays of digits so that the mean becomes zero and standard deviation is reduced, this way convergence occurs comparatively faster

    # X = digits.data                                           # get the inputs in the variable 'X'
    y = digits.target                                         # get the output in the variable 'y'

    #split the data into test and train (60% training data, 40% test data)
    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(X, y, test_size=0.4)    # Split the data into test and train - 40 percent test

    np.random.seed(1)

    # Example of a picture - plot a sample digit picture
    plt.gray()                                              # Initialize a gray figure
    index = 0                                               # Define the index
    plt.matshow(X_train_orig[index].reshape(8,8))           # Display a sample digit - matshow displays an array as a matrix
    plt.show()                                              # Show the plot
    print (Y_train_orig[0])                                 # Print the digit value or class label


    # Flatten the training and test images
    X_train = X_train_orig.reshape(X_train_orig.shape[0], -1).T                 # reshape the input 2d arrays into a column vectors
    X_test = X_test_orig.reshape(X_test_orig.shape[0], -1).T                    # reshape the test input 2d arrays into a column vectors

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, n_output)                    # convert the output values into one hot encoded vectors - n_output is the number of classes
    Y_test = convert_to_one_hot(Y_test_orig, n_output)                      # e.g class '3' will be mapped to [0 0 0 1 0 0 0 0 0 0]

    print("number of training examples = " + str(X_train.shape[1]))         # print number of training examples in train set
    print("number of test examples = " + str(X_test.shape[1]))              # print number of training examples in test set
    print("X_train shape: " + str(X_train.shape))                           # print shape of X_train
    print("Y_train shape: " + str(Y_train.shape))                           # print shape of Y_train
    print("X_test shape: " + str(X_test.shape))                             # print shape of X_test
    print("Y_test shape: " + str(Y_test.shape))                             # print shape of Y_test

    parameters = model(X_train, Y_train, X_test, Y_test)                    # start training the model and store the parameters trained in the variable 'parameters'


