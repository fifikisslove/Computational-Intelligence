import numpy as np

"""
Computational Intelligence TU - Graz
Assignment 1: Linear Regression
Part 1: With polynomial non-linear features

This file contains the main work to be done.
The functions are:
- TODO design_matrix: Create the design matrix including the polynomial expansions and the constant feature
- TODO train: finds the analytical solution of linear regression
- TODO compute_error: return the cost function of linear regression Mean Square Error
- train_and_test: call the compute error function and all sets and return the corresponding errors

"""
__author__ = 'bellec,subramoney'


def design_matrix(x, degree):
    """
    Creates the design matrix given the data x.
    The design matrix is built of all polynomials of x from degree 0 to 'degree' minus one.

    EX: for the data x = [0,1,2] and degree 2
    the function should return: [[1,1,1],[0,1,2],[0,1,4]]

    :param x: numpy array of shape (N,1)
    :param degree: Higher degree of the polynomial
    :return: Expanded data in a numpy array of shape (N,degree)
    """

    ######################
    #
    # TODO
    #
    # Return the numpy array of shape (N,degree)
    # Storing the data of the form x_i^j at row i and column j
    # Look at the function description for more info
    #
    # TIP: use the power function from numpy
    X = np.tile(x, (1,degree))
    for i in range(1,np.shape(X)[1],1):
        for j in range(0,np.shape(X)[0],1):
            X[j][i] = np.power(X[j][i],i+1)
    X = np.insert(X, 0, 1, axis=1)

    #
    # END TODO
    ######################

    return X


def train(x, y, degree):
    """
    Returns the optimal coefficients theta that minimizes the error
    ||  X * theta - y ||**2
    when X is the polynomial expansion of x_train of degree 'degree'.

    :param x: numpy array on the input
    :param y: numpy array containing the output
    :param degree: maximum polynomial degree in the polynomial expansion
    :return: a numpy array containing the coefficient of each polynomial degree in the regression
    """

    ######################
    #
    # TODO
    #
    # Returns the analytical solution of the linear regression
    #
    # TIPs:
    #  - Don't forget to first expand the data
    #  - WARNING:   With numpy array * is a term-term matrix multiplication
    #               The function np.dot performs a classic matrix multiplication
    #
    #  - To compute the pseudo inverse (A*A.T)^-1 * A.T with a more stable algorithm numpy provides the function pinv
    #   pinv is accessible in the sub-library numpy.linalg
    #

    theta_opt = np.zeros(degree + 1)  # TODO: Change me
    x = design_matrix(x,degree)
    theta_opt = np.linalg.inv(np.dot(x.transpose(), x)).dot(x.transpose()).dot(y)

    return theta_opt

    # END TODO
    ######################



def compute_error(theta, degree, x, y):
    """
    Predict the value of y given by the model given by theta and degree.
    Then compare the predicted value to y and provide the mean square error.

    :param theta: Coefficients of the linear regression
    :param degree: Degree in the polynomial expansion
    :param x: Input data
    :param y: Output data to be compared to prediction
    :return: err: Mean square error
    """

    ######################
    #
    # TODO
    #
    # Returns the analytical solution of the linear regression
    #
    # TIPs:
    #  - Don't forget to first expand the data
    #  - WARNING:   With numpy array * is a term-term matrix multiplication
    #               The function np.dot performs a matrix multiplication
    #               A longer alternative is to first change your array to the matrix class using np.matrix,
    #               Then * becomes a matrix multiplication
    #
    #  - One can use the numpy function mean

    #err = -1  # TODO: Change me
    x = design_matrix(x,degree)
    err2 = (np.dot(x, theta) - y) ** 2 #squared error : (prediction - real output)^2
    err = np.mean(err2) #the mean of the squared error matrice

    #
    # END TODO
    ######################
    return err


def train_and_test(data, degree):
    """
    Train the model with degree 'degree' and provide the MSE for the training, validation and testing sets

    :param data:
    :param degree:
    :return:
    """

    theta = train(data['x_train'], data['y_train'], degree)

    err_train = compute_error(theta, degree, data['x_train'], data['y_train'])
    err_val = compute_error(theta, degree, data['x_val'], data['y_val'])
    err_test = compute_error(theta, degree, data['x_test'], data['y_test'])

    return theta, err_train, err_val, err_test

