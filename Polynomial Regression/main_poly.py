#!/usr/bin/env python

import json

import matplotlib.pyplot as plt
import numpy as np

import poly
from plot_poly import plot_poly

"""
Computational Intelligence TU - Graz
Assignment 1: Linear Regression
Part 2: With polynomial non-linear features

This files:
1) loads the data from 'data.json'
2) trains and test a linear regression model for a given number of degree
3) plots the results

TODO boxes are to be found in 'poly.py'
"""
__author__ = 'bellec,subramoney'


def main():
    # Set the degree of the polynomial expansion
    degree = 20
    data_path = 'data.json'

    # Load the data and make sure the shape of all arrays in of the form (n,1)
    f = open(data_path, 'r')
    data = json.load(f)
    for k, v in data.items():
        data[k] = np.array(v).reshape((len(v), 1))

    # Print the training and testing errors
    theta, err_train, err_val, err_test = poly.train_and_test(data, degree)
    print('Training error {} \t Validation error {} \t Testing error {} '.format(err_train, err_val, err_test))

    plot_poly(data, degree, theta)
    #plt.savefig('POLY' + str(degree) + '.png')
    plt.show()


if __name__ == '__main__':
    main()
