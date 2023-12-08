# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: ISHIKA THAKUR(isthakur) PRITHVI AMIN(aminpri) RADHIKA GANESH(rganesh)
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        # Store the input data and corresponding class labels in the model attributes
        self._X = X
        self._y = y

        #raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        # Initialize an empty list to store predictions for each test sample
        predictions = []

        for sample in X:
          # Calculate distances and retrieve the corresponding class labels for each training sample
          neighbors = sorted([(self._distance(self._X[i], sample), self._y[i]) for i in range(len(self._X))])[:self.n_neighbors]

          # Calculate weights
          weights = [1.0] * self.n_neighbors if self.weights == 'uniform' else [1 / (d + 0.1) for d, _ in neighbors]

          # Calculate weighted occurrences of each class among the neighbors
          occurrences = [sum(w * (neighbour_class == c) for w, (_, neighbour_class) in zip(weights, neighbors)) for c in set(self._y)]

          # Append the predicted class for the current test sample to the predictions list
          predictions.append(occurrences.index(max(occurrences)))

        result = np.array(predictions)
        return result
      #raise NotImplementedError('This function must be implemented by the student.')
