from scipy.spatial import KDTree

import numpy as  np


class FastKNNRegressionModel(MachineLearningModel):
    """
    Class for fast KNN regression model using KDTree.
    """

    def __init__(self, k):
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.tree =KDTree(self.X_train)
        return None
        

    def predict(self, X):
        pass

    def evaluate(self, y_true, y_predicted):
        pass