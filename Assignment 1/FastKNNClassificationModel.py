from scipy.spatial import KDTree
import numpy as  np


class FastKNNClassificationModel():
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.kdTree = None

   
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.kdTree = KDTree(self.X_train)
        return None
        

    def predict(self, X):
        predictions =  []
        count_data = {}
        for x in X:
            distance, indices = self.tree.query(x, k=self.k)
            y_value = self.y_train[indices]
            for y in y_value:
                if y in count_data:
                    count_data[y] += 1
                else:
                    count_data[y] = 1

            prediction = max(count_data, key=count_data.get)
            predictions.append(prediction)
        return np.array(predictions)

    def evaluate(self, y_true, y_predicted):
        return np.mean(y_true == y_predicted)