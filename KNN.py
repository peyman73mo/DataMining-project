import numpy as np
from collections import Counter


#------------------------------------------------------------------
def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

def manhatan_distance(x1,x2):
    return np.sum(np.abs(x1-x2))

from scipy.spatial import distance
#------------------------------------------------------------------
class KNN:

    def __init__(self, k = 2):
        self.k = k
        self.distance = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, _distance=euclidean_distance):
        self.distance = _distance
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):

        if self.distance == euclidean_distance:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == manhatan_distance:
            distances = [manhatan_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == distance.minkowski:
            distances = [distance.minkowski(x, x_train, p=3) for x_train in self.X_train]
        
        k_idx = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        common_data_point = Counter(k_neighbor_labels).most_common(1)
        return common_data_point[0][0]
 #------------------------------------------------------------------   

