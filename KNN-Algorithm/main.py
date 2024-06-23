# K Nearest Neighbors
import numpy as np
from collections import Counter

def euclidean_distance(x1: float, x2: float) -> float:
    """ 
        Compute the distance by first finding the distance between two points,
        squaring each difference, summing these squares, and the taking the square
        root of sum.
    """
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, X: list[float], y: list[int]) -> None:
        print(f"x {X}: y: {y}")
        """ This method is used to fit the model with training data. """
        self.x_train = X
        self.y_train = y

    def predict(self, X: list[float]) ->list[int]:
        """ 
            This method predicts the class labels for the input data 'X'. 
            Then returns the predictions
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Compute the closest distance
        """
            This computes the euclidean distance from the input sample 'x'
            to each sample in the training set 'self.x_train', creating 
            a list of distances. 
        """
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]


        # Get the closest K
        """ This sorts the distances and gets the indices of the 'k' nearest neighbors """
        K_indicies = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in K_indicies]

        # Mojority Vote
        """ 
            This uses the Counter class to find the most common label among
            'k' nearest neighbor
        """
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
