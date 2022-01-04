import numpy as np
import typing as t
from collections import defaultdict

class KNearestNeighbour:
    def __init__(self):
        pass
    
    def train(self, X : np.ndarray, Y : np.ndarray):
        if any([
            type(X) != np.ndarray,
            type(Y) != np.ndarray
        ]):
            raise Exception("Both X and Y has to be both ndarray.")
        elif X.shape[0] != Y.shape[0]:
            raise Exception(f"X and Y must have the same number of entry. X has {self.X.shape[0]} and Y has {self.Y.shape[0]}.")
        elif not X.shape[0]:
            raise Exception("X and Y has to contain something!")
        self.X = X
        self.Y = Y
        
    
    def predict(self, image : np.ndarray, k : int):
        dd = defaultdict(int)
        if image.shape != self.X[0].shape:
            raise Exception(f"Image input must have same shape as X. X is {self.X[0].shape} and image input is {image.shape}")
        
        distances = [np.sum(np.abs(self.X[i] - image)) for i in range(self.X.shape[0])]
        for i in range(k):
            index = np.argmin(distances)
            dd[self.Y[index]] += 1
            distances[index] = 10000000
        dd = {k: v for k, v in sorted(dd.items(), key=lambda item: item[1], reverse = True)}
        answer = next(iter(dd))
        return answer
