import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
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
    
KNN = KNearestNeighbour()
KNN.train(x_train, y_train)
#predInd = 6
#print(f"Model predicted: {NN.predict(x_test[predInd])}")
#print(f"Actual: {y_test[predInd]}")

def accuracy(model, testX, testY):
    correct = 0
    incorrect = 0
    for i in range(testX.shape[0]):
        modPred = model.predict(testX[i], 10)
        if modPred != testY[i]:
            incorrect += 1
        else:
            correct += 1
    fig = plt.figure()
    plt.ylabel("Amount")
    plt.xlabel(f"Accuracy: {(correct/(correct+incorrect))*100}%")
    plt.bar(["correct", "incorrect"], [correct, incorrect])
    plt.show()
    print(f"Accuracy: {(correct/(correct+incorrect))*100}%")

accuracy(KNN, x_test, y_test)
