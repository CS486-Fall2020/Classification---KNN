import numpy as np
import pandas as pd

class KNN:

    #Constructor for KNN
    def __init__(self, k=5):
        self.__k = k

    #Fit function just takes data as X and labels as y
    def fit(self, X, y):
        self.__X = X
        self.__y = y

    #Returns X and y
    def getXy(self):
        return self.__X, self.__y

    #Predicts the class labels for provided X
    def predict(self, X):
        predictions = np.array()