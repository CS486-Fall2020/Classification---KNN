import numpy as np
import pandas as pd
import math
from collections import Counter

class KNN:

    #Constructor for KNN
    def __init__(self, k=5):
        self.__k = k


    #Fit function just takes data as X and labels as y
    def fit(self, X, y):
        self.__X = X
        self.__y = y

    #This is the euclidean distance between x and y
    def __dist(self, x, y):
        return (math.sqrt(abs(sum([(a - b) ** 2 for a, b in zip(x, y)]))))

    #Returns X and y
    def getXy(self):
        return self.__X, self.__y

    #This function returns a tuple array of the distance of a point and its label
    #The coordinates of the point are abstracted here
    def get_distances(self, x):
        distances = []
        for train_X, hyp in  zip(self.__X, self.__y):
            dist = self.__dist(x, train_X)
            distances.append((dist, hyp))
        return distances

    #The hypothese is the most occuring label in the K nearest neighbors
    def __get_hyp(self, top_k):
        labels = [x[1] for x in top_k]
        occurence_count = Counter(labels) 
        return occurence_count.most_common(1)[0][0]

    #Returns test data and classification hypotheses
    def predict(self, X):
        #Initialize an empty hypotheses list
        hypotheses = []
        #For each item to predict
        for x in X:
            #Get the distances to every point in the training dataset
            distances = self.get_distances(x)
            #Sort these distances from closest to farthest
            sorted_distances = sorted(distances, key=lambda x: x[0])
            #Keep the closest k points
            top_k = sorted_distances[:self.__k]
            #Assign the most common label within k nearest neighbors
            hyp = self.__get_hyp(top_k)
            #Append it to the hypotheses
            hypotheses.append(hyp)
        #Convert to np array
        hypotheses = np.array(hypotheses)

        return X, hypotheses

