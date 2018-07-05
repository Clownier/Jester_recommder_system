from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans


def predict(Xtr, Ytr, Xte, Yte, mod):
    train_accuracy = 0.
    for i in range(len(Xtr)):
        isspam = mod.predict([Xtr[i]])
        # Calculate accuracy
        if Ytr[i] == isspam:
            train_accuracy += 1./len(Xtr)
    
    print("Accuracy on Train Set:", train_accuracy)
    
    test_accuracy = 0.
    for i in range(len(Xte)):
        isspam = mod.predict([Xte[i]])
        # Calculate accuracy
        if Yte[i] == isspam:
            test_accuracy += 1./len(Xte)

    print("Accuracy on Train Set:", test_accuracy)
    return train_accuracy, test_accuracy


def knn(Xtr, Ytr, Xte, Yte, k = 1):
    neigh = KNeighborsClassifier(n_neighbors = k)
    neigh.fit(Xtr, Ytr)
    return predict(Xtr, Ytr, Xte, Yte, neigh)

def NaiveBayes(Xtr, Ytr, Xte, Yte, mod):
    gnb = mod
    gnb.fit(Xtr, Ytr)
    return predict(Xtr, Ytr, Xte, Yte, gnb)

def SVM(Xtr, Ytr, Xte, Yte):
    clf = LinearSVC(random_state=0)
    clf.fit(Xtr, Ytr)
    return predict(Xtr, Ytr, Xte, Yte, clf)

def kmeans(Xtr, Ytr, Xte, Yte, k):
    km = KMeans(n_clusters = k, random_state = 42)
    km.fit(Xtr)
    return predict(Xtr, Ytr, Xte, Yte, km)
