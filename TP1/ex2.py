#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:33:11 2017

@author: bigue-co
"""

from sklearn.datasets import fetch_mldata
import numpy as np

mnist = fetch_mldata('MNIST original')
help(np.random.randint)
random = np.random.randint(70000, size=5000)
data = mnist.data[random]
target = mnist.target[random]

from sklearn.model_selection import train_test_split

images_train, images_test, target_train, target_test = train_test_split(data, target, train_size=0.8)

"""
Question : faut-il stocker sous forme d'images (28x28) ou de vecteurs (1x784) ?
-> On est obligé de faire avec des vecteurs car fit s'attend à recevoir des vecteurs.
"""

from sklearn import neighbors
import array
classifier = neighbors.KNeighborsClassifier(10)
classifier.fit(images_train, target_train)
prediction = classifier.predict(images_test)

print(prediction[4])
print(target_test[4])

classifier.score(images_test, target_test)
classifier.predict_proba(images_test)

error = 1-classifier.score(images_test, target_test)
"""
import metrics
from metrics import precision_
import score.recal

precision = metrics.precision_score(target_test, prediction, average)
"""

scores = {}
for i in range(2,16):
    classifier = neighbors.KNeighborsClassifier(i)
    classifier.fit(images_train, target_train)
    scores[i] = classifier.score(images_test, target_test)
    
max(scores, key=scores.get)

"""Pas bien compris comment ça marchait :"""
from sklearn import model_selection
help(model_selection.KFold)
kf = model_selection.KFold(n_splits=10, shuffle=True)
kf.get_n_splits(images_test)
for train_index, test_index in kf.split(images_test):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = images_test[train_index], images_test[test_index]
    y_train, y_test = target_test[train_index], target_test[test_index]
    classifier.fit(X_train, y_train)


