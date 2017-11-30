# -*- coding: utf-8 -*-

"""
Created on Wed Nov 15 16:33:11 2017

@author: bigue-co
"""

"""""
Exercice 1
""""
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

"1."
print(mnist)
print(mnist.data)
print(mnist.target)
len(mnist.data)
help(len)
print(mnist.data.shape)
print(mnist.target.shape)
mnist.data[0]
mnist.data[0][1]
mnist.data[:,1]
mnist.data[:100]
len(mnist.data[0])

"2."
from sklearn import datasets 
import matplotlib.pyplot as plt
mnist = datasets.fetch_mldata('MNIST original')
"Reshape permet de passer de vecteur a uen image et de pouvoir l'afficher ensuite."
images = mnist.data.reshape((-1,28,28))
plt.imshow(images[0], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
help(plt.imshow)
type(images[0])


from sklearn import model_selection
help(model_selection.train_test_split)
images_train, images_test, target_train, target_test = model_selection.train_test_split(images,mnist.target, train_size=0.7)