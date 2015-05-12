#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from matplotlib import pyplot as plt
from urllib import urlopen

def cmap(x):
    return 'r' if x==1 else 'b'

def abline(plt, linear_model, label):
    '''
    :param plt: plot
    :param linear_model: linear classifier which we want to plot
    :return:
    '''
    params = linear_model.coef_[0].tolist()+[linear_model.intercept_[0]]
    print(params)
    gca = plt.gca()
    gca.set_autoscale_on(False)
    x = np.array(plt.gca().get_xbound())
    intercept = -params[2]/params[1]
    slope = -params[0]/params[1]
    return plt.plot(x, intercept + slope*x, label=label)

# nonlinear dataset
raw_data = np.loadtxt(
    urlopen("https://raw.githubusercontent.com/olologin/AdaBoost-with-logistic-regression/master/non%20linear%20dataset.csv"),
    delimiter=",")

'''
# linear dataset
raw_data = np.loadtxt(urlopen("https://raw.githubusercontent.com/olologin/AdaBoost-with-logistic-regression/master/ex2data1.txt"),
                    delimiter=",")
# normalization
raw_data[:,:2] = ((raw_data[:,:2]-raw_data[:,:2].mean(0))/raw_data[:,:2].std(0))'''


dataset = raw_data[:,:2]
y = raw_data[:,2]

plt.scatter(x=dataset[:,0], y=dataset[:,1], c=[cmap(i) for i in y])

#base_classifier = linear_model.Perceptron()
base_classifier = linear_model.SGDClassifier(loss='log')
clf = ensemble.AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50, algorithm='SAMME')
clf.fit(dataset[:,:3], y)
print "Score on training set: ", clf.score(dataset[:,:3], y)

for idx, estimator in enumerate(clf.estimators_):
    abline(plt, estimator, "Estimator weight:" + str(clf.estimator_weights_[idx]))

plt.legend()
plt.show()
