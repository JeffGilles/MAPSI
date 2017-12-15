#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import numpy.random as npr


def tirages(N, a, b, sig):
    X = np.random.rand(N)
    e = sig * np.random.randn(N)
    return X, a*X + b + e

a = 6.
b = -1.
N = 100
sig = .4
X, Y = tirages(N, a, b, sig)

#Estimation de parametres probabilistes
#Lineaire : ax+b
def estimationParamProba(X, Y, sig):
    cov = np.cov([X,Y])#/sig**2#comme ratio sig n'annule apres...
    a_proba = cov[0,1]/cov[0,0]
    b_proba = Y.mean() - a_proba * X.mean()
    return a_proba, b_proba

a_proba, b_proba = estimationParamProba(X, Y, sig)
print("paramètres probabilistes:", a_proba, b_proba)

t=np.array([0,1])
plt.figure()
plt.scatter(X,Y)
plt.plot(t, a_proba*t + b_proba, 'r')
plt.show()

#Estimation au sens des moindres carres
Xmc = np.hstack((X.reshape(N,1), np.ones((N,1))))

def estimationMoindresCarres(X,Y):
#    Xt=X.transpose()
#    return np.linalg.solve(np.dot(Xt, X),np.dot(Xt,Y))
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))#=ligne donnée plus loin dans l'énnoncée

a_mc, b_mc = estimationMoindresCarres(Xmc, Y)
print("paramètres moindres carres:", a_mc, b_mc)

def testSansBiais(N,a,b):
    x,y=tirages(N, a, b, 0.)
    a_p, b_p = estimationParamProba(x, y, 0.)
    t=np.array([0,1])
    plt.figure()
    plt.scatter(x,y)
    plt.plot(t, a_p*t + b_p, 'r')
    #Les donnees ne se dispersent pas

#testSansBiais(N,a,b)
"""
def optimDescGradient(X, Y):
    wstar = np.linalg.solve(X.T.dot(X), X.T.dot(y)) # pour se rappeler du w optimal
    
    eps = 5e-3
    nIterations = 30
    w = np.zeros(X.shape[1]) # init à 0
    allw = [w]
    for i in xrange(nIterations):
        # A COMPLETER => calcul du gradient vu en TD
        allw.append(w)
        print w
    
    allw = np.array(allw)
    return w, allw
    """"