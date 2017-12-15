#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

TME 9

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import numpy.random as npr

def tirage(m):
    return [np.random.uniform(-1,1) * m,np.random.uniform(-1,1) * m]

def monteCarlo(N):
    tirages = np.array([tirage(1) for i in range(N)])    
    X = tirages[:,0]
    Y = tirages[:,1]    
    pi = 4 * (np.where(np.sqrt(X**2 + Y**2) <= 1)[0].size) / N
    return (pi,X,Y)

plt.figure()

# trace le carré
plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')

# trace le cercle
x = np.linspace(-1, 1, 100)
y = np.sqrt(1- x*x)
plt.plot(x, y, 'b')
plt.plot(x, -y, 'b')

# estimation par Monte Carlo
pi, x, y = monteCarlo(int(1e4))

# trace les points dans le cercle et hors du cercle
#dist = x*x + y*y 
#plt.plot(x[dist <=1], y[dist <=1], "go")
#plt.plot(x[dist>1], y[dist>1], "ro")
#plt.show()

# si vos fichiers sont dans un repertoire "ressources"
with open("./countWar.pkl", 'rb') as f:
    (count, mu, A) = pkl.load(f, encoding='latin1')

with open("./secret.txt", 'r') as f:
    secret = f.read()[0:-1] # -1 pour supprimer le saut de ligne

with open("./secret2.txt", 'r') as f:
    secret2 = f.read()[0:-1] # -1 pour supprimer le saut de ligne
 
def swapF(d1):
    d2 = {}
    for cle in d1.keys():
        d2[d1[cle]] = cle
    return d2
     
tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }
print("swapF(tau)=",swapF(tau))

#def decrypt(mess, d1):
#    ret_str = ""
#    for c in mess:
#        nouvel_char = d1[c]
#        ret_str = ret_str + nouvel_char
#    return ret_str
def decrypt(m, f):
    return u"".join([f[c] for c in m])
    
chars2index = dict(zip(np.array(list(count.keys())), np.arange(len(count.keys()))))

def logLikelihood(mess, mu, A, chars2index):
    probas = []
    precedant = chars2index[mess[0]]
    probas.append(np.log(mu[precedant]))
    for i in range(1, len(mess)):
        new = chars2index[mess[i]]
        proba = np.log(A[precedant, new])
        probas.append(proba)
        precedant = new
    return np.sum(probas)

logl1 = logLikelihood( "abcd", mu, A, chars2index )
print(logl1)
logl2 = logLikelihood( "dcba", mu, A, chars2index )
print(logl2)

def MetropolisHastings(mess, mu, A, tau, N, chars2index):
    maxDecod = decrypt(mess, tau)
    #TODO
    
    return maxDecod

def identityTau ():
    tau = {} 
    for k in count.keys ():
        tau[k] = k
    return tau
#MetropolisHastings( secret2, mu, A, identityTau (), 10000, chars2index)

