import numpy as np
import matplotlib.pyplot as plt
import random
import data

import pdb

def softmax(x):
  exp = np.exp(x)
  sumexp = np.sum(np.exp(x), axis=1)
  return exp / sumexp[:, None]

def logreg_train(X,Y_):
  '''
    Arguments
      X:  input_data, np.array NxD
      Y_: class indexes, np.array Nx1

    Return
      w, b: logistic regression parameters
  '''

  C = int(max(Y_) + 1)
  N = len(Y_)
  D = 2

  param_delta = 0.03

  w = np.random.randn(C, D)
  b = np.zeros((C,), dtype=int)
  param_niter = 10000

  true_class = np.zeros((N, C), dtype=int)
  for ind in range(N):
      true_class[ind][Y_[ind]] = 1

  for i in range(param_niter):
    # classification scores
    scores = np.dot(X, w.T) + b
    
    # probabilities for each class
    probs = softmax(scores)

    # loss
    loss = (-1/N) * np.sum(np.log(probs))

    # output
    if i % 1000 == 0:
      print("iteration {}: loss {}".format(i, loss))

    # loss derivates with respect to classes measures
    dL_dscores = probs - true_class
    
    # parameter gradients
    grad_w = (1/N)*np.dot(dL_dscores.T, X)
    grad_b = np.sum(dL_dscores, axis=0)

    # improved parameters
    w += -param_delta * grad_w
    np.add(b, -param_delta*grad_b, out=b, casting="unsafe")
  return w, b

def logreg_classify(X, w, b):
  scores = np.dot(X, w.T) + b

  return np.array(softmax(scores))

def myDummyDecision(X):
  scores = X[:,0] + X[:,1] - 5
  return scores

def logreg_decfun(w, b):
    def classify(X):
        return logreg_classify(X, w, b)
    return classify

if __name__=="__main__":
  np.random.seed(100)
  
  # get data
  # X,Y_ = sample_gmm_2d(4, 2, 30)
  X,Y_ = data.sample_gauss_2d(3, 100)

  weights, bias = logreg_train(X,Y_)

  # get the class predictions
  probs = logreg_classify(X, weights, bias) 
  Y = np.argmax(probs, axis=1)
  # Y = myDummyDecision(X)>0.5  

  accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
  AP = data.eval_AP(Y_[Y.argsort()])
  print (accuracy, recall, precision, AP)

  # graph the decision surface
  decfun = logreg_decfun(weights, bias)
  rect=(np.min(X, axis=0), np.max(X, axis=0))
  data.graph_surface(decfun, rect, offset=0.5)
  
  # graph the data points
  data.graph_data(X, Y_, Y, special=[])

  plt.show()