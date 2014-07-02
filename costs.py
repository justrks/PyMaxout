import numpy
import theano
import theano.tensor as T

def mean_squared_error(y, y_hat):
    return (0.5 * (y - y_hat) ** 2).mean()

def cross_entropy(y, y_hat):
    return (-y * T.log(y_hat) - (1-y) * T.log(1-y_hat)).mean()

def l1_norm(w):
    return T.abs_(w).sum()

def l2_norm(w):
    return (w ** 2).sum()
