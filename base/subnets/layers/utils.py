''' Version 1.000
 Code provided by Daniel Jiwoong Im 
 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''


# import os, sys, cPickle, PIL, math, pylab
# import matplotlib as mp
# import matplotlib.pyplot as plt

import numpy as np
# from numpy.lib import stride_tricks

import theano
import theano.tensor as T
# from theano.tensor.shared_randomstreams import RandomStreams
#import theano.sandbox.rng_mrg as RNG_MRG
rng = None  # will be initialized in main_dcgan_lsun.py

# MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_conv_weights(W_low, W_high, filter_shape, numpy_rng, rng_dist='normal'):
    """
    initializes the convnet weights.
    """

    if 'uniform' in rng_dist:
        return np.asarray(
            numpy_rng.uniform(low=W_low, high=W_high, size=filter_shape),
                dtype=theano.config.floatX) 
    elif rng_dist == 'normal':
        # return 0.02 * numpy_rng.normal(size=filter_shape).astype(theano.config.floatX)
        return np.asarray(numpy_rng.normal(loc=0.0, scale=0.02, size=filter_shape), dtype = theano.config.floatX)


def activation_fn_th(X,atype='sigmoid', leak_thrd=0.2):
    '''collection of useful activation functions'''

    if atype == 'softmax':
        return T.nnet.softmax(X)
    elif atype == 'sigmoid':
        return T.nnet.sigmoid(X)
    elif atype == 'tanh':
        return T.tanh(X)
    elif atype == 'softplus':
        return T.nnet.softplus(X)
    elif atype == 'relu':
        return (X + abs(X)) / 2.0
    elif atype == 'linear':
        return X
    elif atype =='leaky':
        f1 = 0.5 * (1 + leak_thrd)
        f2 = 0.5 * (1 - leak_thrd)
        return f1 * X + f2 * abs(X)
        
def initialize_weight(n_vis, n_hid, W_name, numpy_rng, rng_dist='uniform'):
    """
    """

    if 'uniform' in rng_dist:
        W = numpy_rng.uniform(low=-np.sqrt(6. / (n_vis + n_hid)),\
                high=np.sqrt(6. / (n_vis + n_hid)),
                size=(n_vis, n_hid)).astype(theano.config.floatX)
    elif rng_dist == 'normal':
        return theano.shared(np.asarray(numpy_rng.normal(loc=0.0,\
                scale=0.02, size=(n_vis,n_hid)), dtype = theano.config.floatX))
    elif rng_dist == 'ortho': ### Note that this only works for square matrices
        N_ = int(n_vis / float(n_hid))
        sz = np.minimum(n_vis, n_hid)
        W = np.zeros((n_vis, n_hid), dtype=theano.config.floatX)
        for i in xrange(N_):
            temp = 0.01 * numpy_rng.normal(size=(sz, sz)).astype(theano.config.floatX)
            W[:, i*sz:(i+1)*sz] = sp.linalg.orth(temp)


    return theano.shared(value = np.cast[theano.config.floatX](W), name=W_name)



