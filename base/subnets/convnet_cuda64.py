''' Version 1.000
 Code provided by Daniel Jiwoong Im and Chris Dongjoo Kim
 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''

'''Demo of Generating images with recurrent adversarial networks.
For more information, see: http://arxiv.org/abs/1602.05110
'''


import os, sys
import numpy as np

import theano
import theano.tensor as T

from collections import OrderedDict

from layers.batch_norm_conv_layer import BN_Conv_layer
from layers.utils import activation_fn_th, initialize_weight

class convnet64():

    def __init__ (self, model_params, nkerns=[3,1,2,4,8], filter_sizes=[5,5,5,5,5,4], ltype='gan'):
        """Initializes the architecture of the discriminator"""
        
        self.ltype = ltype
        self.num_hid, num_dims, num_class, self.batch_size, self.num_channels, kern = model_params
        self.D =  int(np.sqrt(num_dims / self.num_channels))
        numpy_rng=np.random.RandomState(1234)

        self.nkerns     = np.asarray(nkerns) * kern # of constant gen filters in first conv layer
        self.nkerns[0] = self.num_channels
        self.filter_sizes=filter_sizes

        num_convH = self.nkerns[-1]*filter_sizes[-1]*filter_sizes[-1]

        #self.W      = initialize_weight(num_convH,  self.num_hid,  'W', numpy_rng, 'uniform') 
        #self.hbias  = theano.shared(np.zeros((self.num_hid,), dtype=theano.config.floatX), name='hbias')       
        self.W_y    = initialize_weight(num_convH, num_class,  'W_y', numpy_rng, 'uniform') 

        self.L1 = BN_Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[1], bnkern=self.nkerns[0] , bfilter_sz=filter_sizes[0], tfilter_sz=filter_sizes[1])
        self.L2 = BN_Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[2], bnkern=self.nkerns[1] , bfilter_sz=filter_sizes[1], tfilter_sz=filter_sizes[2])
        self.L3 = BN_Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[3], bnkern=self.nkerns[2] , bfilter_sz=filter_sizes[2], tfilter_sz=filter_sizes[3])
        self.L4 = BN_Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[4], bnkern=self.nkerns[3] , bfilter_sz=filter_sizes[3], tfilter_sz=filter_sizes[4])

        self.num_classes = num_class
        self.params =   self.L1.params + self.L2.params + \
                        self.L3.params + self.L4.params + [self.W_y]

    def propagate(self, X, num_train=None, atype='relu', reshapeF=False):
        """Propagate, return binary output of fake/real image"""   

        if reshapeF: 
            image_shape0=[X.shape[0], self.num_channels, self.D, self.D]
            X = X.reshape(image_shape0)
        H0 = self.L1.conv(X , atype=atype)
        H1 = self.L2.conv(H0, atype=atype)
        H2 = self.L3.conv(H1, atype=atype) 
        H3 = self.L4.conv(H2, atype=atype) 
        H3 = H3.flatten(2)

        if self.ltype=='gan' or self.ltype=='js':
            y  = T.nnet.sigmoid(T.dot(H3, self.W_y))    
        else:
            y = T.dot(H3, self.W_y)

        return y


#     def cost(self, X, y):
        # p_y_x = self.propagate(X)
        # return -T.mean(T.log(p_y_x)[T.arange(y.shape[0]), y])
   
   
    def weight_decay_l2(self):
        return 0.5 * (T.sum(self.W_y**2))


    def weight_decay_l1(self):
        return T.sum(abs(self.W))


    def errors(self, X, y, num_train=None):
        """error computed during battle metric"""

        p_y_x   = self.propagate(X, num_train=num_train).flatten()
        pred_y  = p_y_x  > 0.5
        return T.mean(T.neq(pred_y, y))


    def set_params(self, params):

            [self.W, self.hbias, self.W_y, self.ybias, self.W0, self.b0, self.W1, self.b1] = params
            self.params = params



