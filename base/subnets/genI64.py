import os
import sys

import theano
import theano.tensor as T

import numpy as np
import scipy as sp

from layers.batch_norm_layer import Batch_Norm_layer
from layers.batch_norm_deconv import BN_Deconv_layer
from layers.deconv import Deconv_layer
from layers.utils import rng

import theano.sandbox.rng_mrg as RNG_MRG
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))

class GenI64(object):

    def __init__(self, model_params,filter_sizes=[4,5,5,5,5]):

        """Initialize the architecture of the model"""

        [self.batch_sz, num_dims, self.num_hids, numpy_rng,\
                self.dim_sample, self.nkerns, self.ckern, self.num_channel, _]  = model_params 
        self.nkerns     = np.asarray(self.nkerns) * self.ckern # of constant gen filters in first conv layer
        self.nkerns[-1] = 3

        self.D              = int(np.sqrt(num_dims / self.nkerns[-1]))
        self.numpy_rng      = np.random.RandomState(1234)
        self.filter_sizes   = filter_sizes
        num_convH           = self.nkerns[0]*filter_sizes[0]*filter_sizes[0]

        self.L1 = Batch_Norm_layer(self.dim_sample, num_convH, 'W_z_hf', numpy_rng)
        self.L2 = BN_Deconv_layer(self.batch_sz, numpy_rng, tnkern=self.nkerns[1], bnkern=self.nkerns[0] , bfilter_sz=filter_sizes[0], tfilter_sz=filter_sizes[1])
        self.L3 = BN_Deconv_layer(self.batch_sz, numpy_rng, tnkern=self.nkerns[2], bnkern=self.nkerns[1] , bfilter_sz=filter_sizes[1], tfilter_sz=filter_sizes[2])
        self.L4 = BN_Deconv_layer(self.batch_sz, numpy_rng, tnkern=self.nkerns[3], bnkern=self.nkerns[2] , bfilter_sz=filter_sizes[2], tfilter_sz=filter_sizes[3])
        self.L5 =    Deconv_layer(self.batch_sz, numpy_rng, tnkern=self.nkerns[4], bnkern=self.nkerns[3] , bfilter_sz=filter_sizes[3], tfilter_sz=filter_sizes[4])
   
        self.params =   self.L1.params + self.L2.params + \
                        self.L3.params + self.L4.params + self.L5.params


    def forward(self, Z, testF=False, atype='relu'):
        """
        The forward PROPAGATION TO GENERATE the C_i image.
        """

        H1 = self.L1.propagate(Z, testF=testF, atype=atype)
        H1 = H1.reshape((H1.shape[0], self.nkerns[0], self.filter_sizes[0], self.filter_sizes[0]))
        H2 = self.L2.deconv(H1, atype=atype)
        H3 = self.L3.deconv(H2, atype=atype)
        H4 = self.L4.deconv(H3, atype=atype)
        H5 = self.L5.deconv(H4, atype='sigmoid') #Sigmoid gets applied layer after the cumulation 

        return H5

    
    def get_samples(self, num_sam):
        """
        Retrieves the samples for the current time step. 
        uncomment parts when time step changes.
        """
        Z    = MRG.uniform(size=(num_sam, self.dim_sample), low=-1., high=1.)
        #Z    = MRG.normal(size=(num_sam, self.dim_sample), avg=0., std=1.)
        C    = self.forward(Z) 

        return C


    def weight_decay_l2(self):
        """l2 weight decay used in the optimize_gan for computing the cost of the discriminator"""
        return (self.L1.l2_norm() + self.L2.l2_norm() +\
                self.L3.l2_norm() + self.L4.l2_norm() + self.L5.l2_norm())



