import theano 
import numpy as np
import scipy as sp
# import theano
import theano.tensor as T

# from convnet_cuda32_v2 import *
# from convnet_cuda28 import *
# from convnet_cuda128 import *
# from genI32 import *
from subnets.convnet_cuda64 import convnet64
from subnets.genI64 import GenI64

class DCGAN():

    def __init__(self, model_params):
     
        gen_params, disc_params = model_params
        
        if gen_params[1] == 784: #MNIST
            self.dis_network = convnet28(disc_params) 
        elif gen_params[1] == 64*64*3:
            self.dis_network = convnet64(disc_params) 
            self.gen_network = GenI64(gen_params)
        elif gen_params[1] == 128*128*3:
            self.dis_network = convnet128(disc_params) 
        else:   
            ##32x32x3, i.e., CIFAR10 would fall here
            self.dis_network = convnet32(disc_params) 
            self.gen_network = GenI32(gen_params)

        self.params = self.dis_network.params + self.gen_network.params
   

    def cost_dis(self, X, num_examples):

        #target1  = T.alloc(1., num_examples)
        p_y__x1  = self.dis_network.propagate(X, \
                    reshapeF=True, atype='leaky').flatten()
        target1 = T.ones(p_y__x1.shape)

        #target0      = T.alloc(0., num_examples)
        gen_samples  = self.gen_network.get_samples(num_examples)
        p_y__x0      = self.dis_network.propagate(gen_samples, atype='leaky').flatten()
        target0     = T.zeros(p_y__x0.shape)

        return T.mean(T.nnet.binary_crossentropy(p_y__x1, target1)), \
                T.mean(T.nnet.binary_crossentropy(p_y__x0, target0))


    def cost_gen(self, num_examples):

        #target      = T.alloc(1., num_examples)
        gen_samples = self.gen_network.get_samples(num_examples)
        p_y__x      = self.dis_network.propagate(gen_samples, atype='leaky').flatten()
        target      = T.ones(p_y__x.shape)

        return T.mean(T.nnet.binary_crossentropy(p_y__x, target))


    def get_samples(self, num_examples):

        return self.gen_network.get_samples(num_examples)


    def load(self, np_params_dis, np_params_gen):

        print 'load gen'
        for param, np_param in zip(self.gen_network.params, np_params_gen):
            print param.get_value().shape, np_param.shape
            param.set_value(np_param)

        print 'load disc' 
        for param, np_param in zip(self.dis_network.params, np_params_dis):
            print param.get_value().shape, np_param.shape
            param.set_value(np_param)



