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
import theano
import theano.tensor as T

import numpy as np

from conv_layer import *
from batch_norm_conv_layer import *
from utils import *
from util_cifar10 import *


class convnet32():

    def __init__(self, model_params, nkerns=[3,8,4,2], ckern=128, filter_sizes=[5,5,5,5,4]):
        """Initializes the architecture of the discriminator"""

        self.num_hid, num_dims, num_class, self.batch_size, self.num_channels = model_params
        self.D      = int(np.sqrt(num_dims / self.num_channels))
        numpy_rng   = np.random.RandomState(1234)
        #numpy_rng = np.random.RandomState(42)

        self.nkerns         = np.asarray(nkerns) * ckern # of constant gen filters in first conv layer
        self.nkerns[0]      = self.num_channels
        self.filter_sizes   = filter_sizes
        num_convH           = self.nkerns[-1]*filter_sizes[-1]*filter_sizes[-1]

        self.W_y    = initialize_weight(num_convH, num_class,  'W_y', numpy_rng, rng_dist='normal') 

        #self.L1 =    Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[1], bnkern=self.nkerns[0], bfilter_sz=filter_sizes[0], tfilter_sz=filter_sizes[1])
        self.L1 = BN_Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[1], bnkern=self.nkerns[0] , bfilter_sz=filter_sizes[0], tfilter_sz=filter_sizes[1])
        self.L2 = BN_Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[2], bnkern=self.nkerns[1] , bfilter_sz=filter_sizes[1], tfilter_sz=filter_sizes[2])
        self.L3 = BN_Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[3], bnkern=self.nkerns[2] , bfilter_sz=filter_sizes[2], tfilter_sz=filter_sizes[3])

        self.num_classes = num_class
        self.params = self.L1.params + self.L2.params +\
                      self.L3.params + [self.W_y]


    def propagate(self, X, num_train=None, atype='relu', reshapeF=True):

        """Propagate, return binary output of fake/real image"""

        if reshapeF: 
            image_shape0=[X.shape[0], self.num_channels, self.D, self.D]
            X = X.reshape(image_shape0)

        H0 = self.L1.conv(X, atype=atype)
        H1 = self.L2.conv(H0, atype=atype)
        H2 = self.L3.conv(H1, atype=atype) 
        H2 = H2.flatten(2)

        y  = T.nnet.sigmoid(T.dot(H2, self.W_y))    

        return y

   
    def weight_decay_l2(self):

        return T.sum(self.W_y**2)
        #return (self.L1.l2_norm() + T.sum(self.W_y**2) + \
        #        self.L2.l2_norm() + self.L3.l2_norm())


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


    def load(self, folder):
                
        path_base='uncond_dcgan_0.0001_5_discrim_params.jl'
        
        filenames = [folder+path_base+'_%02d.npy'% i for i in np.arange(1,9)]
        
        weight_list = []
        
        for filename in filenames:
            
            weight_list.append(np.load(filename))

        
        for param,weight in zip(self.params,weight_list):
            
            param.set_value(weight)
            

    def display(self, images, tile_shape=(10,10), img_shape=(32,32), fname=None):
        
        DD = img_shape[0] * img_shape[1]
        images = (images[:, 0:DD],images[:, DD:DD*2],images[:, DD*2:DD*3], None)
        x = tile_raster_images(images, img_shape=img_shape, \
                                tile_shape=tile_shape, tile_spacing=(1,1), output_pixel_vals=False, scale_rows_to_unit_interval=False)
                                
        from PIL import Image

        image = Image.fromarray(np.uint8(x[:,:,0:3]))
        
        image.show()
        
        
    
    def test(self, train_set):
        
        Xu = T.fmatrix('X')
        prop = self.propagate(Xu, atype='leaky', reshapeF=True)
        # prop_out = theano.function([],outputs=prop, givens={Xu:train_set[0][1*100:2*100]})
        test_train = np.array(train_set[0][0:128], dtype=theano.config.floatX)
        print(prop.eval({Xu : test_train}))
        # MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))
        
        # samples = self.get_samples(100, MRG).reshape((100, 32*32*3))

        # self.display(np.asarray((samples.eval()+1) * 127.5, dtype='int32'))


if __name__ == "__main__":

    datapath= '/work/djkim117/cifar10/cifar-10-batches-py/'
    train_set, valid_set, test_set = load_cifar10(path=datapath)
    # 127.5 - 1. in order to rescale to -1 to 1.
    train_set[0] = train_set[0] / 127.5 - 1.
    valid_set[0] = valid_set[0] / 127.5 - 1.
    test_set[0]  = test_set[0]  / 127.5 - 1.
 
    N ,D = train_set[0].shape; Nv,D = valid_set[0].shape; Nt,D = test_set[0].shape    
    # train_set = shared_dataset(train_set)
    valid_set = shared_dataset(valid_set)
    test_set  = shared_dataset(test_set)

        # TODO: print cost
    # load model weights. don't have to i think..

    # D=32*32*3
    num_channel = 3 # FIXED

    # CONV (DISC)
    conv_num_hid= 100
    num_channel = 3 # FIXED
    num_class   = 1 # FIXED

    ### OPT PARAMS
    batch_sz    = 100
    
    disc_params = [conv_num_hid, D, num_class, batch_sz, num_channel] 
   
    # disc_params  = [batch_sz, D, num_hids, np_rng, num_z, nkerns, ckern, num_channel, num_steps]
    
    disc = convnet32(disc_params)
    # TODO: load weights of the disc 
    disc.load(folder='/work/djkim117/params/gap/dcgan/models/')
    disc.test(train_set)


