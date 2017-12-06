def our_dvae_gen():

    ##################
    ## Hyper-params ##
    ##################
    batch_sz      = 100
    epsilon       = 0.0001
    momentum      = 0.0
    num_epoch     = 5000
    num_hid       = 200
    num_hids      = [num_hid]
    num_z         = 50
    num_class     = 10
    num_trial     = 1
    binaryF       = True
    CrossEntropyF = True
    corrupt_in    = 0.0
    model_type    = 'dvae'
    D             = 28*28
 
    data_type     = 'mnist'
  
    print 'data Type: ' + data_type
    ntype         = 'salt_pepper'

    import numpy as np
    rng = np.random.RandomState(1234)
    model_params = [batch_sz, D, num_hids, rng, num_z, binaryF]   
    
    import sys
    sys.path.append('..')
    sys.path.append('../../')
    
    from dvae.dvae import DVAE2
    
    return DVAE2(model_params)


def our_gran_gen():
    
    ### MODEL PARAMS
    # CONV (DISC)
    conv_num_hid= 32
    num_channel = 1#3 # FIXED
    num_class   = 1 # FIXED
    D           = 28*28#64*64*3

    # ganI (GEN)
    filter_sz   = 7 #FIXED
    nkerns      = [8,4,1]
    ckern       = 20
    num_hid1    = nkerns[0]*ckern*filter_sz*filter_sz # FIXED.
    num_hids = [num_hid1]
    num_steps   = 3 # time steps
    num_z       = 60 

    ### OPT PARAMS
    batch_sz    = 100
    # if mname=='GRAN':
#         epsilon_dis = 0.0001 #halved both lr will give greyish lsun samples
#         epsilon_gen = 0.0002 #halved both lr will give greyish lsun samples
#     elif mname=='DCGAN':
#         epsilon_dis = 0.00005
#         epsilon_gen = 0.0001
    momentum    = 0.0 #Not Used
    lam1        = 0.00001#0.000001 

    ### TRAIN PARAMS
    # if mname=='GRAN':
#         num_epoch   = 30
#     elif mname=='DCGAN':
#         num_epoch   = 100
#         input_width = 28
#         input_height = 28
#         input_depth = 1
    epoch_start = 0 
    contF       = False #continue flag. usually FIXED
    N=50000 
    Nv=N 
    Nt=N #Dummy variable
    import numpy as np
    rng = np.random.RandomState(1234)
    
    gen_params =[batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps]
    
    
    return RecGenI28(gen_params)


def our_dcgan_gen():
    
    N=40000 
    Nv=N 
    Nt=N #Dummy variable
    import numpy as np
    rng = np.random.RandomState(1234)
    
    import sys
    sys.path.append('..')
    sys.path.append('../../')
    
    from main_dcgan import (batch_sz, D, num_hids, num_z, 
                            nkerns, ckern, num_channel)
    from main_dcgan import ltype
    print 'running %s' % ltype

    gen_params =[batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel]
    
    from genI32 import Gen32
    
    return Gen32(gen_params), ltype

def gan_gen_net10():
    
    import theano
    import lasagne


    tanh = lasagne.nonlinearities.tanh
    sigmoid = lasagne.nonlinearities.sigmoid
    linear = lasagne.nonlinearities.linear
    nonlin = tanh
    
    network = lasagne.layers.InputLayer(shape=(None, 10))
    network = lasagne.layers.DenseLayer(
                network, 64, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 1024, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 784, nonlinearity=sigmoid)
    return network

def vae_gen_net10():
    
    import theano
    import lasagne


    tanh = lasagne.nonlinearities.tanh
    sigmoid = lasagne.nonlinearities.sigmoid
    linear = lasagne.nonlinearities.linear
    nonlin = tanh
    
    network = lasagne.layers.InputLayer(shape=(None, 10))
    network = lasagne.layers.DenseLayer(
                network, 64, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 1024, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 784*2, nonlinearity=sigmoid)
    return network

def enc_net10():
    
    import theano
    import lasagne


    tanh = lasagne.nonlinearities.tanh
    sigmoid = lasagne.nonlinearities.sigmoid
    linear = lasagne.nonlinearities.linear
    nonlin = tanh
    
    network = lasagne.layers.InputLayer(shape=(None, 784))
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 64,nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 20,nonlinearity=linear)
    return network

def gen_net50():
    
    import theano
    import lasagne


    tanh = lasagne.nonlinearities.tanh
    sigmoid = lasagne.nonlinearities.sigmoid
    linear = lasagne.nonlinearities.linear
    nonlin = tanh
    
    network = lasagne.layers.InputLayer(shape=(None, 50))
    network = lasagne.layers.DenseLayer(network, 1024, nonlinearity=lasagne.nonlinearities.tanh)
    network = lasagne.layers.DenseLayer(network, 1024, nonlinearity=lasagne.nonlinearities.tanh)
    network = lasagne.layers.DenseLayer(network, 1024, nonlinearity=lasagne.nonlinearities.tanh)
    network = lasagne.layers.DenseLayer(network, 784, nonlinearity=lasagne.nonlinearities.sigmoid)
    return network 

def enc_net50():
    
    import theano
    import lasagne


    tanh = lasagne.nonlinearities.tanh
    sigmoid = lasagne.nonlinearities.sigmoid
    linear = lasagne.nonlinearities.linear
    nonlin = tanh
    
    network = lasagne.layers.InputLayer(shape=(None, 784))
    network = lasagne.layers.DenseLayer(
                network, 1024, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 64, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 100, nonlinearity=linear)
    return network
 
