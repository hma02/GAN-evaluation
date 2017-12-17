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


import time, timeit
import hickle as hkl

import numpy as np
import scipy as sp
import os, sys, glob
import gzip

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train(model, train_params, num_batchs, theano_fns, opt_params, model_params):

    ganI_params, conv_params = model_params 
    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params   
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps= ganI_params
    num_epoch, epoch_start, contF, train_filenames, valid_filenames, test_filenames = train_params 
    num_batch_train, num_batch_valid, num_batch_test                        = num_batchs
    get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost = theano_fns
    
    assert(mtype!=None and mtype!='')
    
    
    
    
    train_lmdb = '/scratch/g/gwtaylor/mahe6562/data/lsun/lmdb/bedroom_train_64x64'
    valid_lmdb = '/scratch/g/gwtaylor/mahe6562/data/lsun/lmdb/bedroom_val_64x64'
    from input_provider import ImageProvider
    p_train = ImageProvider(train_lmdb,batch_sz)
    p_valid = ImageProvider(valid_lmdb,batch_sz)
     
     
    
    print '...Start Testing'
    findex= str(num_hids[0])+'_'
    best_vl = np.infty    
    k=0 #FIXED
    constant=4
    
    tr_costs, vl_costs, te_costs = [],[], []
    
    num_epoch = 10
    
    exec_start = timeit.default_timer()
    
    for epoch in xrange(num_epoch+1):

        for batch_i in xrange(p_train.num_batches):
            
            costs=[[],[], []]
            
            if constant**k == batch_i+1: 
                from base.utils import get_epsilon_decay
                eps_dis = 0.1 * epsilon_dis * get_epsilon_decay(k+1, 100, constant)
                k+=1
                
            data = p_train.next()/ 255.
            data = data.astype('float32')
            a,b,c,d = data.shape
            data = data.reshape(a,b*c*d)
                
            cost_mnnd_i = mnnd_update(data, lr=eps_dis, alpha=10)
            costs[0].append(cost_mnnd_i)
            
            
            if batch_i % 100 == 0 or batch_i < 3:

                costs_vl = [[],[],[]]
                for batch_j in xrange(p_valid.num_batches):
                    
                    data = p_valid.next()/ 255.
                    data = data.astype('float32')
                    a,b,c,d = data.shape
                    data = data.reshape(a,b*c*d)
                    
                    cost_mnnd_vl_j = get_valid_cost(data)
                    costs_vl[0].append(cost_mnnd_vl_j)
            
                # costs_te = [[],[],[]]
                # for batch_j in xrange(num_batch_test):
                #
                #     data = hkl.load(test_filenames[batch_i]) / 255.
                #     data = data.astype('float32').transpose([3,0,1,2])
                #     a,b,c,d = data.shape
                #     data = data.reshape(a,b*c*d)
                #
                #     cost_mnnd_te_j = get_test_cost(data)
                #     costs_te[0].append(cost_mnnd_te_j)
                
                cost_mnnd_vl = np.mean(np.asarray(costs_vl[0]))
                # cost_mnnd_te = np.mean(np.asarray(costs_te[0]))
                cost_mnnd_tr = np.mean(np.asarray(costs[0]))

                tr_costs.append(cost_mnnd_tr)
                vl_costs.append(cost_mnnd_vl)
                # te_costs.append(cost_mnnd_te)
                print cost_mnnd_vl,
                
    print
    
    exec_finish = timeit.default_timer() 
    if batch_i==0: print 'Exec Time %f ' % ( exec_finish - exec_start) 
   
    # print 'NND Tr: ', tr_costs
    # print 'NND Vl: ', vl_costs
    # print 'NND Te: ', te_costs

    # save_path=os.environ['LOAD_PATH']
    
    # np.save(save_path+'/%s_mnnd_tr.npy' % mtype,tr_costs)
    # np.save(save_path+'/%s_mnnd_te.npy' % mtype,te_costs)
    # np.save(save_path+'/%s_mnnd_vl.npy' % mtype,vl_costs)

    def find_farthest(array, value, sign):
        
        # when sign is negative, the array is decreasing

        assert sign!=0

        idx = (sign*(np.array(array)-value)).argmax()
        return array[idx], idx
    if mtype=='iw':
        sign=1
    else:
        sign=-1
    vl_score, idx = find_farthest(vl_costs, vl_costs[0], sign=sign)
    te_score = vl_score # te_costs[idx]
    vl_start = vl_costs[0]

    print os.environ['LOAD_EPOCH'], vl_start, vl_score, te_score

    return te_score


def load_model(model_params, contF=True):

    print '...Starting from the beginning'''
    if mname=='GRAN':
        model = GRAN(model_params, ltype)
    elif mname=='DCGAN':
        model = DCGAN(model_params, ltype)
        
    if contF:
        
        # print '...Continuing from Last time'''
        from base.utils import unpickle
        _model = unpickle(os.environ['LOAD_PATH'])
        
        np_gen_params= [param.get_value() for param in _model.gen_network.params]
        np_dis_params= [param.get_value() for param in _model.dis_network.params]
        
        model.load(np_dis_params, np_gen_params, verbose=False)

    return model 


def set_up_train(gan, mnnd, opt_params):

    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params
    if mname=='GRAN':
        opt_params    = batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt
    elif mname=='DCGAN':
        opt_params    = batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, input_width, input_height, input_depth
    compile_start = timeit.default_timer()
    opt           = Optimize(opt_params)
    
    import os
    mnnd_update, get_valid_cost, get_test_cost \
                    = opt.optimize_mnnf(gan.gen_network, mnnd, lam1=lam, mtype=os.environ['MTYPE'])
                    

    get_samples     = opt.get_samples(model)
    compile_finish = timeit.default_timer() 
    # print 'Compile Time %f ' % ( compile_finish - compile_start) 
    return opt, get_samples, mnnd_update, get_valid_cost, get_test_cost


def main(opt_params, ganI_params, train_params, conv_params):

    batch_sz, epsilon_gen, epsilon_dis,  momentum, num_epoch, N, Nv, Nt, lam    = opt_params  
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps    = ganI_params 
    conv_num_hid, D, num_class, batch_sz, num_channel, kern                         = conv_params  
    num_epoch, epoch_start, contF,train_filenames, valid_filenames, test_filenames  = train_params 
    num_batch_train = len(train_filenames)
    num_batch_valid = len(valid_filenames)
    num_batch_test  = len(test_filenames)

    model_params = [ganI_params, conv_params]
    ganI = load_model(model_params, contF)
    
    
    from convnet_cuda64 import convnet64
    import os
    # print int(os.environ['CRI_KERN']), 'critic ckern'
    conv_params[-1]=int(os.environ['CRI_KERN'])
    
    
    for mtype in ['ls', 'iw']:
        
        os.environ['MTYPE']=mtype
    
        mnnd = convnet64(conv_params, ltype=os.environ['MTYPE']) #even complicated than disciminator used in train time
        #from neural_networks import hidden_layer
        #mnnd = hidden_layer(3072, 1, 'layer1') #single linear layer - simplest mode
        opt, get_samples, mnnd_update, get_valid_cost, get_test_cost = set_up_train(ganI, mnnd, opt_params)
        theano_fns = [get_samples, mnnd_update, get_valid_cost, get_test_cost]
        num_batchs = [num_batch_train, num_batch_valid, num_batch_test]
        
        te_cost = train(ganI, train_params, num_batchs, theano_fns, opt_params, model_params)

        
        if mtype=='ls':
            te_score_ls = te_cost
        elif mtype =='iw':
            te_score_iw = te_cost
        else:
            te_score_js = te_cost
        
    mmd_te= mmd_is(ganI, train_params, num_batchs, theano_fns, opt_params, model_params)
    
    return te_score_ls, te_score_iw , mmd_te #, is_sam

def mmd_is(ganI, train_params, num_batchs, theano_fns, opt_params, model_params, sample):
    
    ganI_params, conv_params = model_params 
    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params   
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps  = ganI_params
    num_epoch, epoch_start, contF,train_filenames, valid_filenames, test_filenames = train_params 
    num_batch_train, num_batch_valid, num_batch_test                        = num_batchs
    get_samples, mnnd_update, get_valid_cost, get_test_cost = theano_fns
    
    
    num_samples=5000
    samples = get_samples(num_samples)
    
    # global train_set_np, test_set_np
    
    # ######
#     # IS #
#     ######
#
#     # print 'IS'
#
#     import main_resnet
#     py_x = main_resnet.main(n=5, model=nnd_path+'/cifar10_deep_residual_model.npz', verbose=False)
#
#     # vl_pred = py_x(valid_set_np[0].reshape((Nv, 3,32,32)).astype('float32'))
#     dist_y =[]
#     for c in xrange(10):
#         count_c = np.where(train_set_np[1] == c)[0].shape[0]
#         dist_y.append(count_c)
#     dist_y = np.asarray(dist_y)/float(train_set_np[1].shape[0])
#
#
#     # RUN
#
#     sam_pred= py_x(samples)
#
#     #is_valid, _ = main_resnet.inception_score_from(vl_pred)
#     # is_valid = np.mean(np.sum(vl_pred * np.log(vl_pred) - (vl_pred * np.log(dist_y )), axis=1))
#
#     #is_sample, _ = main_resnet.inception_score_from(sam_pred)
#     is_sample = np.mean(np.sum(sam_pred * np.log(sam_pred) - (sam_pred * np.log(dist_y )), axis=1))
#     # is_vl=np.exp(is_valid)
#     is_sam=np.exp(is_sample)
#
#     # print 'CIFAR10 IS SCORE VL %f SAM %f' % (is_vl,is_sam)
    
        
    #######    
    # MMD #
    #######
    
    # print 'MMD'
    
    from test.eval_gan_mmd import mix_rbf_mmd2
    
    bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    _samples = samples.reshape((num_samples, 64*64*3))
    mmd_te_score = mix_rbf_mmd2(test_set_np[0], _samples, sigmas=bandwidths)
    # mmd_vl_score = mix_rbf_mmd2(valid_set_np[0][:10000], _samples, sigmas=bandwidths)
    
    
    return mmd_te_score #, is_sam


def run(rng_seed,ltype, mtype,load_path, load_epoch, verbose=False, ckernr=None, cri_ckern=None):
    
    assert ckernr!=None
    #  ltype -> GAN LSGAN WGAN 
    #    JS      0.4+-asdf
    #    LS
    #    WA
    #    MMD 
    #    IS


    ### MODEL PARAMS
    ### MODEL PARAMS
    # ltype       = sys.argv[3]
    # mtype       = 'js'
    # print 'ltype: ' + ltype
    # print 'mtype: ' + mtype
    mmdF        = False
    nndF        = False

    # CONV (DISC)
    conv_num_hid= 100
    num_channel = 3 #Fixed
    num_class   = 1 #Fixed
    D=64*64*3
    kern=int(ckernr.split('_')[0])

    ### OPT PARAMS
    batch_sz    = 100
    momentum    = 0.0 #Not Used
    lam         = 0.0
    
    epsilon_dis = 0.0002
    epsilon_gen = 0.0001
    
    # if mtype =='js' :
    #     epsilon_dis = 0.0002
    #     epsilon_gen = 0.0001
    #     K=5 #FIXED
    #     J=1
    # elif mtype == 'ls':
    #     epsilon_dis = 0.0002
    #     epsilon_gen = 0.0001
    #     K=5 #FIXED
    #     J=1
    # else:
    #     epsilon_dis = 0.0002
    #     epsilon_gen = 0.0001
    #     K=2 #FIXED
    #     J=1

    # ganI (GEN)
    filter_sz   = 4 #FIXED
    nkerns      = [8,4,2,1,3]
    ckern       = int(ckernr.split('_')[-1]) #20
    num_hid1    = nkerns[0]*ckern*filter_sz*filter_sz #Fixed
    num_z       = 100

    ### TRAIN PARAMS
    num_epoch   = 10
    epoch_start = 0 #Fixed
    contF       = True #Fixed
    
    num_hids     = [num_hid1]
    
    input_width = 64
    input_height = 64
    input_depth = 3
    
    N=1000 
    Nv=N 
    Nt=N #Dummy variable
    
    ### SAVE PARAM
    model_param_save = 'num_hid%d.batch%d.eps_dis%g.eps_gen%g.num_z%d.num_epoch%g.lam%g.ts%d.data.100_CONV_lsun'%(conv_num_hid,batch_sz, epsilon_dis, epsilon_gen, num_z, num_epoch, lam1, num_steps)

    
    # device=sys.argv[1]
    import os
    os.environ['RNG_SEED'] = str(rng_seed)
    os.environ['LOAD_PATH'] = load_path
    os.environ['LOAD_EPOCH'] = str(load_epoch)
    os.environ['LTYPE'] = ltype
    # os.environ['MTYPE'] = mtype
    try:
        a=os.environ['CRI_KERN']
    except:
        if cri_ckern!=None: 
            os.environ['CRI_KERN']=cri_ckern
        else:
            raise RuntimeError('cri_kern not provided')
    
    import theano 
    import theano.sandbox.rng_mrg as RNG_MRG
    rng = np.random.RandomState(int(os.environ['RNG_SEED']))
    MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))
    
    
    import pwd
    username = pwd.getpwuid(os.geteuid()).pw_name

    # if username=='djkim117':
    #     save_path = '/work/djkim117/params/gap/lsun/'
    #     datapath = '/work/djkim117/lsun/church/preprocessed_toy_100/'
    # elif username=='imj':
    #     datapath = '/work/djkim117/lsun/church/preprocessed_toy_100/'
    #     save_path = '/work/imj/gap/dcgans/lsun/dcgan4_100swap_30epoch_noise'
    if username=='mahe6562':
        datapath = '/scratch/g/gwtaylor/mahe6562/data/lsun/bedroom/preprocessed_toy_100/'


    #---
    
    # store the filenames into a list.
    train_filenames = sorted(glob.glob(datapath + 'train_hkl_b100_b_100/*' + '.hkl'))
    
    #4.shuffle train data order for each worker
    indices=np_rng.permutation(len(train_filenames))
    train_filenames=np.array(train_filenames)[indices].tolist()
    #---
    
    valid_filenames = sorted(glob.glob(datapath + 'val_hkl_b100_b_100/*' + '.hkl'))
    test_filenames = sorted(glob.glob(datapath + 'test_hkl_b100_b_100/*' + '.hkl'))

    
    num_hids     = [num_hid1]
    train_params = [num_epoch, epoch_start, contF, train_filenames, valid_filenames, test_filenames]
    opt_params   = [batch_sz, epsilon_gen, epsilon_dis,  momentum, num_epoch, N, Nv, Nt, lam1]    
    ganI_params  = [batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps]
    conv_params  = [conv_num_hid, D, num_class, batch_sz, num_channel, kern]
    book_keeping = main(opt_params, ganI_params, train_params, conv_params)

    
    te_score_ls, te_score_iw , mmd_te = main(opt_params, ganI_params, train_params, conv_params)

    return te_score_ls, te_score_iw , mmd_te #, is_sam
