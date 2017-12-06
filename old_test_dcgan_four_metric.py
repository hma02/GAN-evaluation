import time, timeit

import numpy as np
import scipy as sp
import os, sys
import gzip

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datapath='/home/imj/data/cifair10/cifar-10-batches-py/'
datapath='/home/daniel/Documents/data/cifar10/cifar-10-batches-py/'
save_path='./figs-params-cifar10/'


def train(model, train_params, num_batchs, theano_fns, opt_params, model_params, sample):

    ganI_params, conv_params = model_params 
    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params   
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel           = ganI_params
    num_epoch, epoch_start, contF                                           = train_params 
    num_batch_train, num_batch_valid, num_batch_test                        = num_batchs
    get_samples, mnnd_update, get_valid_cost, get_test_cost = theano_fns


        
    mtype = os.environ['MTYPE']
    
    # print '...Start Training'
    findex= str(num_hids[0])+'_'
    best_vl = np.infty    
    k=0 #FIXED
    constant=4;

    tr_costs, vl_costs, te_costs = [],[], []
    
    # print 'K:::%d constant:::%d' % (K, constant)
    num_epoch = 10
    
    exec_start = timeit.default_timer()
    
    for epoch in xrange(num_epoch+1):
        
        for batch_i in xrange(num_batch_train):
            costs=[[],[], []]
            
            if constant**k == batch_i+1: 
                from utils import get_epsilon_decay
                eps_dis = 0.1 * epsilon_dis * get_epsilon_decay(k+1, 100, constant)
                k+=1
          
            cost_mnnd_i = mnnd_update(batch_i, lr=eps_dis, alpha=10)
            costs[0].append(cost_mnnd_i)

            if batch_i % 100 == 0 or batch_i < 3:

                costs_vl = [[],[],[]]
                for batch_j in xrange(num_batch_valid):
                    cost_mnnd_vl_j = get_valid_cost(batch_j)
                    costs_vl[0].append(cost_mnnd_vl_j)
            
                costs_te = [[],[],[]]
                for batch_j in xrange(num_batch_test):
                    cost_mnnd_te_j = get_test_cost(batch_j)
                    costs_te[0].append(cost_mnnd_te_j)
                
                cost_mnnd_vl = np.mean(np.asarray(costs_vl[0]))
                cost_mnnd_te = np.mean(np.asarray(costs_te[0]))
                cost_mnnd_tr = np.mean(np.asarray(costs[0]))

                tr_costs.append(cost_mnnd_tr)
                vl_costs.append(cost_mnnd_vl)
                te_costs.append(cost_mnnd_te)
                print cost_mnnd_vl,
                # print 'Update %d, epsilon_dis %f5, alpha %f5, tr disc %g vl disc %g, te disc %g'\
                #         % (batch_i, eps_dis, alpha, cost_mnnd_tr, cost_mnnd_vl, cost_mnnd_te)

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
    te_score = te_costs[idx]
    vl_start = vl_costs[0]

    print os.environ['LOAD_EPOCH'], vl_start, vl_score, te_score

    return te_score
    
   

def load_model(model_params, contF=True):
    
    from dcgan import DCGAN
    import os
    model = DCGAN(model_params, ltype=os.environ['LTYPE'])
    if contF:
        # print '...Continuing from Last time'''
        from utils import unpickle
        _model = unpickle(os.environ['LOAD_PATH'])
        
        np_gen_params= [param.get_value() for param in _model.gen_network.params]
        np_dis_params= [param.get_value() for param in _model.dis_network.params]
        
        model.load(np_dis_params, np_gen_params, verbose=False)
        
    return model


def set_up_train(gan, mnnd, train_set, valid_set, test_set, opt_params,sample):

    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params  
    opt_params    = batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt 
    compile_start = timeit.default_timer()
    from optimize_dcgan import Optimize
    opt           = Optimize(opt_params)
    
    if sample==False:
        import os
        mnnd_update, get_valid_cost, get_test_cost\
                        = opt.optimize_mnnf(gan.gen_network, mnnd, train_set, valid_set, test_set, lam1=lam, ltype=os.environ['LTYPE'], mtype=os.environ['MTYPE'])
    else:
        mnnd_update, get_valid_cost, get_test_cost = None, None, None
        
    get_samples     = opt.get_samples(gan)
    compile_finish = timeit.default_timer() 
    # print 'Compile Time %f ' % ( compile_finish - compile_start) 

    return opt, get_samples, mnnd_update, get_valid_cost, get_test_cost

def main(train_set, valid_set, test_set, opt_params, ganI_params, train_params, conv_params, sample):

    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params  
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel   = ganI_params 
    conv_num_hid, D, num_class, batch_sz, num_channel, kern         = conv_params  
    num_epoch, epoch_start, contF                                   = train_params 

    # compute number of minibatches for training, validation and testing
    num_batch_train = N  / batch_sz
    num_batch_valid = Nv / batch_sz
    num_batch_test  = Nt / batch_sz

    model_params = [ganI_params, conv_params]
    ganI = load_model(model_params, contF)
    
    from convnet_cuda32 import convnet32
    import os
    # print int(os.environ['CRI_KERN']), 'critic ckern'
    conv_params[-1]=int(os.environ['CRI_KERN'])
    
    
    for mtype in ['ls', 'iw']:
        
        os.environ['MTYPE']=mtype
    
        mnnd = convnet64(conv_params, ltype=os.environ['MTYPE']) #even complicated than disciminator used in train time
        #from neural_networks import hidden_layer
        #mnnd = hidden_layer(3072, 1, 'layer1') #single linear layer - simplest mode
        opt, get_samples, mnnd_update, get_valid_cost, get_test_cost\
                                        = set_up_train(ganI, mnnd, train_set, valid_set, test_set, opt_params,sample)
        theano_fns = [get_samples, mnnd_update, get_valid_cost, get_test_cost]
        num_batchs = [num_batch_train, num_batch_valid, num_batch_test]
        
        te_cost = train(ganI, train_params, num_batchs, theano_fns, opt_params, model_params, sample)

        
        if mtype=='ls':
            te_score_ls = te_cost
        elif mtype =='iw':
            te_score_iw = te_cost
        else:
            te_score_js = te_cost
    
        
    mmd_te, is_sam = mmd_is(ganI, train_params, num_batchs, theano_fns, opt_params, model_params, sample)
    
    
    return te_score_ls, te_score_iw , mmd_te , is_sam


def mmd_is(ganI, train_params, num_batchs, theano_fns, opt_params, model_params, sample):
    
    ganI_params, conv_params = model_params 
    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params   
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel           = ganI_params
    num_epoch, epoch_start, contF                                           = train_params 
    num_batch_train, num_batch_valid, num_batch_test                        = num_batchs
    get_samples, mnnd_update, get_valid_cost, get_test_cost = theano_fns
    
    
    num_samples=5000
    samples = get_samples(num_samples)
    
    # global train_set_np, test_set_np
    
    ######
    # IS #
    ######
    
    # print 'IS'
    
    import main_resnet
    py_x = main_resnet.main(n=5, model=nnd_path+'/cifar10_deep_residual_model.npz', verbose=False)

    # vl_pred = py_x(valid_set_np[0].reshape((Nv, 3,32,32)).astype('float32'))
    dist_y =[]
    for c in xrange(10):
        count_c = np.where(train_set_np[1] == c)[0].shape[0] 
        dist_y.append(count_c)
    dist_y = np.asarray(dist_y)/float(train_set_np[1].shape[0])
    
    
    # RUN
    
    sam_pred= py_x(samples)

    #is_valid, _ = main_resnet.inception_score_from(vl_pred)           
    # is_valid = np.mean(np.sum(vl_pred * np.log(vl_pred) - (vl_pred * np.log(dist_y )), axis=1))

    #is_sample, _ = main_resnet.inception_score_from(sam_pred)
    is_sample = np.mean(np.sum(sam_pred * np.log(sam_pred) - (sam_pred * np.log(dist_y )), axis=1))
    # is_vl=np.exp(is_valid)
    is_sam=np.exp(is_sample)

    # print 'CIFAR10 IS SCORE VL %f SAM %f' % (is_vl,is_sam)
    
        
    #######    
    # MMD #
    #######
    
    # print 'MMD'
    
    from eval_gan_mmd import mix_rbf_mmd2
    
    bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    _samples = samples.reshape((num_samples, 32*32*3))
    mmd_te_score = mix_rbf_mmd2(test_set_np[0], _samples, sigmas=bandwidths)
    # mmd_vl_score = mix_rbf_mmd2(valid_set_np[0][:10000], _samples, sigmas=bandwidths)
    
    
    
    return mmd_te_score, is_sam


model_param_save = 'wdcgan_param_cifar10_'
    
def run(rng_seed,ltype, mtype,load_path, load_epoch, sample=False, nclass=10, whichclass=None, verbose=False, class_list=None, ckernr=None, cri_ckern=None):
    
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
    nkerns      = [1,8,4,2,1]
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
    
    from util_cifar10 import load_cifar10
    from utils import shared_dataset, unpickle
    
    
    import pwd; username = pwd.getpwuid(os.geteuid()).pw_name
    
    global nnd_path
    if username in ['hma02', 'mahe6562']:
        if username=='hma02':
            datapath = '/mnt/data/hma02/data/cifar10/cifar-10-batches-py/'
            save_path = '/mnt/data/hma02/gap/dcgan-cifar10/'
            nnd_path = '/mnt/data/hma02/gap/'
        else:
            datapath = '/scratch/g/gwtaylor/mahe6562/data/cifar10/cifar-10-batches-py/'
            save_path = '/scratch/g/gwtaylor/mahe6562/gap/dcgan-cifar10/'
            nnd_path = '//scratch/g/gwtaylor/mahe6562/gap/'
            
        import time; date = '%d-%d' % (time.gmtime()[1], time.gmtime()[2])
        import os; worker_id = os.getpid()
        save_path+= date+'-%d-%s/' % (worker_id,ltype)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path); print 'create dir',save_path
        #
        # save_the_env(dir_to_save='../mnist', path=save_path)
        
    global train_set_np,valid_set_np,test_set_np
    
    train_set_np, valid_set_np, test_set_np = load_cifar10(path=datapath, verbose=False)
    # 127.5 - 1. in order to rescale to -1 to 1.
    
    
    train_set_np[0] = train_set_np[0] / 255.0 #127.5 - 1.
    valid_set_np[0] = valid_set_np[0] / 255.0 #127.5 - 1.
    test_set_np[0]  = test_set_np[0]  / 255.0 #127.5 - 1.
    
    N ,D = train_set_np[0].shape; Nv,D = valid_set_np[0].shape; Nt,D = test_set_np[0].shape
    
    train_set = shared_dataset(train_set_np)
    valid_set = shared_dataset(valid_set_np)
    test_set  = shared_dataset(test_set_np )

    # print 'batch sz %d, epsilon gen %g, epsilon dis %g, hnum_z %d, num_conv_hid %g, num_epoch %di, lam %g' % \
#                                     (batch_sz, epsilon_gen, epsilon_dis, num_z, conv_num_hid, num_epoch, lam)

    book_keeping = []

    num_hids     = [num_hid1]
    train_params = [num_epoch, epoch_start, contF]
    opt_params   = [batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam]    
    ganI_params  = [batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel]
    conv_params  = [conv_num_hid, D, num_class, batch_sz, num_channel, kern]
    
    if sample==True:
        samples = main(train_set, valid_set, test_set, opt_params, ganI_params, train_params, conv_params, sample)
        return 0,0,0,0
    else:
        te_score_ls, te_score_iw , mmd_te , is_sam = main(train_set, valid_set, test_set, opt_params, ganI_params, train_params, conv_params, sample)
    
        return te_score_ls, te_score_iw , mmd_te , is_sam
    
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description="test_dcgan")
    
    parser.add_argument("-d","--device", type=str, default='cuda0',
                            help="the theano context device to be used",
                            required=True)
                            
    parser.add_argument("-r","--rngseed", type=int, default=1234,
                            help="which rng seed to be used in training",
                            required=False)
                            
    ## experiment arguments:
                            
    parser.add_argument("-c","--nclass", type=str, default='10',
                            help="which nclass to be experimented",
                            required=True)
    parser.add_argument("-w","--whichclass", type=str, default='0',
                            help="which class to be experimented",
                            required=True)
                            
    args = parser.parse_args()
    
    nclass=int(args.nclass)
    
    
    device=args.device
    backend='cudandarray' if device.startswith('gpu') else 'gpuarray'
    os.environ['THEANO_BACKEND'] = backend
    os.environ['RNG_SEED'] = str(args.rngseed)

    if backend=='cudandarray':
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(device)
    else:
        import os
        if 'THEANO_FLAGS' in os.environ:
            raise ValueError('Use theanorc to set the theano config')
        os.environ['THEANO_FLAGS'] = 'device={0}'.format(device)
        import theano.gpuarray
        ctx=theano.gpuarray.type.get_context(None)
    
    import theano 
    import theano.sandbox.rng_mrg as RNG_MRG
    rng = np.random.RandomState(int(os.environ['RNG_SEED']))
    MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))
    from optimize_dcgan import *
    from dcgan import *
    from utils import * 
    from util_cifar10 import *
    
    
    import pwd; username = pwd.getpwuid(os.geteuid()).pw_name
    if username in ['hma02', 'mahe6562']:
        if username=='hma02':
            datapath = '/mnt/data/hma02/data/cifar10/cifar-10-batches-py/'
            save_path = '/mnt/data/hma02/gap/dcgan-cifar10/'
            nnd_path = '/mnt/data/hma02/gap/'
        else:
            datapath = '/scratch/g/gwtaylor/mahe6562/data/cifar10/cifar-10-batches-py/'
            save_path = '/scratch/g/gwtaylor/mahe6562/gap/dcgan-cifar10/'
            nnd_path = '//scratch/g/gwtaylor/mahe6562/gap/'
        import time; date = '%d-%d' % (time.gmtime()[1], time.gmtime()[2])
        import os; worker_id = os.getpid()
        save_path+= date+'-%d-%s/' % (worker_id,ltype)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path); print 'create dir',save_path
        # save_the_env(dir_to_save='../cifar10', path=save_path)
        
    if nndF:
        X = T.matrix('X')
        logistic_reg = unpickle(nnd_path+'/best_model.pkl')
        get_lr_pred = theano.function([X], logistic_reg.forward(X))

    import cPickle, gzip
    f = gzip.open(datapath, 'rb')
    train_set_np, valid_set_np, test_set_np = cPickle.load(f)
    f.close()

    N ,D = train_set_np[0].shape; Nv,D = valid_set_np[0].shape; Nt,D = test_set_np[0].shape
    train_set = shared_dataset(train_set_np)
    valid_set = shared_dataset(valid_set_np)
    test_set  = shared_dataset(test_set_np )

    print 'batch sz %d, epsilon gen %g, epsilon dis %g, hnum_z %d, num_conv_hid %g, num_epoch %di, lam %g' % \
                                    (batch_sz, epsilon_gen, epsilon_dis, num_z, conv_num_hid, num_epoch, lam)

    book_keeping = []

    num_hids     = [num_hid1]
    train_params = [num_epoch, epoch_start, contF]
    opt_params   = [batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam]    
    ganI_params  = [batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel]
    conv_params  = [conv_num_hid, D, num_class, batch_sz, num_channel]
    min_vl_cost = main(train_set, valid_set, test_set, opt_params, ganI_params, train_params, conv_params)
    book_keeping.append(min_vl_cost)
    print  book_keeping


