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

def do_gam2(model, samples, comm, avoid_ranks, save_file_path='./gam2.csv'):
    
    rank=comm.rank
    size=comm.size
    #9.gam2 with 100 samples

    gam2=True
    
    winner_rank=None
    
    if gam2:
        
        from base.gam2 import gam2

        M = gam2(model, samples, comm)
        
        if rank==0:
            np.savetxt(save_file_path, M, delimiter=",", fmt='%.3f') # save to a csv file
        
        n = M.shape[0]
        
        for i in range(n):
            
            M[i][i]=0
            
        ranking_arr = np.sum(M,axis=0)
        
        winner_ranks = ranking_arr.argsort()[::-1].tolist()
        
        if rank==0 and size>=2: 
            print 'winners all: %d %d ...' % (winner_ranks[0], winner_ranks[1])
            
        for avoid in avoid_ranks: 
            winner_ranks.remove(avoid)
        
        winner_ranks = winner_ranks[:2] # get the highest two scores
        
        if rank==0 and size>=2:
            print 'winners gap: %d %d ...' % (winner_ranks[0], winner_ranks[1])
        
    return winner_ranks
    
def smooth_swp_lr(bk_eps_gen, bk_eps_dis, eps_gen, eps_dis, smooth_count, total_smooth_count):
    
    if smooth_count==0:
        
        pass
        
    else:
        
        # for _rank in range(size):
#             if _rank not in avoid_ranks:
#                 print 'smoothing %.6f %.6f' % (eps_gen,eps_dis)
#                 break
        
        smooth_count = smooth_count-1
        
        eps_gen = bk_eps_gen* (1+0.3/(total_smooth_count-smooth_count+0.3))
        
        eps_dis = bk_eps_dis* (1-0.3/(total_smooth_count-smooth_count+0.6))
    
    return smooth_count, eps_gen, eps_dis
            
def do_tsne():
    
    #9.t-SNE plot with 100 samples, spawning a child process from rank0 for doing this
    
    tsne=False
    
    if tsne and gam2:
        
        pass
        
        from base.tsne import tsne
        
        tsne_list = [data]
        tsne_list.extend(sample_list)
        
        if rank==0: tsne(tsne_list)
    

def lets_train(model, train_params, num_batchs, theano_fns, opt_params, model_params):

    
    #5. initialize exchanger
    if size>1:
        
        if someconfigs.backend=='gpuarray':
            from base.swp import Exch_swap_gpuarray, get_pairs, get_winner
    
            exchanger = Exch_swap_gpuarray(comm)
            exchanger.prepare(ctx, model.dis_network.params) #only swap the dis params
            
        elif someconfigs.backend=='cudandarray':
            from base.swp import Exch_swap_cudandarray, get_pairs, get_winner
            
            exchanger = Exch_swap_cudandarray(comm)
            exchanger.prepare(ctx,drv,model.dis_network.params) #only swap the dis params
        else:
            raise ValueError('wrong backend: %s' % someconfigs.backend )
        
    # print '=============='
    # for param in model.dis_network.params:
    #     print param.get_value().shape
    #
    # print '=============='
    # for param in model.gen_network.params:
    #     print param.get_value().shape
    # print '=============='
    
    #---
        
    ganI_params, conv_params = model_params 
    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params   
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps= ganI_params
    num_epoch, epoch_start, contF, train_filenames, valid_filenames, test_filenames = train_params 
    num_batch_train, num_batch_valid, num_batch_test                        = num_batchs
    get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost = theano_fns
    
    train_lmdb = '/scratch/g/gwtaylor/mahe6562/data/lsun/lmdb/bedroom_train_64x64'
    valid_lmdb = '/scratch/g/gwtaylor/mahe6562/data/lsun/lmdb/bedroom_val_64x64'
    from input_provider import ImageProvider
    p_train = ImageProvider(train_lmdb,batch_sz)
    p_valid = ImageProvider(valid_lmdb,batch_sz)
    
    samples = p_train.next(100).reshape((100, 64*64*3))
    display_images(np.asarray(samples, dtype='int32'), \
                                tile_shape = (10,10), img_shape=(64,64), \
                                fname=save_path+'/data')
    
     
    print '...Start Training'
    findex= str(num_hids[0])+'_'
    best_vl = np.infty    
    K=1 #FIXED
    num_samples =100;
    count=0
    smooth_count=0
    

    eps_gen = epsilon_gen
    for epoch in xrange(num_epoch+1):

        costs=[[],[], []]
        exec_start = timeit.default_timer()

        eps_gen = get_epsilon(epsilon_gen, num_epoch*4, epoch) #gen lr decrease slower than dis lr

        eps_dis = get_epsilon(epsilon_dis, num_epoch, epoch)
        
        bk_eps_gen = eps_gen
        
        bk_eps_dis = eps_dis
        
        total_smooth_count= int(num_batch_train * 0.05)

        for batch_i in xrange(p_train.num_batches):
            
            count+=1
            
            if count%2000==0 and rank==0:
                print 'batch: %d' % count
                
            if count%2000==0:
                
                costs_vl = [[],[],[]]

                for batch_j in xrange(p_valid.num_batches):
                    data = p_valid.next()/ 255.
                    data = data.astype('float32')
                    # if epoch < num_epoch * 0.25 :
    #                     data = np.asarray(corrupt_input(rng, data, 0.3), dtype='float32')
    #                 elif epoch < num_epoch * 0.5 :
    #                     data = np.asarray(corrupt_input(rng, data, 0.1), dtype='float32')
                    a,b,c,d = data.shape
                    data = data.reshape(a, b*c*d)
                    cost_test_vl_j, cost_gen_vl_j = get_valid_cost(data)
                    cost_sample_vl_j=0
                    costs_vl[0].append(cost_test_vl_j)
                    costs_vl[1].append(cost_sample_vl_j)
                    costs_vl[2].append(cost_gen_vl_j)
                #    print("validation success !");

                cost_test_vl = np.mean(np.asarray(costs_vl[0]))
                cost_sample_vl = np.mean(np.asarray(costs_vl[1]))
                cost_gen_vl = np.mean(np.asarray(costs_vl[2]))             

                cost_test_tr = np.mean(np.asarray(costs[0]))
                cost_sample_tr = np.mean(np.asarray(costs[1]))
                cost_gen_tr = np.mean(np.asarray(costs[2]))

                # cost_tr = cost_dis_tr+cost_gen_tr
                # cost_vl = cost_dis_vl+cost_gen_vl

                print 'Epoch %d, count %d, epsilon_gen %f5, epsilon_dis %f5, tr dis gen %g, %g, %g | vl disc gen %g, %g, %g '\
                        % (epoch, count, eps_gen, eps_dis, cost_test_tr, cost_sample_vl, cost_gen_tr, cost_test_vl, cost_sample_tr, cost_gen_vl)

                num_samples=100
                samples = get_samples(num_samples).reshape((num_samples, 64*64*3))
                display_images(np.asarray(samples * 255, dtype='int32'), \
                                            tile_shape = (10,10), img_shape=(64,64), \
                                            fname=save_path+'/' + str(epoch) +'-'+ str(count))

                #7.save curve

                # date = '-%d-%d' % (time.gmtime()[1], time.gmtime()[2])
                curve.append([cost_test_tr ,cost_test_vl , cost_sample_tr, cost_sample_vl, cost_gen_tr, cost_gen_vl ])

                np.save(save_path+'curve'+str(size)+str(rank)+'.npy', np.array(curve))

                #---

                #8.curve plot

                import matplotlib.pyplot as plt
                colors = ['-r','--r', '-b', '--b','-m', '--m','-g', '--g']
                labs = ['test_tr', 'test_vl','sample_tr', 'sample_vl', 'gen_tr', 'gen_vl']
                arrays = np.array(curve).transpose()[:4] # only show dis
                fig = plt.figure()

                for index, cost in enumerate(arrays):
                    plt.plot(cost, colors[index], label=labs[index])

                plt.legend(loc='upper right')
                plt.xlabel('epoch')
                plt.ylabel('cost')
                fig.savefig(save_path+'curve'+str(size)+str(rank)+'.png',format='png')
                #plt.show()
                plt.close('all')
                #---
                
            if count%10000==0:
                
                save_the_weight(model, save_path + 'weight'+ '-'+ str(epoch)+'-'+ str(count))
                

            
            def dcgan_update(batch_i, eps_gen, eps_dis):
                
                cost_gen_i = generator_update(lr=eps_gen)
                cost_gen_i = generator_update(lr=eps_gen)

                data = p_train.next()/ 255.
                data = data.astype('float32')
                # if epoch < num_epoch * 0.25 :
#                     data = np.asarray(corrupt_input(rng, data, 0.3), dtype='float32')
#                 elif epoch < num_epoch *0.5 :
#                     data = np.asarray(corrupt_input(rng, data, 0.1), dtype='float32')
                a,b,c,d = data.shape
                data = data.reshape(a,b*c*d)
                cost_test_i  = discriminator_update(data, lr=eps_dis)
                cost_sample_i = 0
                return cost_test_i, cost_sample_i, cost_gen_i
                
                
            def gran_update(batch_i, eps_gen, eps_dis):
                

                data = hkl.load(train_filenames[batch_i]) / 255.
                data = data.astype('float32').transpose([3,0,1,2])
                # if epoch < num_epoch * 0.25 :
#                     data = np.asarray(corrupt_input(rng, data, 0.3), dtype='float32')
#                 elif epoch < num_epoch *0.5 :
#                     data = np.asarray(corrupt_input(rng, data, 0.1), dtype='float32')
                a,b,c,d = data.shape
                data = data.reshape(a,b*c*d)

                cost_test_i  = discriminator_update(data, lr=eps_dis)
                cost_sample_i = 0
        
                if batch_i % K == 0:
                    cost_gen_i = generator_update(lr=eps_gen)
                    cost_gen_i = generator_update(lr=eps_gen)
                else:
                    cost_gen_i = 0
                    
                return cost_test_i, cost_sample_i, cost_gen_i
                    
            if mname=='GRAN': 
                cost_test_i, cost_sample_i, cost_gen_i = gran_update(batch_i, eps_gen, eps_dis)
            elif mname=='DCGAN':
                cost_test_i, cost_sample_i, cost_gen_i = dcgan_update(batch_i, eps_gen, eps_dis)
                
                
            costs[0].append(cost_test_i)
            costs[1].append(cost_sample_i)
            costs[2].append(cost_gen_i)
            

        exec_finish = timeit.default_timer() 
        print 'Exec Time %f ' % ( exec_finish - exec_start)


        if epoch % 1 == 0 or epoch > 2 or epoch == (num_epoch-1):
            
            # change the name to save to when new model is found.
            # if epoch%4==0 or epoch>(num_epoch-3) or epoch<2:
            pass
            # save_the_weight(model, save_path+str(size)+str(rank)+'dcgan_'+ model_param_save + str(epoch)+'-'+ str(count))# + findex+ str(K))
                
                
                

    num_samples=400
    samples = get_samples(num_samples).reshape((num_samples, 3*64*64))
    display_images(np.asarray(samples * 255, dtype='int32'), tile_shape=(20,20), img_shape=(64,64), \
                            fname= save_path + '/' +str(size)+str(rank)+ '_'+ findex + str(K))

    return model


def load_model(model_params, contF=True):

    if not contF:
        print '...Starting from the beginning'''
        if mname=='GRAN':
            model = GRAN(model_params, ltype)
        elif mname=='DCGAN':
            model = DCGAN(model_params, ltype)
    else:
        print '...Continuing from Last time'''
        path_name = raw_input("Enter full path to the pre-trained model: ")
        model = unpickle(path_name)

    return model 


def set_up_train(model, opt_params):

    
    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params
    if mname=='GRAN':
        opt_params    = batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt
    elif mname=='DCGAN':
        opt_params    = batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, input_width, input_height, input_depth
    compile_start = timeit.default_timer()
    opt           = Optimize(opt_params)

    print ("Compiling...it may take a few minutes")
    discriminator_update, generator_update, get_valid_cost, get_test_cost\
                    = opt.optimize_gan_hkl(model, ltype)
    get_samples     = opt.get_samples(model)
    compile_finish = timeit.default_timer() 
    print 'Compile Time %f ' % ( compile_finish - compile_start) 
    return opt, get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost


def main(opt_params, ganI_params, train_params, conv_params):

    batch_sz, epsilon_gen, epsilon_dis,  momentum, num_epoch, N, Nv, Nt, lam    = opt_params  
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps    = ganI_params 
    conv_num_hid, D, num_class, batch_sz, num_channel                           = conv_params  
    num_epoch, epoch_start, contF,train_filenames, valid_filenames, test_filenames  = train_params 
    num_batch_train = len(train_filenames)
    num_batch_valid = len(valid_filenames)
    num_batch_test  = len(test_filenames)

    model_params = [ganI_params, conv_params]
    ganI = load_model(model_params, contF)
    opt, get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost\
                                = set_up_train(ganI, opt_params)
                                
    #TODO: If you want to train your own model, comment out below section and set the model parameters below accordingly
    ##################################################################################################
#     num_samples=100
    # fname='./figs/lsun/gran_lsun_samples500.pdf'
    # samples = get_samples(num_samples).reshape((num_samples, 3*64*64))
    # display_images(np.asarray(samples * 255, dtype='int32'), tile_shape=(10,10), img_shape=(64,64),fname=fname)
    # print ("LSUN sample fetched and saved to " + fname)
#     exit()
    ###################################################################################################
    theano_fns = [get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost]
    num_batchs = [num_batch_train, num_batch_valid, num_batch_test]
    lets_train(ganI, train_params, num_batchs, theano_fns, opt_params, model_params)

if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description="main_dcgan")

    parser.add_argument("-d","--device", type=str, default='cuda0',
                     help="the theano context device to be used",
                     required=True)
    parser.add_argument("-w","--workers", type=int, default=1,
                     help="how many workers",
                     required=False)
    parser.add_argument("-m","--mname", type=str, default='DCGAN',
                     help="DCGAN OR GRAN?",
                     required=False)         
    parser.add_argument("-b","--combined", type=int, default=0,
                     help="DCGAN and GRAN combined?",
                     required=False)
                             
    parser.add_argument("-l","--ltype", type=str, default='gan',
                     help="which gan type to be used in training",
                     required=True)
    parser.add_argument("-r","--rngseed", type=int, default=1234,
                     help="which rng seed to be used in training",
                     required=True)
    
    args = parser.parse_args()
    
    ltype=args.ltype
    
    #0.initialize MPI

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank=comm.rank
    size=comm.size
    import os
    worker_id = os.getpid()
    
    sum_worker_id = comm.allreduce(worker_id)
    #---
    
    #1. parsing sys arguments

    import sys
    import base.subnets.layers.someconfigs as someconfigs
    try:
    
        device=args.device #sys.argv[1]
        someconfigs.indep_workers=args.workers #int(sys.argv[2])
        mname=args.mname #str(sys.argv[3])
    except:
        print 'USAGE: python main.py [device] [indep_worker] [mname] [combined_f](optional)'
        print 'example: device=cuda0, indep_worker=0, mname=GRAN'
        raise
        
    if size>1:
        try:
            combined_f = args.combined #int(sys.argv[4])
            if combined_f==1:
                print 'Combined'
        except:
            combined_f=0
            pass
    
    # gran=0.2, dcgan=0.2, combined=0.2
    swp_every=0.1 # swp every (num_epoch * num_batch_train * swp_every) iterations during training
    assert swp_every<=0.5 # swp at lease twice before training ends
    
    someconfigs.clipg_dis=1
    
    assert mname in ['GRAN', 'DCGAN']
    someconfigs.mname=mname
    
    if someconfigs.indep_workers==0:
        someconfigs.indep_workers=False
    else:
        someconfigs.indep_workers=True
    
    indep = [rank,someconfigs.indep_workers]
    
    all_indep = comm.allgather(indep)
    avoid_ranks = [ r for r, _indep in all_indep if _indep==True]
    
    
    # #1.5 Split into subcomm
    # if combined_f==1:
    #     if mname=='GRAN':
    #         color=1
    #     elif mname=='DCGAN':
    #         color=0
    #     subcomm = comm.Split(color)
    # #---
    
    
    #2.initialize devices
    if device.startswith('gpu'):
        backend='cudandarray'
        someconfigs.backend='cudandarray'
        
    else:
        backend='gpuarray'
        someconfigs.backend='gpuarray'

    gpuid=int(device[-1])

    if backend=='cudandarray':

        import pycuda.driver as drv
        drv.init()
        dev=drv.Device(gpuid)
        ctx=dev.make_context()

        import theano.sandbox.cuda
        theano.sandbox.cuda.use(device)

        # import pycuda.gpuarray as gpuarray
        # #import theano
        # import theano.misc.pycuda_init
        # import theano.misc.pycuda_utils
    else:
        import os
        if 'THEANO_FLAGS' in os.environ:
            raise ValueError('Use theanorc to set the theano config')
        os.environ['THEANO_FLAGS'] = 'device={0}'.format(device)
        import theano.gpuarray
        ctx=theano.gpuarray.type.get_context(None)
        # from pygpu import collectives
    #---
    
    #3.use pid to make rng different for each worker batch shuffle
    np_rng = np.random.RandomState(1234+rank) # only for shuflling files
    import base.subnets.layers.utils as utils
    utils.rng = np.random.RandomState(args.rngseed) # for init network and corrupt images
    rng = utils.rng
    
    curve=[]
    #---
    
    
    # 3.0 import things after device setup
    
    import theano 
    # import theano.sandbox.rng_mrg as RNG_MRG
    # MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))
    if mname=='GRAN':
        from base.optimize_gran import Optimize
        from base.gran import GRAN
    elif mname=='DCGAN':
        from base.optimize_gan import Optimize
        from base.dcgan import DCGAN
    # from deconv import *
    from base.utils import save_the_weight, save_the_env, get_epsilon, unpickle, corrupt_input
    from base.util_cifar10 import display_images
    
    debug = sys.gettrace() is not None
    if debug:
        theano.config.optimizer='fast_compile'
        theano.config.exception_verbosity='high'
        theano.config.compute_test_value = 'warn'
       
    # 3.05 hyper params
    
    
    
    ### MODEL PARAMS
    # CONV (DISC)
    conv_num_hid= 100
    num_channel = 3 # FIXED
    num_class   = 1 # FIXED
    D           = 64*64*3

    # ganI (GEN)
    filter_sz   = 4 #FIXED
    nkerns      = [1,8,4,2,1]
    ckern       = 172
    num_hid1    = nkerns[0]*ckern*filter_sz*filter_sz # FIXED.
    num_steps   = 3 # time steps
    num_z       = 100 

    ### OPT PARAMS
    batch_sz    = 100
    if mname=='GRAN':
        epsilon_dis = 0.0001 #halved both lr will give greyish lsun samples
        epsilon_gen = 0.0002 #halved both lr will give greyish lsun samples
    elif mname=='DCGAN':
        
        if ltype == 'gan':
            epsilon_dis = 0.00005
            epsilon_gen = 0.0001
        elif ltype =='lsgan':
            epsilon_dis = 0.0002
            epsilon_gen = 0.0004
        elif ltype =='wgan':
            epsilon_dis = 0.0002
            epsilon_gen = 0.0004
            
    momentum    = 0.0 #Not Used
    lam1        = 0.000001 

    ### TRAIN PARAMS
    if mname=='GRAN':
        num_epoch   = 15
    elif mname=='DCGAN':
        num_epoch   = 100
        input_width = 64
        input_height = 64
        input_depth = 3
    epoch_start = 0 
    contF       = False #continue flag. usually FIXED
    N=1000 
    Nv=N 
    Nt=N #Dummy variable


    ### SAVE PARAM
    model_param_save = 'num_hid%d.batch%d.eps_dis%g.eps_gen%g.num_z%d.num_epoch%g.lam%g.ts%d.data.100_CONV_lsun'%(conv_num_hid,batch_sz, epsilon_dis, epsilon_gen, num_z, num_epoch, lam1, num_steps)
    #model_param_save = 'gran_param_lsun_ts%d.save' % num_steps
    
    if someconfigs.indep_workers==0:
        print 'rank%d %s swap every %.2f epochs' % (rank, mname, swp_every*num_epoch)
    
    
    #3.1 create save path
    
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
        if mname=='GRAN':
            save_path = '/scratch/g/gwtaylor/mahe6562/gap/gran-lsun/'
        elif mname=='DCGAN':
            save_path = '/scratch/g/gwtaylor/mahe6562/gap/dcgan-lsun/'
        if size>1 and combined_f==1:
            save_path = '/scratch/g/gwtaylor/mahe6562/gap/combined-lsun/'
            
        import time
        date = '%d-%d' % (time.gmtime()[1], time.gmtime()[2])
            
        # save_path+= date+ '-swp'+str(swp_every)+ '-'+str(size)+'-'+backend+'-%d/' % sum_worker_id
        save_path+= date+ '-' + str(sum_worker_id) + '-' + ltype + '-' + str(args.rngseed) + '/' 

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print 'create dir',save_path

    #---
    
    # 3.2 save the env
    
    with open('./paths.txt','w') as f:
        string=os.path.realpath(__file__)+'\n'+save_path
        f.write(string)
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_the_env(dir_to_save=dir_path, path=save_path)
    
    #---
    
    # store the filenames into a list.
    train_filenames = sorted(glob.glob(datapath + 'train_hkl_b100_b_100/*' + '.hkl'))
    
    #4.shuffle train data order for each worker
    indices=np_rng.permutation(len(train_filenames))
    train_filenames=np.array(train_filenames)[indices].tolist()
    #---
    
    valid_filenames = sorted(glob.glob(datapath + 'val_hkl_b100_b_100/*' + '.hkl'))
    test_filenames = sorted(glob.glob(datapath + 'test_hkl_b100_b_100/*' + '.hkl'))

    print 'num_hid%d.batch sz %d, epsilon_gen %g, epsilon_disc %g, num_z %d,  num_epoch %d, lambda %g, ckern %d' % \
                                    (conv_num_hid, batch_sz, epsilon_gen, epsilon_dis, num_z, num_epoch, lam1, ckern)
    num_hids     = [num_hid1]
    train_params = [num_epoch, epoch_start, contF, train_filenames, valid_filenames, test_filenames]
    opt_params   = [batch_sz, epsilon_gen, epsilon_dis,  momentum, num_epoch, N, Nv, Nt, lam1]    
    ganI_params  = [batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps]
    conv_params  = [conv_num_hid, D, num_class, batch_sz, num_channel]
    book_keeping = main(opt_params, ganI_params, train_params, conv_params)


