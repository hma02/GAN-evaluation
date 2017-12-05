# evaluating gam2 on selected models

import numpy as np
import time

def load_model(fname, verbose=False):

    print '...loading model from %s' % fname
    
    from base.utils import unpickle
    
    if verbose: t1=time.time()
    model = unpickle(fname)
    
    import theano.tensor as T
    num_sam = T.iscalar('i')
    get_samples = theano.function([num_sam], model.get_samples(num_sam))
    if verbose: print 'loading time:%.3fs' % (time.time()-t1)

    return model, get_samples
    
def gen_samples(get_samples,num_sam,model_file, save=False):
    
    save_path = '/'.join(model_file.split('/')[:-1])+ '/'
    
    save_name = model_file.split('/')[-1]
    rank= int(save_name[1])
    
    size= int(save_name[0])
    
    import re
    epoch_str = re.findall(r'\d+', save_name.split('.')[-2])[-1]
    epoch = int(epoch_str)
    
    samples = get_samples(1000)

    if save:
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print 'create dir',save_path
        
        np.save(save_path+'samples_%d%d_e%d' % (size, rank, epoch), samples)
    
    return samples
    
    
if __name__ == '__main__':
    
    #0.initialize MPI

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank=comm.rank
    size=comm.size
    verbose=rank==0
    import os
    worker_id = os.getpid()
    
    # sum_worker_id = comm.allreduce(worker_id)
    #---
    
    #1. parsing sys arguments

    import sys
    import base.subnets.layers.someconfigs as someconfigs
    try:
    
        device=sys.argv[1]

    except:
        print 'USAGE: python eval.py [device]'
        print 'example: device=cuda0, indep_worker=0, mname=GRAN'
        raise
        

    
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
    utils.rng = np.random.RandomState(1234) # for init network and corrupt images
    rng = utils.rng
    #---
    
    
    
    #4. now load models, if needs to compile, recompile it
    base_path = '/scratch/mahe6562/gap/'
    #base_path = '/work/imj/gap/'
    
    #model_path = 'grans/lsun_reo/swp0.025-8-gpuarray-1397924_genclipped/'
    #model_path = 'gran-lsun-nccl/11-19-swp0.1-8-gpuarray-1526900/'
    
    model_gran = 'gran-lsun-nccl/11-21-swp0.1-8-gpuarray-139076/'
    model_dcga = 'dcgan-lsun-nccl/11-17-swp0.1-8-gpuarray-1255492/'
    model_comb = 'combined-lsun-nccl/11-29-swp0.1-4-gpuarray-41382/'

    def get_model_save_name(_size, _rank, epoch):
        model_save0 = '%d%ddcgan_num_hid100.batch100.eps_dis5e-05.eps_gen0.0001.num_z100.num_epoch36.lam1e-06.ts3.data.100_CONV_lsun%d.save' % (_size, _rank, epoch)
        model_save1 = '%d%ddcgan_num_hid100.batch100.eps_dis0.0001.eps_gen0.0002.num_z100.num_epoch10.lam1e-06.ts3.data.100_CONV_lsun%d.save' % (_size, _rank, epoch)
        model_save2 = '%d%ddcgan_num_hid100.batch100.eps_dis0.0001.eps_gen0.0002.num_z100.num_epoch20.lam1e-06.ts3.data.100_CONV_lsun%d.save' % (_size, _rank, epoch)
        model_save3 = '%d%ddcgan_num_hid100.batch100.eps_dis0.0001.eps_gen0.0002.num_z100.num_epoch15.lam1e-06.ts3.data.100_CONV_lsun%d.save' % (_size, _rank, epoch)
        return model_save0, model_save1, model_save2, model_save3
    
    model_paths=[
        [base_path+model_gran,  get_model_save_name(8, 0, 12)[3], 'GRAN'],
        [base_path+model_gran,  get_model_save_name(8, 1, 12)[3], 'GRAN'],
        [base_path+model_gran,  get_model_save_name(8, 2, 12)[3], 'GRAN'],
        [base_path+model_gran,  get_model_save_name(8, 3, 12)[3], 'GRAN'],
        [base_path+model_gran,  get_model_save_name(8, 4, 12)[3], 'GRAN'],
        [base_path+model_gran,  get_model_save_name(8, 5, 12)[3], 'GRAN'],
        [base_path+model_gran,  get_model_save_name(8, 6, 12)[3], 'GRAN'],
        [base_path+model_gran,  get_model_save_name(8, 7, 12)[3], 'GRAN'],
        [base_path+model_comb,  get_model_save_name(4, 2, 12)[3], 'GRAN'],
        [base_path+model_comb,  get_model_save_name(4, 3, 12)[3], 'GRAN']
    ]
    
    
    assert len(model_paths)==size
    model_path = model_paths[rank][0]
    model_file = model_path+model_paths[rank][1]
    mname      = model_paths[rank][2]
    
    recompile=False
    if recompile:
        pass
        model=reload_model(model_file, mname)
    else:
        
        model, get_samples=load_model(model_file,verbose=verbose)
        

        if False:
            
            from base.utils import save_the_numpy_params
            
            save_the_numpy_params(model,size,rank,epoch=14,model_path=model_path)
         
        
    #5. generate samples
    
    samples = gen_samples(get_samples,1000,model_file)
    
    #6. do gam2
    
    from base.gam2 import gam2
    
    gam2(model, samples, comm, verbose=verbose)
        
        
        
    
    
    