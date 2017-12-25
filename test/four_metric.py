import numpy as np
home='/home/g/gwtaylor/mahe6562/'
# home='/home/mahe6562/'
# home='/export/mlrg/hma02/'
save_path=home+'gan-zoo/dcgan-lsun/'


def plot(arr, label=None):
    
    color = ['-r','-b','-m','-g', '--r','--b','--m']
    
    import matplotlib.pyplot as plt

    fig = plt.figure(1, figsize=(5,8))
    fig.subplots_adjust(left = 0.15, bottom = 0.07,
                        right = 0.94, top = 0.94,
                        hspace = 0.14)

    ax = plt.subplot(111) # one record per 40 iterations , total 250 recordings for 10008 iterations in an epoch
    ax.plot(arr, color[0], label=label)
    ax.legend(loc='upper right')
    #ax.set_xlabel('epoch')
    ax.set_ylabel(label)
            
def mean_std(npy_file, argmin=None):
    
    npy_list=[]
    for f in folds:
        npy=np.load(save_path+f+npy_file)
        npy_list.append(npy)
        
    arr=np.array(npy_list)
    
    if argmin==None:
        
        mins=np.amin(arr,axis=1)
        
        # plot(mins)
        
        print np.mean(mins),'+-', np.std(mins)
        
        argmin= np.argmin(arr,axis=1)
        
        return argmin
        
    else:
    
        mins=[arr[ind,argm] for ind,argm in enumerate(argmin)]
        
        print np.mean(mins),'+-', np.std(mins)
            
        return None

    
def retrieve_name(var):
    import inspect
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]
    
    
def get_test_list():
    
    sample_list = [
        '12-21-9960-gan-1234', # toy 128_172 (finished 100e)
        # '12-21-11743-wgan-1234', # toy 128_172 (finished 100e)
        # '12-22-21043-lsgan-1234'  # toy 128_172 (finished 100e)
    ]
    
    ckernr_list = [
        '128_172',
        # '128_172',
        # '128_172' 
    ]
    
    return sample_list, ckernr_list
    
    
if __name__=='__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description="mnnd_gather")
        
    parser.add_argument("-d","--device", type=str, default='cuda0',
                            help="the theano context device to be used",
                            required=True)
                            
    args = parser.parse_args()
                            
                            
    import os
    device=args.device

    backend='cudandarray' if device.startswith('gpu') else 'gpuarray'
    os.environ['THEANO_BACKEND'] = backend

    if backend=='cudandarray':
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(device)
    else:
        if 'THEANO_FLAGS' in os.environ:
            raise ValueError('Use theanorc to set the theano config')
        os.environ['THEANO_FLAGS'] = 'device={0}'.format(device)
        import theano.gpuarray
        ctx=theano.gpuarray.type.get_context(None)

    
    import sys
    sys.path.append('..')
    
    import test_dcgan_four_metric
    
    
    # folds=globals()['folds_%s_%s' % (args.ltype, args.ckernr) ]
    #
    # assert len(folds) != 0
    #
    # names=retrieve_name(folds)
    #
    # session= [name for name in names if name!='folds'][0]
    # print session
         
    
    sample_list, ckernr_list = get_test_list()
    
    
    
    for fold_index, (fold, ckernr) in enumerate(zip(sample_list,ckernr_list)):
        
        f=open('result/test_output_%s.txt' % fold, 'w')
        
        print 'evaluating ', fold
        
        load_path =  fold
        
        load_epochs = [20,40,60,80,100]
        
        for load_epoch in load_epochs:
        
            ltype=fold.split('-')[-2]
        
            load_path_file = save_path+'/'+ load_path +'/'+'weight-'+str(load_epoch)+'-'+str(load_epoch*900)+'.save'
        
            try:
                assert os.path.isfile(load_path_file)
            except:
                print load_path_file
                raise
            rng_seed=1234
        
            mtype=None
    
            ls, iw, MMD = test_dcgan_four_metric.run(rng_seed, ltype, mtype, load_path_file, load_epoch, ckernr=ckernr, cri_ckern='128')
        
            print 'fold %s epoch %d: (%d/%d) LS: %.7f IW: %.7f MMD: %.7f:' % (fold, load_epoch, fold_index, len(sample_list), ls, iw, MMD)
        
            f.write('fold %s epoch %d: %f %f %f\n' % (fold, load_epoch, ls, iw, MMD))
        
        f.close()
        
            

                

            