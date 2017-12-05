import theano
import theano.tensor as T
import numpy as np
import time

def dis_error(model, images, target, num_train): 
    
    # images should be in RGB (0,255)
        
    err=model.dis_network.errors(images, target, num_train=num_train).eval()
    
    return err

def test_error(model, images, Nt):
    
    # images should be in RGB (0,255)
    
    target1 = T.alloc(1, Nt)
    
    image_shape0=[test_set[0].shape[0], 3, 64, 64]
    images = test_set[0].reshape(image_shape0)

    err=dis_error(model, images, target1, num_train=Nt)
    
    return err
    
def sample_error(model,images,num_sam):
    
    target0 = T.alloc(0, num_sam)

    err=dis_error(model, images, target0, num_train=num_sam)
    
    return err
    
def samples_error(model,samples_list,num_sam, inrange='(0,1)'):
    
    errs=[]
    
    for samples in samples_list:
        
        if samples.shape!= (num_sam,3,64,64): #(b,c,0,1)
            samples=samples.reshape((num_sam,3,64,64))
            
        if inrange=='(-1,1)':
        
            assert samples.min()<=0 and samples.min()>=-1
            assert samples.max()>=0 and samples.max()<=1
        
        elif inrange=='(0,1)':
        
            assert samples.min()>=0 and samples.min()<=1
            assert samples.max()>=0 and samples.max()<=1
        
        if hasattr(samples, 'type')==False:
            # print 'not has <type> attr'
            samples = theano.shared(np.asarray(samples, dtype=theano.config.floatX))
        
        e = sample_error(model,samples,num_sam)
        
        errs.append(e)
        
    return errs
    
def print_ig(resultM, size):
    
    if size==10:
        #ignore self
        from copy import deepcopy
    
        resultM_ig = deepcopy(resultM)
    
        for row in [0,1,2,3]:
            for col in [0,1,2,3]:
                resultM_ig[row][col]=0
            
        for row in [4,5,6,7]:
            for col in [4,5,6,7]:
                resultM_ig[row][col]=0
        
        try:
            for row in [8,9]:
                for col in [8,9]:
                    resultM_ig[row][col]=0   
        except:
            pass
        
        print 'mean ignore self'
        rmean = resultM_ig.mean(axis=0)
        for col in [0,1,2,3,4,5,6,7]:
            rmean[col] = rmean[col]*size/(size-4)
        for col in [8,9]:
            rmean[col] = rmean[col]*size/(size-2)
        
        print '', np.array2string(rmean, formatter={'float': lambda x: '{: 0.3f},'.format(x)}, max_line_width=200)
    
    
    
        for row in [0,1,2,3]:
            for col in [0,1,2,3]:
                resultM_ig[row][col]=np.inf
            
        for row in [4,5,6,7]:
            for col in [4,5,6,7]:
                resultM_ig[row][col]=np.inf
            
        try:
            for row in [8,9]:
                for col in [8,9]:
                    resultM_ig[row][col]=np.inf
        except:
            pass
        
        print 'min ignore self'
        print '', np.array2string(resultM_ig.min(axis=0), formatter={'float': lambda x: '{: 0.3f},'.format(x)}, max_line_width=200)
        print
    
    
    else:
        
        pass
    
    
    
        
def gam2(model, samples, comm, verbose=False):
    
    rank=comm.rank
    size=comm.size
    
    comm.Barrier()
    
    # the shape of samples are (100, 12288)
    
    shape0, shape1 = samples.shape[0], samples.shape[1]
    
    samples_gather = np.empty(shape=(size*shape0,shape1))
    
    # sin0, sin1 sin2, sin3, gap0,gap1,gap2,gap3, each has 100 samples totally 800 samples, we can resuse those samples in gam2
    
    if verbose: t0=time.time()
    
    samples_gather = comm.allgather(samples)
    
    if verbose: print 'allgather time:%.3fs' % (time.time()-t0)
        
    # assert np.array_equal(samples,samples_gather[rank])==True
    
    result = samples_error(model,samples_gather,num_sam=shape0) # result is a list of each model classifying on 8 x 100 samples
    
    comm.Barrier()
    
    result_array = comm.allgather(result)
    
    resultM = np.array(result_array).reshape((size,len(result)))
    
    if rank==0: 
        resultM_string=np.array2string(resultM, formatter={'float': lambda x: '{: 0.3f},'.format(x)}, max_line_width=200)
        print
        print 'gam2 matrix:'
        print resultM_string
        print
        
    if rank==0:
        
        rmean = resultM.mean(axis=0)
        
        print 'mean'
        print '', np.array2string(rmean, formatter={'float': lambda x: '{: 0.3f},'.format(x)}, max_line_width=200)
        
        rmin = resultM.min(axis=0)
        print 'min'
        print '', np.array2string(rmin, formatter={'float': lambda x: '{: 0.3f},'.format(x)}, max_line_width=200)
        
        assert size==len(result)
        print_ig(resultM, size)
            
            
        
    comm.Barrier()
    
    samples_gather[:]=[]
    
    return resultM
    
    