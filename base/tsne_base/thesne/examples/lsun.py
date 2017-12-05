import gzip
import pickle

from sklearn.utils import check_random_state

from model.tsne import tsne
import plot

import numpy as np


def subsample(X, y, size, random_state=None):
    random_state = check_random_state(random_state)

    shuffle = random_state.permutation(X.shape[0])

    X, y = X[shuffle], y[shuffle]
    X, y = X[0:size], y[0:size]

    return X, y


def data_load(i,test_filenames):
    
    """loads data and reshapes it to: #example x #pixels in img.
    arg:
        i: batch index
    return:
        test_data, a flattened #eg x #pxls in img.

    """
    import hickle as hkl
    test_data = hkl.load(test_filenames[i])
    test_data = test_data.astype('float32').transpose([3,0,1,2]);
    a,b,c,d = test_data.shape
    test_data = test_data.reshape(a,b*c*d)

    return (test_data)
    
def load_testset():

    datapath = '/work/djkim117/lsun/church/preprocessed_toy_100/'
    import glob
    test_filenames = sorted(glob.glob(datapath + 'test_hkl_b100_b_100/*' + '.hkl'))
    test_set = data_load(0,test_filenames)
    for i in xrange(1,10):
        test_set = np.vstack((test_set,data_load(i,test_filenames)))

    Nt, D = test_set.shape

    return test_set
    
    
def tsne_lsun(X,y,mname,perplexity=100, n_epochs=500, sigma_iters=50, verb=1):

    seed=0
    # Available at http://deeplearning.net/tutorial/gettingstarted.html
    
    X, y = subsample(X, y, size=2000, random_state=seed)

    Y = tsne(X, perplexity=perplexity, n_epochs=n_epochs, sigma_iters=sigma_iters,
             random_state=seed, verbose=verb)

    alldata, fig = plot.plot(Y, y, mname=mname)
    
    return alldata,fig


if __name__ == "__main__":
    import sys
    
    model=int(sys.argv[1])
    idx=int(sys.argv[2])
    
    if model==0:
        mname='GRAN'
    elif model==1:
        mname='DCGAN'
    else:
        raise ValueError('incorrect model')
        
        
        
    X_data = load_testset()/255.
    y_data = np.ones(shape=(len(X_data),))*0
    
    
    if mname=='GRAN':
        
        X_gran = np.load('/scratch/mahe6562/samples/gran4_100swap_200ckern_nolrdecay/GRAN-e10/samples_gran0.npy')
        
        print X_gran.shape, X_gran.min(), X_gran.max()
        
        if X_gran.max()>1:
        
            X_gran=X_gran.reshape((1000,3*64*64))/255.
            
        if X_gran.min()<0:
            
            raise ValueError('wrong input range')
        
        
        
        X_gran_gap =np.load('/scratch/mahe6562/samples/gran4_100swap_200ckern_nolrdecay/GRAN-e10/samples_gran%d.npy' % idx).reshape((1000,3*64*64))/255.
        
        print X_gran_gap.shape, X_gran_gap.min(), X_gran_gap.max()
        
        if X_gran_gap.max()>1:
            
            X_gran_gap=X_gran_gap.reshape((1000,3*64*64))/255.
            
        if X_gran_gap.min()<0:
            
            raise ValueError('wrong input range')
        
    else:
        X_gran = np.load('/scratch/mahe6562/samples/dcgan-10-31/samples_dcgan0.npy').reshape((1000,3*64*64))/255.
        X_gran_gap =np.load('/scratch/mahe6562/samples/dcgan-10-31/samples_dcgan%d.npy' % idx).reshape((1000,3*64*64))/255.
    
    y_gran = np.ones(shape=(len(X_gran),))*1
    y_gran_gap =np.ones(shape=(len(X_gran_gap),))*2
    
    X = np.concatenate((X_data,X_gran,X_gran_gap),axis=0)

    y = np.concatenate((y_data,y_gran,y_gran_gap),axis=0)
    
    
    
    
    alldata,fig = tsne_lsun(X,y,mname,idx,perplexity=100, n_epochs=400, sigma_iters=50)
    
    
    np.save('fig/all%s%d.npy' % (mname,idx),alldata)
    fig.savefig('fig/t-SNE%s%d.pdf' % (mname,idx),format='pdf')
