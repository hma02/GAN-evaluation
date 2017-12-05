import numpy as np

def correct_shape_inrange(X):
    
    print X.shape, X.min(), X.max()
    
    if X.shape!= (1000,3*64*64):
    
        X=X.reshape((1000,3*64*64))
        
        # print X.shape
        
        if X.shape!= (1000,3*64*64):
            
            raise ValueError('wrong shape')
    
    if X.max()>1:
    
        X=X/255.
        
    if X.min()<0:
        
        raise ValueError('wrong input range')
        
    return X
    
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
       
def load_samples(path, _rank, _size, epoch):
    
    samples_name = 'samples_%d%d_e%d.npy' % (_size,_rank,epoch)
    samples=np.load(path+samples_name)
    
    return samples
    
def load_single(path_single, _size_single, n_single, rank_start_single, epoch):

    path=path_single

    _size=_size_single

    samples_single=None

    for index, _rank in enumerate(range(rank_start_single, rank_start_single+n_single, 1)):
        s = load_samples(path, _rank, _size, epoch)
        s=correct_shape_inrange(s)
        
        if index==0:
            samples_single=s
        else:
            samples_single = np.concatenate((samples_single,s),axis=0)

    return samples_single
    
    
def tsne(mname, data, samples_single, samples_gap, samples_gap_comb, verbose):
    
    samples_single=samples_single[::4]
    samples_gap=samples_gap[::4]
    samples_gap_comb=samples_gap_comb[::2]
    
    if not len(data)==len(samples_single)==len(samples_gap)==len(samples_gap_comb):
        print len(data)
        print len(samples_single)
        print len(samples_gap)
        print len(samples_gap_comb)
        
        raise ValueError('should be same len')
    
    y_data = np.ones(shape=(len(data),))*0
    y_single = np.ones(shape=(len(samples_single),))*1
    y_gap =np.ones(shape=(len(samples_gap),))*2
    y_gap_comb =np.ones(shape=(len(samples_gap_comb),))*3
    
    X = np.concatenate((data, samples_single, samples_gap, samples_gap_comb),axis=0)

    y = np.concatenate((y_data, y_single, y_gap, y_gap_comb),axis=0)
    
    
    
    X = X[::2]
    y = y[::2]
        
    from tsne_base.thesne.examples.lsun import tsne_lsun
    
    try:
       alldata,fig = tsne_lsun(X,y,mname,perplexity=70, n_epochs=500, sigma_iters=50, verb=verbose)

    except Exception as e:
   
       print e
   
       raise
       
    return alldata, fig


import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

hsv_colors = [(0.56823266219239377, 0.82777777777777772, 0.70588235294117652),
              (0.078146611341632088, 0.94509803921568625, 1.0),
              (0.33333333333333331, 0.72499999999999998, 0.62745098039215685),
              (0.99904761904761907, 0.81775700934579443, 0.83921568627450982),
              (0.75387596899224807, 0.45502645502645506, 0.74117647058823533),
              (0.028205128205128216, 0.4642857142857143, 0.5490196078431373),
              (0.8842592592592593, 0.47577092511013214, 0.8901960784313725),
              (0.0, 0.0, 0.49803921568627452),
              (0.16774193548387095, 0.82010582010582012, 0.74117647058823533),
              (0.51539855072463769, 0.88888888888888884, 0.81176470588235294)]

rgb_colors = matplotlib.colors.hsv_to_rgb(np.array(hsv_colors).reshape(10, 1, 3))
#colors = matplotlib.colors.ListedColormap(rgb_colors.reshape(10, 3))

colors = ['k','m','m','m']


def plot_s(Y, labels,mname='GRAN'):
    
    labls = ['LSUN', '%s' % mname, 'GAP(%s)' % mname, 'GAP(%s)_comb' % mname]
    
    x=Y[:,0]
    y=Y[:,1]
    
    labels=np.array(labels).astype(np.int32)
    
    # sorting different class sets
    
    DATA=None
    SINGLE=None
    GAP=None
    GAP_COMB=None
    
    all_xy= [DATA, SINGLE, GAP, GAP_COMB]

    for (index, cla) in enumerate(set(labels)):
        
        print cla
        
        indexer = np.array([j for (j,_) in enumerate(x) if labels[j]==cla])
        
        xc,yc = x[indexer],y[indexer]
        
        all_xy[cla]=[xc,yc]
        
        
    fig = plt.figure(figsize=(5,15))
    fig.subplots_adjust(left = 0.1, bottom = 0.05,right = 0.9, top = 0.95,hspace = 0.14)
    
    for cla in range(1, len(all_xy),1):
        
        
        ax = plt.subplot(100*(len(all_xy)-1)+10+cla)   
        
        xc,yc=all_xy[0] 
    
        ax.scatter(xc,yc,s=10,c=colors[0],label=labls[0],linewidth=0, alpha=0.3)
    
        xc,yc=all_xy[cla] 
        
        print len(xc)

        ax.scatter(xc,yc,s=10,c=colors[cla],label=labls[cla],linewidth=0, alpha=0.3)
        
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
    
        ax.legend(loc='lower left', scatterpoints=1, fontsize=8,ncol=10)
    
    
    plt.suptitle('t-SNE of LSUN data and %s' % mname)
        
    
    plt.show()
        
    
    return fig
    
    # plt.show()
    
    

def plot_separate(alldata, mname):
    
    print alldata.shape
    
    Y=(alldata.T)[0:2].T

    print Y.shape

    y=(alldata.T)[2]

    print y.shape
    
    fig = plot_s(Y, y, mname=mname)
    
    fig.savefig('tsne.pdf',format='pdf')
    
    return fig
    
    
    
    
def main():
    
    figsaved=True
    
    
    if not figsaved:
    
        #1. parsing sys arguments

        import sys
        import subnets.layers.someconfigs as someconfigs
        try:
    
            device=sys.argv[1]

        except:
            print 'USAGE: python tsne.py [device]'
            print 'example: device=cuda0'
            raise
    
    
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
    
    
        # 3. create save_path and info_matrix for loading samples
        import time
        date = '%d-%d' % (time.gmtime()[1], time.gmtime()[2])
        
        import os
        pid=os.getpid()
    
        save_path = './fig-%s-%d/' % (date,pid)
    
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print 'create dir',save_path
        
        data = load_testset()/255.
    
        base_path = '/scratch/mahe6562/gap/'
        #base_path = '/work/imj/gap/'
    
        #model_path = 'grans/lsun_reo/swp0.025-8-gpuarray-1397924_genclipped/'
        #model_path = 'gran-lsun-nccl/11-19-swp0.1-8-gpuarray-1526900/'
        model_path = 'dcgan-lsun-nccl/11-17-swp0.1-8-gpuarray-1255492/'
        model_path1 = 'combined-lsun-nccl/11-19-swp0.1-4-gpuarray-658562/'
    
        info_matrix=[
            [base_path+model_path,  8, 4, 0, 34],
            [base_path+model_path,  8, 4, 4, 34],
            [base_path+model_path1, 4, 2, 0, 34]
        ]
        mname='DCGAN'
    
        # 4. load samples based on info_matrix
    
        samples_single  = load_single(*info_matrix[0])
        samples_gap     = load_single(*info_matrix[1])
        samples_gap_comb= load_single(*info_matrix[2])
    
        # 5. tsne based on loaded samples
    
        alldata,fig = tsne(mname, data, samples_single, samples_gap, samples_gap_comb, verbose=True)
    
    
        np.save(save_path+'/all%s.npy' % mname,alldata)
        fig.savefig(save_path+'/t-SNE%s.pdf' % mname,format='pdf')
        
        alldata=np.load(save_path+'/all%s.npy' % mname)
        plot_separate(alldata, mname='DCGAN')
        
    else:
        
        import time
        date = '%d-%d' % (time.gmtime()[1], time.gmtime()[2])
        
        save_path = './fig-11-26-51116/'  #'./fig-%s/' % date
        mname='DCGAN'
    
        alldata=np.load(save_path+'/all%s.npy' % mname)
        plot_separate(alldata, mname='DCGAN')
    
    
if __name__=='__main__':
    
    main()
    
    
    
    