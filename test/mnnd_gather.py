import numpy as np
home='/home/g/gwtaylor/mahe6562/'
# home='/home/mahe6562/'
# home='/export/mlrg/hma02/'
save_path=home+'gan-zoo/dcgan-cifar10/ckern_test/'


#gan

folds_gan_128_128=[
'6-16-43711-gan-128-128-1234',
'6-16-43712-gan-128-128-1235',
'6-16-43713-gan-128-128-1236',
'6-16-43710-gan-128-128-1237',
'6-17-44942-gan-128-128-1238',
'6-17-44940-gan-128-128-1239',
'6-17-44941-gan-128-128-1240',
'6-17-44939-gan-128-128-1241',
# '6-17-45773-gan-128-128-1242',
# '6-17-45774-gan-128-128-1243',
# '6-17-45771-gan-128-128-1244',
# '6-17-45772-gan-128-128-1245',
]


folds_gan_128_16=[
'6-13-91743-gan-128-16-1236',
'6-13-91744-gan-128-16-1235',
'6-13-91745-gan-128-16-1234',
'6-13-91746-gan-128-16-1237',
'6-13-95366-gan-128-16-1238',
'6-13-95367-gan-128-16-1239',
'6-13-95368-gan-128-16-1241',
'6-13-95369-gan-128-16-1240',
    
]

folds_gan_16_128=[

'6-12-81293-gan-16-128-1235',
'6-12-81294-gan-16-128-1236',
'6-12-81295-gan-16-128-1237',
'6-12-81296-gan-16-128-1234',
'6-12-86311-gan-16-128-1240',
'6-12-86312-gan-16-128-1238',
'6-12-86313-gan-16-128-1239',
'6-12-86314-gan-16-128-1241',
    
]

folds_gan_16_16=[
    
'6-9-42391-gan-16-16-1234',
'6-9-42392-gan-16-16-1236',
'6-9-42393-gan-16-16-1235',
'6-9-42394-gan-16-16-1237',
'6-9-45971-gan-16-16-1239',
'6-9-45972-gan-16-16-1240',
'6-9-45973-gan-16-16-1238',
'6-9-45974-gan-16-16-1241',
    
]



##########################################

#lsgan

folds_lsgan_128_128=[
'5-26-80309-lsgan-128-128-1234',
'5-26-80312-lsgan-128-128-1235',
'5-26-80310-lsgan-128-128-1236',
'5-26-80311-lsgan-128-128-1237',
'5-27-85971-lsgan-128-128-1238',
'5-27-85974-lsgan-128-128-1239',
'5-27-85973-lsgan-128-128-1240',
'5-27-85972-lsgan-128-128-1241',
]

folds_lsgan_128_16=[
    
'6-1-73821-lsgan-128-16-1235',
'6-1-73822-lsgan-128-16-1234',
'6-1-79515-lsgan-128-16-1236',
'6-1-79516-lsgan-128-16-1237',
'6-1-79947-lsgan-128-16-1238',
'6-1-79948-lsgan-128-16-1239',
'6-2-80282-lsgan-128-16-1240',
'6-2-80283-lsgan-128-16-1241',
# '6-2-80610-lsgan-128-16-1242',
# '6-2-80611-lsgan-128-16-1243',
# '6-2-81826-lsgan-128-16-1244',
# '6-2-81827-lsgan-128-16-1245',

]


folds_lsgan_16_128=[
'5-30-69029-lsgan-16-128-1234',
'5-30-69030-lsgan-16-128-1235',
'5-30-69978-lsgan-16-128-1236',
'5-30-69979-lsgan-16-128-1237',
'5-31-70617-lsgan-16-128-1238',
'5-31-70618-lsgan-16-128-1239',
'5-31-71116-lsgan-16-128-1241',
'5-31-71117-lsgan-16-128-1240',
# '5-31-71827-lsgan-16-128-1242',
# '5-31-71828-lsgan-16-128-1243',
# '6-1-73209-lsgan-16-128-1244/',
# '6-1-73210-lsgan-16-128-1245/    ',
    
]


folds_lsgan_16_16=[

'5-22-92405-lsgan',
'5-22-92406-lsgan',
'5-22-92407-lsgan',
'5-22-92408-lsgan',
'5-22-92552-lsgan',
'5-22-92553-lsgan',
'5-22-92554-lsgan',
'5-22-92555-lsgan',
# '5-22-92817-lsgan',
# '5-22-92818-lsgan',
# '5-22-92819-lsgan',
# '5-22-92820-lsgan    ',

]

##############################

#wgan

folds_wgan_128_128=[
'6-8-55630-wgan-128-128-1234',
'6-8-55633-wgan-128-128-1235',
'6-8-55632-wgan-128-128-1236',
'6-8-55631-wgan-128-128-1237',
'6-8-56162-wgan-128-128-1238',
'6-8-56165-wgan-128-128-1239',
'6-8-56163-wgan-128-128-1240',
'6-8-56164-wgan-128-128-1241',
]


folds_wgan_128_16=[
    '6-9-61158-wgan-128-16-1235',
    '6-9-61159-wgan-128-16-1237',
    '6-9-61160-wgan-128-16-1234',
    '6-9-61161-wgan-128-16-1236',
    '6-9-61701-wgan-128-16-1238',
    '6-9-61702-wgan-128-16-1239',
    '6-9-61703-wgan-128-16-1240',
    '6-9-61704-wgan-128-16-1241',
]

folds_wgan_16_128=[
    '6-8-59799-wgan-16-128-1234',
    '6-8-59800-wgan-16-128-1237',
    '6-8-59801-wgan-16-128-1235',
    '6-8-59802-wgan-16-128-1236',
    '6-8-60516-wgan-16-128-1238',
    '6-8-60517-wgan-16-128-1240',
    '6-8-60518-wgan-16-128-1239',
    '6-8-60519-wgan-16-128-1241',
    ]
    
folds_wgan_16_16=[
'6-7-48586-wgan-16-16-1234',
'6-7-48587-wgan-16-16-1235',
'6-7-48588-wgan-16-16-1236',
'6-7-48589-wgan-16-16-1237',
'6-7-54295-wgan-16-16-1238',
'6-7-54296-wgan-16-16-1240',
'6-7-54297-wgan-16-16-1239',
'6-7-54298-wgan-16-16-1241',
    
]


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
    
if __name__=='__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description="mnnd_gather")
        
    parser.add_argument("-d","--device", type=str, default='cuda0',
                            help="the theano context device to be used",
                            required=True)
    parser.add_argument("-m","--mtype", type=str, default='iw',
                            help="the mtype of mnnd training",
                            required=True)
    
    parser.add_argument("-c","--ckernr", type=str, default='128_128',
                            help="which dis_gen folds of ckern experiment",
                            required=False)
                            
    parser.add_argument("-l","--ltype", type=str, default='wgan',
                            help="which gan type was the snapshot trained with",
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
    
    import test_dcgan
    
    
    folds=globals()['folds_%s_%s' % (args.ltype, args.ckernr) ]
    
    assert len(folds) != 0
    
    names=retrieve_name(folds)
    
    session= [name for name in names if name!='folds'][0]
    print session
    
    for mtype in [args.mtype]:
    
        vl_list=[]
        te_list=[]
    
        for fold_index, fold in enumerate(folds):
            
            print 'testing fold %s (%d/%d) on mtype %s' % (fold, fold_index, len(folds),mtype)
        
            # ltype=fold.split('-')[-1]
            rng_seed=1234

            load_path = save_path+args.ltype+'/'+args.ckernr + '/' + fold

            tmp_vl, tmp_te, tmp_vl_start = [], [], []
            
            load_epochs=range(0,210,10) + [1,2]
            
            for load_epoch in load_epochs:

                load_path_file = load_path + '/'+'wdcgan_param_cifar10_'+str(load_epoch)+'.save'
                
                mnnd_score_vl, mnnd_score_te, vl_start = test_dcgan.run(rng_seed, args.ltype, mtype, load_path_file, load_epoch, ckernr=args.ckernr)
                tmp_vl.append(mnnd_score_vl)
                tmp_te.append(mnnd_score_te)
                tmp_vl_start.append(vl_start)
                

            # find the epoch whose best score is nearest to its starting point score
            if mtype=='iw':
                idx = np.array(tmp_vl).argmin()
            elif mtype in ['ls','js']:
                idx = np.array(tmp_vl).argmax()
            else:
                raise NotImplementedError()
            # idx = (np.abs(np.array(tmp_vl)-np.array(tmp_vl_start))).argmin()
            # print np.abs(np.array(tmp_vl)-np.array(tmp_vl_start)).tolist()
            
            print 'best at epoch', load_epochs[idx]
            
            vl_list.append(tmp_vl[idx]) # vl score of the best epoch of this fold
            te_list.append(tmp_te[idx]) # te score of the best epoch of this fold
                
    
        print fold, mtype, 'VL:', np.mean(vl_list),'+-', np.std(vl_list)
        print fold, mtype, 'TE:', np.mean(te_list),'+-', np.std(te_list)