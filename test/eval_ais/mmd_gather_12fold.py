import numpy as np
home='/export/mlrg/hma02/'
# save_path=home+'gan-zoo/dcgan-mnist/'
save_path=home+'gan-zoo/dcgan-cifar10/'

       
     
   
folds_gan=[
'4-27-89026-gan',      
'4-27-89027-gan',     
'4-27-89028-gan',      
'4-27-89029-gan',      
'4-28-91029-gan',
'4-28-91030-gan',
'4-28-91031-gan',
'4-28-91032-gan',
]

folds_lsgan=[
'4-29-94463-lsgan',  
'4-29-94464-lsgan',  
'4-29-94465-lsgan',  
'4-29-94466-lsgan',  
'4-29-96961-lsgan',
'4-29-96962-lsgan',
'4-29-96963-lsgan',
'4-29-96964-lsgan',
]

folds_wgan=[
'4-30-100819-wgan',
'4-30-100820-wgan',
'4-30-100821-wgan',
'4-30-100822-wgan',
'4-30-98775-wgan' ,
'4-30-98776-wgan' ,
'4-30-98777-wgan' ,
'4-30-98778-wgan' ,
]


folds=folds_gan

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
    
    # print arr
    if npy_file.split('.')[0][1:].startswith('mmd'):
        
        if argmin==None:
        
            mins=np.amin(arr,axis=1)
        
            # plot(mins)
        
            print '%s: %.4f+-%.4f' % (npy_file.split('.')[0][1:], np.mean(mins), np.std(mins))
        
            argmin= np.argmin(arr,axis=1)
        
            return argmin
        
        else:
    
            mins=[arr[ind,argm] for ind,argm in enumerate(argmin)]
        
            print '%s: %.4f+-%.4f' % (npy_file.split('.')[0][1:], np.mean(mins), np.std(mins))
            
            return None
        
    else:

        
        maxs=np.amax(arr,axis=1)

        # plot(mins)

        print '%s: %.4f+-%.4f' % (npy_file.split('.')[0][1:], np.mean(maxs), np.std(maxs))


    
    
if __name__=='__main__':
    
    files=['/mmd_vl.npy',
           '/mmd_te.npy',
           '/is_vl.npy',
           '/is_sam.npy']
    argmin=None
    for ind, f in enumerate(files):       
        argmin=mean_std(f,argmin)
    
    