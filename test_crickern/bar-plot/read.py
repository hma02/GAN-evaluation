import matplotlib.pyplot as plt
# plt.style.use('bmh')
from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.rcParams.update({'font.size': 9})

fontP = FontProperties()
fontP.set_size('small')

# colors ={
# 'blue'      :'b',
# 'green'     :'g',
# 'red'       :'r',
# 'cyan'      :'c',
# 'magenta'   :'m',
# 'yellow'    :'y',
# 'black'     :'k',
# 'white'     :'w'
# }

colors=['r','c','b','y', 'm']

linestyles = ['-', '-', '-', '-', '-', '-','--']  # '_', '-', '--', ':']
markers = ['', '', '','', '', '.', '']

import numpy as np



def read_file(filename):
    
    import subprocess
    n=subprocess.Popen("wc -l < %s" % filename, shell=True, stdout=subprocess.PIPE).stdout.read()
    n=int(n)
    print filename, n
    
    order=[2,0,1]
    
    labels= ['1:1', '1:5', '5:1']
    labels = [labels[i] for i in order]
    
    res=np.loadtxt(filename,usecols=[0])
    
    res_tmp=[]
    for i in order:
        try:
            res_tmp.append(res[i])
        except IndexError:
            res_tmp.append(np.nan)
    res=res_tmp
        
    try:
        res_std = np.loadtxt(filename,usecols=[1])
        res_std = [res_std[i] for i in order]
    except:
        # raise ValueError('needs std to plot bars with std.')
        res_std=np.zeros(len(res))
    
    # split_res =  np.split(res, res.shape[0]/n, axis=0)

    return labels, res, res_std


def plot_mtype_bar(mtype, filenames, plt, print_legend=True, prop=fontP, xlabel='', ltype_together=True):
    
    tmp=[]
    
    data = []
    
    tick_positions = []
    
    for k, f in enumerate(filenames):
    
        labels, res, res_std = read_file(f)
        data.append([labels, res, res_std])
        N=len(res)
        
        width=0.88/N
    
        ind=np.arange(N)
        
        if ltype_together:
            p = plt.bar(k+ind*width, res, width, alpha=1, color=colors[k],edgecolor = 'k', linewidth=0.5, yerr=res_std)
            tick_positions= tick_positions+list(k+ind*width+0.5*width)
            tick_labels = labels*(k+1)
            plt.xticks(tick_positions, tick_labels)
            plt.xlim((-1*width,len(filenames)-1+(len(labels)+1)*width))
            
        else:
            p = plt.bar(ind+k*width, res, width, alpha=1, color=colors[k],edgecolor = 'k', linewidth=0.5, yerr=res_std)
            if k==0:  plt.xticks(ind+0.5 * 0.8, labels)
            plt.xlim((-1*width,len(labels)))
            
        tmp.append(p)
    
    legends=[ f.split('.out')[0].split('_')[0].split('/')[-1] for f in filenames]

    def change_legend(l):
        if l=='gan':
            return 'DCGAN'
        elif l=='wgan':
            return 'W-DCGAN'
        elif l=='lsgan':
            return 'LS-DCGAN'
        else:
            print l, 'weird'
            return l

    legends=[change_legend(l) for l in legends]
    # print len(legends), len(tmp)
    assert len(legends)==len(tmp)

    

    if mtype=='iw':
        plt.ylabel('%s distance' % mtype.upper())
        plt.ylim((0,1050))
    elif mtype=='js':
        plt.ylabel('%s divergence' % mtype.upper())
        # plt.ylim((0,0.000001))
    elif mtype=='ls':
        plt.ylabel('%s loss' % mtype.upper())

    plt.xlabel(xlabel)
    
    # plt.xlabel('Epoch')
    # xticks = [0.5, 0.5+width*1.5, 1.5+width*1.5, 3.5+width*1.5, 4+width*1.5]

    # ax.xaxis.set_ticks_position('none')
    # ax.xaxis.set_tick_params(pad=10)
    if print_legend==True:
        plt.legend(tmp, legends , loc=3,mode='expand', bbox_to_anchor=(0., 1.05, 1., 0.105 ), ncol=3, prop = fontP)
        
    
    # check for nan
    
    for k, f in enumerate(filenames): 
        
        labels, res, res_std = read_file(f)
    
        for i, r in enumerate(res):
            if np.isnan(r):
                ymin, ymax = ax.get_ylim()
                if mtype=='iw': 
                    nan_height=ymax*0.9
                else:
                    nan_height=ymax*0.05
                    
                
                print 'nan_height', nan_height
                p, = plt.bar(k+i*width, nan_height, width, alpha=0.1, color=colors[k],edgecolor = 'k', linewidth=0.5,hatch="//")
                ax.text(k+i*width, p.get_height(), 'NaN', fontproperties=fontP)
                

    return data, legends

def print_mtype_table(mtype_data, ltype_legends, mtype):
    
    # mtype_data shape =  '(len(ltypes),  len(label,mean,std), len(nclasses)) =  (3, 3, 4)' 
    # legends corresponds to ltypes
    
    # tablename=mtype
    # ltype 1 2 5 10
    # gan   
    # lsgan
    # wgan
    
    mtype_data=np.array(mtype_data)
    
    Nclass_labels=mtype_data[0][0]
    
    mean_data = mtype_data[:,1,:]
    std_data = mtype_data[:,2,:]
        
    data = [[ mean+'+-'+std for mean,std in zip(means,stds) ] for means,stds in zip(mean_data,std_data)]
    
    data = np.array(data)
    
    assert data.shape==(len(ltype_legends), len(Nclass_labels))  # 4x3
    
    import pandas as pd
    
    df = pd.DataFrame(data,index=ltype_legends, columns=Nclass_labels)
    df.name=mtype
    pd.set_option('display.width', 100)
    
    print df
    # print df.info()
    
if __name__ == '__main__':
    

    import argparse

    parser = argparse.ArgumentParser(description="plot")
    

    parser.add_argument("-m","--mtype", type=str, default='all',
                            help="the mtype of mnnd training",
                            required=False)
                            
    parser.add_argument("-s","--separate", type=bool, default=True,
                            help="if plot separately",
                            required=False)
                            
    parser.add_argument("-x","--xlabel", type=str, default='# of Dis. Update : # of Gen. Update',
                            help="the x-axis label",
                            required=False)
                          
    # parser.add_argument("-l","--ltype", type=str, default='wgan',
    #                         help="which gan type was the snapshot trained with",
    #                         required=True)
                          
    args = parser.parse_args()

    mtype=args.mtype
    
    xlabel=args.xlabel
    
    separate=args.separate
    
    if mtype !='all':
        
        import glob

        locations={'iw':'upper right', 'ls': 'lower right'}

        files=glob.glob('tmp/*_'+mtype+'.out.txt')

        filenames=sorted(files)
    
        print filenames

        assert len(filenames)!=0
    

        fig = plt.figure(1, figsize=(6.5,4))
        fig.subplots_adjust(left = 0.12, bottom = 0.17,
                            right = 0.96, top = 0.75,
                            hspace = 0.14)

        plot_mtype_bar(mtype, filenames, plt, print_legend=True, prop=fontP) 
                    
        plt.show()

        fig.savefig('CIFAR10-JK-bar-%s.pdf' % mtype,format='pdf')
        
            
    else:
        
        mtypes=['ls', 'js', 'iw']

        if separate==False:
            fig = plt.figure(0, figsize=(4,5.5))
            fig.subplots_adjust(left = 0.24, bottom = 0.06,
                                right = 0.97, top = 0.90,
                                hspace = 0.25)
        
        table_data = []
        
        for mtype_index, mtype in enumerate(mtypes): 

            import glob

            locations={'iw':'upper right', 'ls': 'lower right'}

            files=glob.glob('tmp/*_'+mtype+'.out.txt')

            filenames=sorted(files)

            print filenames

            assert len(files)!=0
            
            if separate==True:
                
                fig = plt.figure(mtype_index+1, figsize=(4.5,2.5))
                fig.subplots_adjust(left = 0.20, bottom = 0.17,
                                    right = 0.97, top = 0.84,
                                    hspace = 0.25)
                ax = plt.subplot(111)
                
                plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='on') # labels along the bottom edge are off
            
            else:

                ax = plt.subplot(len(mtypes)*100+10+(mtype_index+1))
            
            import matplotlib.ticker as mtick
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.0e'))
            
            if separate==True:
                mtype_data, legends = plot_mtype_bar(mtype, filenames, plt, print_legend=True,xlabel=xlabel)
            else:
                mtype_data, legends = plot_mtype_bar(mtype, filenames, plt, print_legend=(mtype_index == 0),xlabel=xlabel)
                
            
            print_mtype_table(mtype_data, legends, mtype)
            table_data.append(mtype_data)
            
            if separate==True: fig.savefig('separate/CIFAR10-JK-bar-%s.pdf' % mtype,format='pdf')
                
        
        # print '(mtype,  ltype,  len(label,mean,std), 4nclass) = ',
        # print np.array(table_data).shape

        plt.show()
        if separate==False: fig.savefig('CIFAR10-JK-bar-all.pdf',format='pdf')



