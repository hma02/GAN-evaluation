import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import pylab

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

colors = ['k','m','y','c']
def plot(Y, labels,mname='GRAN'):
    
    labls = ['LSUN', '%s' % mname, 'GAP(%s)' % mname, 'GAP(%s)_comb' % mname]
    
    fig = plt.figure()
    
    x=Y[:,0]
    y=Y[:,1]
    
    labels=np.array(labels).astype(np.int32)

    for (index, cla) in enumerate(set(labels)):
        
        print cla
        
        indexer = np.array([j for (j,_) in enumerate(x) if labels[j]==cla])
        
        xc,yc = x[indexer],y[indexer]

        color=colors[cla]    

        plt.scatter(xc,yc,s=10,c=color,label=labls[cla],linewidth=0, alpha=0.3)
        
        # scatter = plt.scatter(_Y[0], _Y[1], s=30, color=colors[labels[index]], linewidth=0)
        
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    
    plt.legend(loc='lower left', scatterpoints=1, fontsize=8,ncol=10)
    alldata = np.vstack((Y.T,[labels])).T
    
    return alldata,fig
    
    # plt.show()
    

if __name__ == '__main__':
    
    Y= np.load('t-SNE_Y.npy')# random.random((1000,2))
    y= np.load('t-SNE_y.npy')#np.concatenate([np.ones((500,)),np.zeros((500,))])
    
    plot(Y, y)
