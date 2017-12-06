import os, sys,gzip
import numpy as np
from maximum_mean_discripency import mix_rbf_mmd2
from utils import * 

mnistF=0
cifarF=0
advF=0
bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]


DIR='/home/daniel/Documents/research/gan_zoo'

##############################
############ MNIST ###########
##############################


if advF:

    datapath='/home/daniel/Documents/data/mnist/mnist.pkl.gz'
    f = gzip.open(datapath, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)


    ### LS-GAN ###
    tmp_sam, tmp_adv = [], []
    for i in xrange(1,2):

        lsdcgan_mnist_adv_sam = np.load(DIR+'/models/lsdcgan_mnist_adversary_samples_10000.npy').reshape([-1, 784])
        lsdcgan_mnist_sam     = np.load(DIR+'/models/lsdcgan_mnist_samples_10000.npy').reshape([-1, 784])
        lsdcgan_mnist_sam_score = mix_rbf_mmd2(test_set[0], lsdcgan_mnist_sam, sigmas=bandwidths)
        lsdcgan_mnist_adv_score = mix_rbf_mmd2(test_set[0], lsdcgan_mnist_adv_sam, sigmas=bandwidths)
        tmp_sam.append(lsdcgan_mnist_sam_score)
        tmp_adv.append(lsdcgan_mnist_adv_score)
        print 'Least Sqaure DCGAN MNIST MMD SCORE TR %f TE %f, diff %f' % (lsdcgan_mnist_adv_score, lsdcgan_mnist_sam_score, lsdcgan_mnist_adv_score - lsdcgan_mnist_sam_score)

    #tmp_tr = np.asarray(tmp_tr)
    #tmp_vl = np.asarray(tmp_vl)
    #print 'LSDCGAN MNIST MMD SCORE TR %f +- %f TE %f +- %f' % (np.mean(tmp_tr), np.std(tmp_tr), np.mean(tmp_vl), np.std(tmp_vl))
    import pdb; pdb.set_trace()




if mnistF:

    datapath='/home/daniel/Documents/data/mnist/mnist.pkl.gz'
    f = gzip.open(datapath, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)


    ### LS-GAN ###
    tmp_tr, tmp_vl = [], []
    for i in xrange(1,11):

        lsdcgan_mnist_sam   = np.load(DIR+'/models/lsgan_mnist_samples_40000_rng'+str(i)+'.npy').reshape([-1, 784])
        lsdcgan_mnist_tr_score = mix_rbf_mmd2(train_set[0], lsdcgan_mnist_sam[:10000], sigmas=bandwidths)
        lsdcgan_mnist_te_score = mix_rbf_mmd2(test_set[0], lsdcgan_mnist_sam[:10000], sigmas=bandwidths)
        tmp_tr.append(lsdcgan_mnist_tr_score)
        tmp_vl.append(lsdcgan_mnist_te_score)
        print 'Least Sqaure DCGAN MNIST MMD SCORE TR %f TE %f' % (lsdcgan_mnist_tr_score, lsdcgan_mnist_te_score)

    tmp_tr = np.asarray(tmp_tr)
    tmp_vl = np.asarray(tmp_vl)
    print 'LSDCGAN MNIST MMD SCORE TR %f +- %f TE %f +- %f' % (np.mean(tmp_tr), np.std(tmp_tr), np.mean(tmp_vl), np.std(tmp_vl))
    import pdb; pdb.set_trace()


    ### DCGAN  ###
    tmp_tr, tmp_vl = [], []
    for i in xrange(1,11):
        dcgan_mnist_sam        = np.load(DIR+'/models/dcgan_mnist_sample_40000_'+str(i)+'.npy').reshape([-1, 784])
        dcgan_mnist_tr_score   = mix_rbf_mmd2(train_set[0], dcgan_mnist_sam[:10000], sigmas=bandwidths)
        dcgan_mnist_te_score   = mix_rbf_mmd2(test_set[0], dcgan_mnist_sam[:10000], sigmas=bandwidths)
        tmp_tr.append(dcgan_mnist_tr_score)
        tmp_vl.append(dcgan_mnist_te_score)
        print 'DCGAN MNIST MMD SCORE TR %f TE %f' % (dcgan_mnist_tr_score, dcgan_mnist_te_score)

    tmp_tr = np.asarray(tmp_tr)
    tmp_vl = np.asarray(tmp_vl)
    print 'DCGAN MNIST MMD SCORE TR %f +- %f TE %f +- %f' % (np.mean(tmp_tr), np.std(tmp_tr), np.mean(tmp_vl), np.std(tmp_vl))
    import pdb; pdb.set_trace()



    ### DVAE ###
    tmp_tr, tmp_vl = [], []
    for i in xrange(1,11):
        dvae_mnist_sam     = np.load(DIR+'/models/dvae_mnist_samples_10000_rng'+str(i)+'.npy')
        dvae_mnist_tr_score   = mix_rbf_mmd2(train_set[0], dvae_mnist_sam[:10000], sigmas=bandwidths)
        dvae_mnist_te_score   = mix_rbf_mmd2(test_set[0],  dvae_mnist_sam[:10000], sigmas=bandwidths)
        tmp_tr.append(dvae_mnist_tr_score)
        tmp_vl.append(dvae_mnist_te_score)
        print 'DVAE MNIST MMD SCORE TR %f TE %f' % (dvae_mnist_tr_score, dvae_mnist_te_score)

    tmp_tr = np.asarray(tmp_tr)
    tmp_vl = np.asarray(tmp_vl)
    print 'DVAE MNIST MMD SCORE TR %f +- %f TE %f +- %f' % (np.mean(tmp_tr), np.std(tmp_tr), np.mean(tmp_vl), np.std(tmp_vl))
    import pdb; pdb.set_trace()


    ### GRAN ###
    gran_mnist_sam     = np.load(DIR+'/models/gran_mnist_samples_400000.npy')
    gran_mnist_tr_score   = mix_rbf_mmd2(train_set[0], gran_mnist_sam[:10000], sigmas=bandwidths)
    gran_mnist_te_score   = mix_rbf_mmd2(test_set[0],  gran_mnist_sam[:10000], sigmas=bandwidths)
    print 'GRAN MNIST MMD SCORE TR %f TE %f' % (gran_mnist_tr_score, gran_mnist_te_score)



    ### VAE ###
    vae_mnist_sam     = np.load(DIR+'/models/vae_mnist_samples_10000.npy')
    vae_mnist_tr_score   = mix_rbf_mmd2(train_set[0], vae_mnist_sam[:10000], sigmas=bandwidths)
    vae_mnist_te_score   = mix_rbf_mmd2(test_set[0],  vae_mnist_sam[:10000], sigmas=bandwidths)
    print 'VAE MNIST MMD SCORE TR %f TE %f' % (vae_mnist_tr_score, vae_mnist_te_score)

    ### W1-DCGAN ###
    wdcgan_mnist_sam    = np.load(DIR+'/models/wdcgan_mnist_samples_40000.npy').reshape([-1, 784])
    wdcgan_mnist_tr_score   = mix_rbf_mmd2(train_set[0], wdcgan_mnist_sam[:10000], sigmas=bandwidths)
    wdcgan_mnist_te_score   = mix_rbf_mmd2(test_set[0], wdcgan_mnist_sam[:10000], sigmas=bandwidths)
    print 'Wasserstain-1 DCGAN MNIST MMD SCORE TR %f TE %f' % (wdcgan_mnist_tr_score, wdcgan_mnist_te_score)


##############################
########### CIFAR10 ##########
#############################
#

if cifarF:
    ### DCGAN  ###
    mix_rbf_mmd2(G, images, sigmas=bandwidths)
    
    ### W1-DCGAN ###
    mix_rbf_mmd2(G, images, sigmas=bandwidths)
    
    ### LS-GAN ###
    mix_rbf_mmd2(G, images, sigmas=bandwidths)


