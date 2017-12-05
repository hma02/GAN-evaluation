import gzip, math
import pickle, cPickle

from sklearn.utils import check_random_state

from thesne.model.tsne import tsne
from thesne.examples import plot

import numpy as np
rng = np.random.RandomState(1234)

def subsample(X, y, size, random_state=None):
    random_state = check_random_state(random_state)

    shuffle = random_state.permutation(X.shape[0])

    X, y = X[shuffle], y[shuffle]
    X, y = X[0:size], y[0:size]

    return X, y



def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data 


'''Given tiles of raw data, this function will return training, validation, and test sets.
r_train - ratio of train set
r_valid - ratio of valid set
r_test  - ratio of test set'''
def gen_train_valid_test(raw_data, raw_target, r_train, r_valid, r_test):
    N = raw_data.shape[0]
    perms = rng.permutation(N)
    raw_data   = raw_data[perms,:]
    raw_target = raw_target[perms]

    tot = float(r_train + r_valid + r_test)  #Denominator
    p_train = r_train / tot  #train data ratio
    p_valid = r_valid / tot  #valid data ratio
    p_test  = r_test / tot	 #test data ratio
    
    n_raw = raw_data.shape[0] #total number of data		
    n_train =int( math.floor(n_raw * p_train)) # number of train
    n_valid =int( math.floor(n_raw * p_valid)) # number of valid
    n_test  =int( math.floor(n_raw * p_test) ) # number of test

    
    train = raw_data[0:n_train, :]
    valid = raw_data[n_train:n_train+n_valid, :]
    test  = raw_data[n_train+n_valid: n_train+n_valid+n_test,:]
    
    train_target = raw_target[0:n_train]
    valid_target = raw_target[n_train:n_train+n_valid]
    test_target  = raw_target[n_train+n_valid: n_train+n_valid+n_test]
    
    print 'Among ', n_raw, 'raw data, we generated: '
    print train.shape[0], ' training data'
    print valid.shape[0], ' validation data'
    print test.shape[0],  ' test data\n'
    
    train_set = [train, train_target]
    valid_set = [valid, valid_target]
    test_set  = [test, test_target]
    return [train_set, valid_set, test_set]



def load_cifar10(path):
    '''processes the raw downloaded cifar10 dataset, and returns test/val/train set'''

    data_batch1 = unpickle(path+'data_batch_1')
    data_batch2 = unpickle(path+'data_batch_2')
    data_batch3 = unpickle(path+'data_batch_3')
    data_batch4 = unpickle(path+'data_batch_4')
    data_batch5 = unpickle(path+'data_batch_5')
    test_batch  = unpickle(path+'test_batch')

    data_batch = {}
    data_batch['data'] = np.concatenate((data_batch1['data'], data_batch2['data']), axis=0)
    data_batch['data'] = np.concatenate((data_batch['data'],  data_batch3['data']), axis=0)
    data_batch['data'] = np.concatenate((data_batch['data'],  data_batch4['data']), axis=0)
    data_batch['data'] = np.concatenate((data_batch['data'],  data_batch5['data']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch1['labels'], data_batch2['labels']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch['labels'], data_batch3['labels']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch['labels'], data_batch4['labels']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch['labels'], data_batch5['labels']), axis=0)
    test_set = [test_batch['data'], np.asarray(test_batch['labels'], dtype='float32')]

    data = gen_train_valid_test(data_batch['data'],data_batch['labels'],8,1,1) 
    train_set, valid_set, _ = data[0], data[1], data[2]
   
    return train_set, valid_set, test_set 



    
def main():

    seed=0
    # Available at http://deeplearning.net/tutorial/gettingstarted.html
    
    # image=True
#
#     if image:
#
#         source='gran'
#
#         if source=='gran':
#             datapath = 'data/samples_gran0.npy'
#             X= np.load(datapath).reshape((1000,3*64*64))
#         elif source=='data':
#
#             # X=load_testset()
#     else:
#         # using dis feature map before the last conv layer
#
#         pass

    datapath='/home/imj/data/cifair10/cifar-10-batches-py/'
    X_data, _, _ = load_cifar10(path=datapath)
    X_data = X_data[0][:5000,:] / 255.
    y_data = np.ones(shape=(len(X_data),))*0
    X_gran = np.load('/home/imj/Documents/gap_testing/gap/code/eval/tmp/samples_gran0.npy').reshape((1000,3*32*32))
    X_dcgan =np.load('/home/imj/Documents/gap_testing/gap/code/eval/tmp/samples_gran3.npy').reshape((1000,3*32*32))

    #X_gran = np.load('/home/imj/Documents/gap_testing/gap/code/eval/tmp/samples_dcgan_chris2_0.npy').reshape((1000,3*32*32))
    #X_dcgan =np.load('/home/imj/Documents/gap_testing/gap/code/eval/tmp/samples_dcgan_chris2_3.npy').reshape((1000,3*32*32))

    y_gran = np.ones(shape=(len(X_gran),))*1
    y_dcgan =np.ones(shape=(len(X_dcgan),))*2
  
    print X_data.shape
    print X_gran.shape
    print X_dcgan.shape

    X = np.concatenate((X_data,X_gran,X_dcgan),axis=0)

    y = np.concatenate((y_data,y_gran,y_dcgan),axis=0)
    
    X, y = subsample(X, y, size=3000, random_state=seed)

    Y = tsne(X, perplexity=100, n_epochs=1000, sigma_iters=50,
             random_state=seed, verbose=1)

    plot.plot(Y, y)


if __name__ == "__main__":
    main()
