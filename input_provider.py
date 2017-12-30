import argparse
import cv2
import lmdb
import numpy
import numpy as np
import scipy.misc
# import hickle as hkl
import os
from os.path import exists, join


def image_val_provider(db_path,limit):
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    
    num_images =  env.stat()['entries']
    
    count=0
    
    
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        
        for key, val in cursor:  # will stop automatically when the cursor reached the last key
            
            if count%1000==0:
                print '%d/%d = %.2f%%' % (count,num_images, count*100.0/num_images)
    
            if count>limit and limit>0:
                print 'break'
                break
            else:
                count+=1
                
            yield key, val
                
        print '%d/%d' % (count,num_images)
        
class ImageProvider(object):
    
    def __init__(self, db_path, batchsize, limit=-1, start=-1):
        
        self.db_path = db_path
        
        self.batchsize = batchsize
        
        self.current_key = None
        
        self.env = lmdb.open(db_path, map_size=1099511627776,
                        max_readers=100, readonly=True)
        
        if start>0:
            self.start=start
        else:
            self.start=0
            
        if limit>0:
            assert limit>self.start
            num_images = limit-self.start
        else:
            num_images =  self.env.stat()['entries']
                   
        self.num_batches = num_images/batchsize
            
        
        def image_provider(db_path=self.db_path):
    
            count=0
    
            with self.env.begin(write=False) as txn:
                cursor = txn.cursor()
                
                cursor.first()
                for i in range(self.start): 
                    cursor.next()   
                # print 'start key=', cursor.key()
        
                for key, val in cursor:  # will stop automatically when the cursor reached the last key
                    
                    count+=1
                    
                    if count>self.num_batches*self.batchsize-2:
                        # print 'returned at %d, %d, %d' % (count,self.num_batches, self.batchsize)
                        cursor.first()
                        for i in range(self.start): 
                            cursor.next()
                        # print 'start key=', cursor.key()
                        count=0
                        
                    # if count%5000==0:
                    #       print '%d/%d = %.2f%%' % (count,self.num_batches*self.batchsize, count*100.0/(self.num_batches*self.batchsize))
   
                    yield key, decode_from_val_to_img(val)
        
        self.iter = image_provider()
        
        
    def next(self, batchsize=None):
        
        imgs = []
        
        if batchsize==None:
            batchsize=self.batchsize
        
        for _ in range(batchsize):
        
            key, img = next(self.iter)  # 01c
            
            imgs.append(img)
        
        
        imgs=np.rollaxis(np.array(imgs),3,1)  # b01c to bc01
        
        self.current_key = key
        
        return imgs  # bc01
    
    def close(self):
        
        self.env.close()
        
        
        



def resize_img(img, img_size):
    
    # input shape= 01c, output shape = 01c
    # print img.shape #(Height Width Channels)

    target_shape = (img_size, img_size, 3) # img shape = (Height Width Channels)

    assert img.dtype == 'uint8', img_name

    if len(img.shape) == 2:
        img = scipy.misc.imresize(img, (img_size, img_size))
        img = np.asarray([img, img, img])
    else:
        if img.shape[2] > 3: # rbga
            img = img[:, :, :3]
        img = scipy.misc.imresize(img, target_shape)
        
        # img = np.rollaxis(img, 2)  # 01c to c01
    return img
    
def decode_from_val_to_img(val):

    return cv2.imdecode(numpy.fromstring(val, dtype=numpy.uint8), 1)
        
def encode_from_img_to_val(img):
    
    r, buf = cv2.imencode(".png",img)

    return buf.tostring()
    
def visualize_img(img):
    
    # input shape = 01c
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(img)
    plt.show()
    

def main_resize():
    
    '''
    
    read_db_resize_img_and_write_to_db
    
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_lmdb_path', type=str)
    parser.add_argument('out_lmdb_path', type=str)
                             
    args = parser.parse_args()
    
    in_lmdb_path = args.in_lmdb_path
    out_lmdb_path = args.out_lmdb_path
    
    
    out_env = lmdb.open(out_lmdb_path, map_size=1099511627776,max_readers=100)

    def write_to_db(key, val):

        with out_env.begin(write=True) as txn:

            txn.put(key, val)
                    
    # out_env = lmdb.open(lmdb_data_name, map_size=int(1e12))
    
    for key,val in image_val_provider(in_lmdb_path,limit=-1):
        
        img=decode_from_val_to_img(val) # 01c
            
        img=resize_img(img, img_size=64)  # 01c
        
        # visualize_img(img)
        
        _val=encode_from_img_to_val(img)
        
        write_to_db(key, _val)
         
    out_env.close()
    
def main_inspect():
    
    '''
    inspect images in db
    
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_lmdb_path', type=str)
                             
    args = parser.parse_args()
    
    in_lmdb_path = args.in_lmdb_path
    
    for key,val in image_val_provider(in_lmdb_path,limit=-1):
        
        img=decode_from_val_to_img(val) # 01c
            
        # img=resize_img(img, img_size=64)  # 01c
        
        visualize_img(img)


if __name__ == '__main__':
    
    # main_inspect()
    
    # main_resize()
    
    batch_sz=100
    train_lmdb = '/scratch/g/gwtaylor/mahe6562/data/lsun/lmdb/bedroom_train_64x64'
    valid_lmdb = train_lmdb #'/scratch/g/gwtaylor/mahe6562/data/lsun/lmdb/bedroom_val_64x64'
    # from input_provider import ImageProvider
    p_train = ImageProvider(train_lmdb,batch_sz,limit=900*batch_sz)
    p_valid = ImageProvider(valid_lmdb,batch_sz,limit=2*900*batch_sz, start=900*batch_sz)
    
    
    for e in range(10):
        
        for batch_j in xrange(p_train.num_batches):
            if batch_j%1000==0:
                print 'batch', batch_j
            data = p_train.next()/ 255.
    
        for batch_j in xrange(p_valid.num_batches):
            if batch_j%1000==0:
                print 'batch', batch_j
            data = p_valid.next()/ 255.
