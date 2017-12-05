import base.subnets.layers.someconfigs as someconfigs

if someconfigs.backend == 'cudandarray':
    
    import pycuda.gpuarray as gpuarray
    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils
    
elif someconfigs.backend == 'gpuarray':
    
    from pygpu import collectives
        

def get_winner(comm, rng, winner_ranks, avoid_ranks=None):
    
    # split winners in winner_ranks and pair each winner with each of the losers
    rank=comm.rank
    size=comm.size
    
    """Creates permuted pairs for swapping.
    """
    if rank==0:
        
        pairs = [winner_ranks]

        unpaired=True
        
        while unpaired:
    
            rand_gpu = list(rng.permutation(size))

            if avoid_ranks==None:
                pass
            else:
                for e_to_avoid in avoid_ranks:
                    rand_gpu.remove(e_to_avoid)

            pairs = []
            while len(rand_gpu) != 0:
                r1 = rand_gpu.pop()
                r2 = rand_gpu.pop()
                pairs.append([r1,r2])
    
            _up = False

            for pair in pairs:
                if set(pair)==set(winner_ranks):
                    _up=True # winners are still together, keep trying
                    break
        
            if not _up:
                break

    else:
        pairs = None

    pairs = comm.bcast(pairs,root=0)
    return pairs
    
    
    
def get_pairs(comm, rng, avoid_ranks=None):
    
    rank=comm.rank
    size=comm.size
    
    """Creates permuted pairs for swapping.
    """
    if rank==0:
        rand_gpu = list(rng.permutation(size))
        
        if avoid_ranks==None:
            pass
        else:
            for e_to_avoid in avoid_ranks:
                rand_gpu.remove(e_to_avoid)
            
        pairs = []
        while len(rand_gpu) != 0:
            r1 = rand_gpu.pop()
            r2 = rand_gpu.pop()
            pairs.append([r1,r2])

    else:
        pairs = None

    pairs = comm.bcast(pairs,root=0)
    return pairs
    

class Exch_swap_cudandarray(object):
    """
    CUDA-aware parameter transfer
    """
    def __init__(self, comm):
        self.comm = comm
        self.size = self.comm.size
        self.rank = self.comm.rank


        # for CUDA-aware MPI
        #bufint_cn= lambda arr: arr.container.value.as_buffer(arr.container.value.size*4,0)
        self.bufint = lambda arr: arr.gpudata.as_buffer(arr.nbytes)

    def dtype_to_mpi(self, t):
        
        from mpi4py import MPI
        import numpy as np
    
        if hasattr(MPI, '_typedict'):
            mpi_type = MPI._typedict[np.dtype(t).char]
        elif hasattr(MPI, '__TypeDict__'):
            mpi_type = MPI.__TypeDict__[np.dtype(t).char]
        else:
            raise ValueError('cannot convert type')
        return mpi_type

    def prepare(self, ctx, drv, source_param_list, dest_param_list=None):
        self.source_param_list = source_param_list
        self.ctx = ctx
        self.drv = drv
        if dest_param_list!=None:
            self.dest_param_list = dest_param_list
        else:
            self.dest_param_list = self.source_param_list
         
        self.param_update_ga_list = []
        #MPI data type, should input fp32 array here
        for param in self.source_param_list:
            # Prepare data in host (CPU) memory
            param_update = param.get_value()
            
            #Prepare data in decive (GPU) memory
            param_update_ga = gpuarray.to_gpu(param_update)
            self.param_update_ga_list.append(param_update_ga)


        self.mpidtype = self.dtype_to_mpi(self.param_update_ga_list[0].dtype)
 

    def exchange(self, pre_random_array):
        # pass explicit MPI datatypes
        # copy weight from param_ga to param_update_ga
        for param, param_update_ga in \
                        zip(self.source_param_list, self.param_update_ga_list):

            param_ga = \
             theano.misc.pycuda_utils.to_gpuarray(param.container.value)

            self.drv.memcpy_dtod(param_update_ga.ptr,
                                  param_ga.ptr,
                                  param_ga.dtype.itemsize *
                                  param_ga.size)
                                  
            self.ctx.synchronize() 

        pair = []
        for tmp_pair in pre_random_array:
            if (tmp_pair[0]==self.rank) or (tmp_pair[1] ==self.rank):
                # print "Found it !" ,tmp_pair
                pair = tmp_pair
                break

        for param in self.param_update_ga_list: 
            if (self.rank == pair[0]):
                self.comm.Sendrecv_replace([self.bufint(param),self.mpidtype],dest=pair[1],source=pair[1])
            
            elif (self.rank == pair[1]):
                self.comm.Sendrecv_replace([self.bufint(param),self.mpidtype],dest=pair[0],source=pair[0])

        # copy weight from param_update_ga back to param_ga
        for param, param_update_ga in \
                        zip(self.dest_param_list, self.param_update_ga_list):
            # dest_param_lsit contains the shared values of our original source_param.
            # Just to copy param_update_ga into param create a fake cotainer.
            param_ga = \
             theano.misc.pycuda_utils.to_gpuarray(param.container.value)

            self.drv.memcpy_dtod(param_ga.ptr,
                                  param_update_ga.ptr,
                                  param_update_ga.dtype.itemsize *
                                  param_ga.size)
                      
            self.ctx.synchronize()
            
            return pair
            
    def replace(self, winner_ranks, pre_random_array):  
        # replace the bad rank with winner rank params
        
        # pass explicit MPI datatypes
        # copy weight from param_ga to param_update_ga
        for param, param_update_ga in \
                        zip(self.source_param_list, self.param_update_ga_list):

            param_ga = \
             theano.misc.pycuda_utils.to_gpuarray(param.container.value)

            self.drv.memcpy_dtod(param_update_ga.ptr,
                                  param_ga.ptr,
                                  param_ga.dtype.itemsize *
                                  param_ga.size)
                                  
            self.ctx.synchronize() 

        pair = []
        for tmp_pair in pre_random_array:
            if (tmp_pair[0]==self.rank) or (tmp_pair[1] ==self.rank):
                # print "Found it !" ,tmp_pair
                pair = tmp_pair
                break
                
        import copy
        bk_pair = copy.deepcopy(pair)
        
        bk_pair.remove(self.rank)
        
        partner = bk_pair[0]

        for param in self.param_update_ga_list: 
            if self.rank in winner_ranks:
                self.comm.Send([self.bufint(param),self.mpidtype],dest=partner)
            
            elif self.rank not in winner_ranks:
                self.comm.Recv([self.bufint(param),self.mpidtype],source=partner)

        # copy weight from param_update_ga back to param_ga
        for param, param_update_ga in \
                        zip(self.dest_param_list, self.param_update_ga_list):
            # dest_param_lsit contains the shared values of our original source_param.
            # Just to copy param_update_ga into param create a fake cotainer.
            param_ga = \
             theano.misc.pycuda_utils.to_gpuarray(param.container.value)

            self.drv.memcpy_dtod(param_ga.ptr,
                                  param_update_ga.ptr,
                                  param_update_ga.dtype.itemsize *
                                  param_ga.size)
                      
            self.ctx.synchronize()
            
            return pair  

    
class Exch_swap_gpuarray(object):
    
    def __init__(self, intercomm, test=False):
        
        self.intercomm = intercomm
        self.intersize = intercomm.size
        self.interrank = intercomm.rank
        
        self.test=test
        
    def get_1d_value(self, ndarray):
        
        array = ndarray
        dim_left =  array.ndim 
        
        while dim_left!=1:
            
            array = array[0]
            
            dim_left = array.ndim
            
            # print dim_left
            
        return array  
        
    def get_intranode_comm_pair(self, pre_random_array):
    
        _local_id = collectives.GpuCommCliqueId(context=self.ctx)

        string =  _local_id.comm_id.decode('utf-8')

        import os
        pid = str(os.getpid())
        len_pid =len(pid)

        # replace the process-unique id to be the universal id "0......" so that a intranode gpucomm can be created
    
    
        pair = []
        for index, tmp_pair in enumerate(pre_random_array):
            if (tmp_pair[0]==self.interrank) or (tmp_pair[1] ==self.interrank):
                # print "Found it !" ,tmp_pair
                pair = tmp_pair
                pair_index=index
                break
            
        assert pair_index<=9
        replacement = ''.join(('%d' % pair_index) for i in range(len_pid))
        _string = string.replace(pid, replacement)

        _local_id.comm_id = bytearray(_string.encode('utf-8'))
        _local_size = len(pair) # how many intra-node workers, in the case of copper maximum 8 workers per node, assuming running within a node here 
    
        if self.interrank==pair[0]:
            _local_rank=0
        else:
            _local_rank=1
        
        _local_rank = _local_rank # assuming running within a node here 
     
        gpucomm = collectives.GpuComm(_local_id,_local_size,_local_rank)
    
        if self.test==True: 
            print 'on rank %d, pair %s generated' % (self.interrank, pair)
    
        return gpucomm, pair
    
          
    def prepare(self, ctx, source_param_list):
        
        self.ctx= ctx
        self.source_param_list = source_param_list

    def exchange(self, pre_random_array):
        
        intracomm, pair = self.get_intranode_comm_pair(pre_random_array)
        intrasize = intracomm.count
        intrarank = intracomm.rank
        
        if self.test==True: 
            
            print 'rank %d exchanges with rank %d' % (pair[0], pair[1])

            print 'before exchange rank%d : %s' % (self.interrank, self.get_1d_value(self.source_param_list[0].get_value()))
            
        # allgather between two ranks is equivalent to sendrecv
        
        # use nccl allgather
        for param in self.source_param_list:
            
            resgpu = intracomm.all_gather(param.container.value, nd_up=1)
            
            param.set_value(resgpu[1-intrarank])
            
        if self.test==True:
            
            print 'after exchange rank%d : %s' % (self.interrank, self.get_1d_value(self.source_param_list[0].get_value()))
        
        return pair
        
    def replace(self, winner_ranks, pre_random_array):
        
        
        intracomm, pair = self.get_intranode_comm_pair(pre_random_array)
        intrasize = intracomm.count
        intrarank = intracomm.rank
        
        if self.test==True: 
            
            print 'rank %d exchanges with rank %d' % (pair[0], pair[1])

            print 'before exchange rank%d : %s' % (self.interrank, self.get_1d_value(self.source_param_list[0].get_value()))
        
        
        if self.interrank not in winner_ranks:
            
            loser = self.interrank
            loser_gpurank =intrarank
            
            import copy
            bk_pair = copy.deepcopy(pair)
            bk_pair.remove(self.interrank)
            
            winner = bk_pair[0]
            winner_gpurank = 1-loser_gpurank
        else:
            winner = self.interrank
            winner_gpurank = intrarank
            
            
        if self.test==True:
            print 'on rank %d (gpurank %d) winner is %d (gpurank %d)' % (self.interrank, intrarank, winner, winner_gpurank)
            
        # bcast between two ranks is equivalent to send
        
        # use nccl bcast
        for param in self.source_param_list:

            intracomm.broadcast(param.container.value, root=winner_gpurank)
            
        
        if self.test==True:
            
            print 'after exchange rank%d : %s' % (self.interrank, self.get_1d_value(self.source_param_list[0].get_value()))
        
        return pair
        

