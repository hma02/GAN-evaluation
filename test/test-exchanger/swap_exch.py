from mpi4py import MPI

import numpy as np
rng = np.random.RandomState(1234)

def get_pairs(comm):
    
    rank=comm.rank
    size=comm.size
    
    """Creates permuted pairs for swapping.
    """
    if rank==0:
        rand_gpu = list(rng.permutation(size))
        pairs = []
        while len(rand_gpu) != 0:
            r1 = rand_gpu.pop()
            r2 = rand_gpu.pop()
            pairs.append([r1,r2])

    else:
        pairs = None

    pairs = comm.bcast(pairs,root=0)
    return pairs
    
class Exch_swap(object):
    
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
    
        from pygpu import collectives
    
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
    
        # if rank==0:
        #     comm.send(_string, dest=1)
        # else:
        #     res = comm.recv(source=0)
        #
        #     print res == _string
        #
        # comm.Barrier()

        _local_id.comm_id = bytearray(_string.encode('utf-8'))
        _local_size = len(pair) # how many intra-node workers, in the case of copper maximum 8 workers per node, assuming running within a node here 
    
        if self.interrank==pair[0]:
            _local_rank=0
        else:
            _local_rank=1
        
        _local_rank = _local_rank # assuming running within a node here 
     
        gpucomm = collectives.GpuComm(_local_id,_local_size,_local_rank)
    
        print 'on rank %d, pair %s generated' % (self.interrank, pair)
    
        return gpucomm, pair
    
          
    def prepare(self, ctx, source_param_list):
        
        self.ctx= ctx
        self.source_param_list = source_param_list
        
        # self.ga_local_list = []
        #
        # for param in self.source_param_list:
        #
        #     ga_data = param.container.value
        #     #ga_local_copy = pygpu.asarray(param.get_value(), context=ctx)
        #     ga_local_copy = ga_data.copy(order='C')
        #     #ga_local_copy = pygpu.empty(ga_data.shape, ga_data.dtype, context=ctx)
        #
        #     self.ga_local_list.append(ga_local_copy)

    def exchange(self, pre_random_array):
        
        intracomm, pair = self.get_intranode_comm_pair(pre_random_array)
        intrasize = intracomm.count
        intrarank = intracomm.rank
        
        if self.test==True: 
            
            print 'rank %d exchanges with rank %d' % (pair[0], pair[1])

            print 'before exchange rank%d : %s' % (self.interrank, self.get_1d_value(self.source_param_list[0].get_value()))
            
        
        # use nccl allgather
        for param in self.source_param_list:
            
            resgpu = intracomm.all_gather(param.container.value, nd_up=1)
            
            param.set_value(resgpu[1-intrarank])
            
        if self.test==True:
            
            print 'after exchange rank%d : %s' % (self.interrank, self.get_1d_value(self.source_param_list[0].get_value()))

            
        
        