
import numpy as np

# inter-node comm

def get_internode_comm():
    
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    
    return comm

# intra-node comm

# def  get_intranode_comm(rank,size, ctx):
#
#     from pygpu import collectives
#
#     _local_id = collectives.GpuCommCliqueId(context=ctx)
#
#     string =  _local_id.comm_id.decode('utf-8')
#
#     import os
#     pid = str(os.getpid())
#     len_pid =len(pid)
#
#     # replace the process-unique id to be the universal id "0......" so that a intranode gpucomm can be created
#     replacement = ''.join('0' for i in range(len_pid))
#     _string = string.replace(pid, replacement)
#
#     # if rank==0:
#     #     comm.send(_string, dest=1)
#     # else:
#     #     res = comm.recv(source=0)
#     #
#     #     print res == _string
#     #
#     # comm.Barrier()
#
#     _local_id.comm_id = bytearray(_string.encode('utf-8'))
#     _local_size = size # how many intra-node workers, in the case of copper maximum 8 workers per node, assuming running within a node here
#     _local_rank = rank # assuming running within a node here
#
#     gpucomm = collectives.GpuComm(_local_id,_local_size,_local_rank)
#
#     return gpucomm
#
# def  get_intranode_comm_pair(rank, size, ctx ,pre_random_array):
#
#     from pygpu import collectives
#
#     _local_id = collectives.GpuCommCliqueId(context=ctx)
#
#     string =  _local_id.comm_id.decode('utf-8')
#
#     import os
#     pid = str(os.getpid())
#     len_pid =len(pid)
#
#     # replace the process-unique id to be the universal id "0......" so that a intranode gpucomm can be created
#
#
#     pair = []
#     for index, tmp_pair in enumerate(pre_random_array):
#         if (tmp_pair[0]==rank) or (tmp_pair[1] ==rank):
#             # print "Found it !" ,tmp_pair
#             pair = tmp_pair
#             pair_index=index
#             break
#
#     assert pair_index<=9
#     replacement = ''.join(('%d' % pair_index) for i in range(len_pid))
#     _string = string.replace(pid, replacement)
#
#     # if rank==0:
#     #     comm.send(_string, dest=1)
#     # else:
#     #     res = comm.recv(source=0)
#     #
#     #     print res == _string
#     #
#     # comm.Barrier()
#
#     _local_id.comm_id = bytearray(_string.encode('utf-8'))
#     _local_size = len(pair) # how many intra-node workers, in the case of copper maximum 8 workers per node, assuming running within a node here
#
#     if rank==pair[0]:
#         _local_rank=0
#     else:
#         _local_rank=1
#
#     _local_rank = _local_rank # assuming running within a node here
#
#     gpucomm = collectives.GpuComm(_local_id,_local_size,_local_rank)
#
#     print 'on rank %d, pair %s generated' % (rank, pair)
#
#     return gpucomm


if __name__ == '__main__':
    
    comm = get_internode_comm()
    
    rank=comm.rank
    device='cuda'+str(rank)
    size=comm.size

    from test_exchanger import init_device, clean_device
    _,ctx,arr,shared_x,shared_xx, f = init_device(device=device, rng=np.random.RandomState(rank+60))      

    # prepare nccl32 exchanger

    from swap_exch import Exch_swap, get_pairs

    exch = Exch_swap(comm, test=True)

    exch.prepare(ctx, [shared_x])
        
    pairs = get_pairs(comm)

    exch.exchange(pairs)
    

    #
    # print 'after func rank%d: %s' % (rank, shared_x.get_value())

