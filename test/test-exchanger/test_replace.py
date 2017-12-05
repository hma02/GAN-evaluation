
import numpy as np

rng = np.random.RandomState(1234)
# inter-node comm

def get_internode_comm():
    
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    
    return comm


if __name__ == '__main__':
    
    
    
    comm = get_internode_comm()
    
    rank=comm.rank
    device='cuda'+str(rank)
    size=comm.size
    
    import sys
    
    sys.path.append('../..')
    
    import base.subnets.layers.someconfigs as someconfigs
    someconfigs.backend='gpuarray'

    from test_exchanger import init_device, clean_device
    _,ctx,arr,shared_x,shared_xx, f = init_device(device=device, rng=np.random.RandomState(rank+60))      

    # prepare nccl32 exchanger

    from base.swp import Exch_swap_gpuarray, get_winner

    exch = Exch_swap_gpuarray(comm, test=True)

    exch.prepare(ctx, [shared_x])
    
    winner_ranks = [1,0]
    
    pairs = get_winner(comm, rng, winner_ranks)

    exch.replace(winner_ranks, pairs)
    

    #
    # print 'after func rank%d: %s' % (rank, shared_x.get_value())

