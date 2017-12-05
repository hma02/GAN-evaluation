source /opt/sharcnet/testing/set4theano-new.sh

mpirun --mca mpi_warn_on_fork 0 -x CUDA_VISIBLE_DEVICES=0,1,2,3 -n $1 --bind-to none python -u test_swap.py
	
