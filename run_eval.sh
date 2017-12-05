if [[ -z $1 ]]; then
	echo 'need to specify the size of BSP'
	exit 1
else
	size=$1
fi


if [[ -z $2 ]]; then
	echo 'need to specify the backend to use: "gpu" or "cuda" '
	exit 1
else
	backend=$2
fi

if [[ "$backend" == "gpu" ]]; then
	echo "cudandarray backend"
	source /opt/sharcnet/testing/set4theano.sh
else
	echo "gpuarray backend"
	source /opt/sharcnet/testing/set4theano-new.sh
fi

pref="--prefix /opt/sharcnet/openmpi/1.8.7/intel-15.0.3/std -x PYTHONPATH=$PYTHONPATH -x PATH=$PATH -x CPATH=$CPATH -x LIBRARY_PATH=$LIBRARY_PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

case $size in
	
	1)
		mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./eval.py $backend'0'
		;;
	2)
		mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./eval.py $backend'0' : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./eval.py $backend'1'
		;;
	4)
        mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./eval.py $backend'0' : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./eval.py $backend'1' : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./eval.py $backend'2' : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./eval.py $backend'3'
		;;	
		
	8)
        mpirun --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'0' : \
               --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'1' : \
               --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'2' : \
               --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'3' : \
			   --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'4' : \
			   --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'5' : \
			   --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'6' : \
			   --mca mpi_warn_on_fork 0 $pref -n 1 -host cop3 python -u ./eval.py $backend'7'
		;;                              
	10)                                 
        mpirun --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'0' : \
               --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'1' : \
               --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'2' : \
               --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'3' : \
			   --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'4' : \
			   --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'5' : \
			   --mca mpi_warn_on_fork 0 $pref -n 1 -host cop4 python -u ./eval.py $backend'6' : \
			   --mca mpi_warn_on_fork 0 $pref -n 1 -host cop3 python -u ./eval.py $backend'2' : \
			   --mca mpi_warn_on_fork 0 $pref -n 1 -host cop3 python -u ./eval.py $backend'3' : \
	           --mca mpi_warn_on_fork 0 $pref -n 1 -host cop3 python -u ./eval.py $backend'7' : \
		;;
	*)
		echo $"Not implemented with this size"
		exit 1
esac
