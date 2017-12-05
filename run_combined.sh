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

if [[ -z $3 ]]; then
	echo 'need to specify model0 mname: "GRAN" or "DCGAN" '
	exit 1
else
	mname0=$3
fi

if [[ -z $3 ]]; then
	echo 'need to specify model1 mname: "GRAN" or "DCGAN" '
	exit 1
else
	mname1=$4
fi


case $size in
	2)
	    mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'0' 0 $mname0 1 : \
	           --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'1' 0 $mname1 1
		;;
	4)
        mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'0' 0 $mname0 1 : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'1' 0 $mname0 1 : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'2' 0 $mname1 1 : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'3' 0 $mname1 1
		;;	
	*)
		echo $"Not implemented with this size"
		exit 1
esac
