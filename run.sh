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
	echo 'need to specify the mname: "GRAN" or "DCGAN" '
	exit 1
else
	mname=$3
fi


case $size in
	
	1)
		mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'0' 1 $mname
		;;
	2)
		mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'0' 1 $mname : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'1' 1 $mname
		;;
	4)
        mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'0' 0 $mname : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'1' 0 $mname : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'2' 0 $mname : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'3' 0 $mname
		;;	
		
	8)
        mpirun --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'0' 1 $mname : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'1' 1 $mname : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'2' 1 $mname : \
               --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'3' 1 $mname : \
			   --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'4' 0 $mname : \
			   --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'5' 0 $mname : \
			   --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'6' 0 $mname : \
			   --mca mpi_warn_on_fork 0 --mca btl_smcuda_use_cuda_ipc 1 --mca mpi_common_cuda_cumemcpy_async 1 -n 1 python -u ./main.py $backend'7' 0 $mname
		;;
	*)
		echo $"Not implemented with this size"
		exit 1
esac
