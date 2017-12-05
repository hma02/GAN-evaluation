num_worker=$1
unit='g'
mem=$((num_worker*10))$unit
backend=$2
mname0=$3
mname1=$4

sqsub -q gpu -f mpi -n $num_worker -r 7d -o %J.out --gpp=$num_worker --mpp=$mem --nompirun ./run_combined.sh $num_worker $backend $mname0 $mname1 $5 $6 $7 $8