num_worker=$1
unit='g'
mem=$((num_worker*10))$unit
backend=$2

sqsub -q gpu -f mpi -n $num_worker -r 7d -o %J.out --gpp=$num_worker --mpp=$mem --nompirun ./run_eval.sh $num_worker $backend