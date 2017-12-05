num_worker=$1
unit='g'
mem=$((num_worker*10))$unit
backend=$2
mname=$3

sqsub -q gpu -f mpi -n $num_worker -r 7d -o %J.out --gpp=$num_worker --mpp=$mem --nompirun ./run.sh $num_worker $backend $mname $4 $5 $6 $7 $8