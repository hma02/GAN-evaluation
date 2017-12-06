mtypes=(ls iw js)
ltypes=(gan lsgan wgan)

dir=./tmp
mkdir -p $dir

for (( i=0; i<${#mtypes[@]}; i++ ))
do 
	for (( j=0; j<${#ltypes[@]}; j++ ))
	do
		name=$dir/${ltypes[$j]}_${mtypes[$i]}.out
		./extract.sh ${mtypes[$i]} ${ltypes[$j]} > $name
		grep -oP '(?<=TE: )[0-9]+\.[0-9]+[e-]*[0-9]*\s+\+\-\s+[0-9]+\.[0-9]+[e-]*[0-9]*' $name | cut -d' ' -f1,3 > $name.txt
	done
done

