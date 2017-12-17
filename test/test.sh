
device=$1
mtype=$2
ltype=$3
ckernr=$4

if [[ -z $device ]]
then
	echo 'need device as arg 1'
	exit 1
fi

if [[ -z $mtype ]]
then
	echo 'need mtype as arg 2'
	exit 1
fi

if [[ -z $ltype ]]
then
	echo 'need ltype as arg 3'
	exit 1
fi

if [[ -z $ckernr ]]
then
	echo 'need ckernr as arg 4'
	exit 1
fi


dir=$(pwd)

function test_size
{
	
	ckernr=$1
	
	device=$2
	
	mtype=$3
	
	ltype=$4

	echo "testing $ltype ckern:$ckernr on $mtype on $device"
	
	name=$dir/$mtype/$ltype

	mkdir -p $name

	CRI_KERN=128 python -u mnnd_gather.py -d $device -m $mtype -l $ltype -c $ckernr &> $name/$ckernr.txt
}


test_size $ckernr $device $mtype $ltype
