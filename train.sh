if [[ -z $1 ]]; then
	echo 'need to specify the device'
	exit 1
else
	device=$1
fi

if [[ -z $2 ]]; then
	echo 'need to specify the ltype'
	exit 1
else
	ltype=$2
fi


python -u ./main.py -d $device -l $ltype -r 1234