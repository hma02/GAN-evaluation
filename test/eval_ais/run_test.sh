TRIALS=(0 1 2 3 4 5 6 7 8 9)
for t in ${TRIALS[@]}
do
	TRIAL=$t RNG_SEED=1235 python run.py
done