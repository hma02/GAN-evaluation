import numpy as np

rng = np.random.RandomState()

size= 8
avoid_ranks = None

winner_ranks = [2,0]

pairs = [winner_ranks]

unpaired=True

i=0

while unpaired:
    
    i=i+1
        
    rand_gpu = list(rng.permutation(size))

    if avoid_ranks==None:
        pass
    else:
        for e_to_avoid in avoid_ranks:
            rand_gpu.remove(e_to_avoid)
    
    pairs = []
    while len(rand_gpu) != 0:
        r1 = rand_gpu.pop()
        r2 = rand_gpu.pop()
        pairs.append([r1,r2])
        
    _up = False
    
    for pair in pairs:
        if set(pair)==set(winner_ranks):
            _up=True
            break
            
    print '#%d try: %s' % (i, pairs)
            
    if not _up:
        break