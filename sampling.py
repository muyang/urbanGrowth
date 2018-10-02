from SALib.sample import saltelli
from SALib.analyze import sobol

import numpy as np

problem = {
	'num_vars': 4,
	'names': ['a', 'b', 'r', 'nb'],
	'bounds': [[0.0,3.0],     #'bounds':[a,b], a<b
			   [0.0,1.0],
			   [1,6],
			   [0,2]
			   ]
}

param_values = saltelli.sample(problem, 1000, calc_second_order=True) 

args=[]
for i,p in enumerate(param_values):
	args.append( (p[0],p[1],int( round(p[2]) ),int( round(p[3]) ), i) )
	
cores=700
block_size=len(args)/cores+1

index=1
for i in range(0,len(args),block_size):
	block=args[i:i+block_size]
	np.savetxt('arg_block_sobol' + str(index) + '.txt', block, delimiter=' ')
	index+=1	
	
'''
for p in param_values:
	print p[0],p[1],int( round(p[2]) ),int( round(p[3]) )
'''	
	
'''
def round_up(value):
    return round(value * 100) / 100.0
'''