from SALib.sample import saltelli
from SALib.analyze import sobol
import os
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

#Y = np.empty([param_values.shape[0]])
Y = np.empty(10000)

dirList=os.listdir("./outputs")
for i,f in enumerate(dirList):
	merit_fig = np.loadtxt("./outputs/"+f, delimiter=' ')[4]
	Y[i] = float(merit_fig)   #error no index: i,

#param_values = saltelli.sample(problem, 1000, calc_second_order=True)  ##read first 4 columns

Si = sobol.analyze(problem, Y, print_to_console=False)
print Si['S1'],Si['ST']

file = open('./outputs/SI.txt', 'w')
file.write(str(Si))
file.close()

#np.savetxt('./outputs' + '/LU_' + str(a) + '_' + str(b) + '_' + str(radius) + '_' + str(nb_type_index) + '.out', X, delimiter=',')
