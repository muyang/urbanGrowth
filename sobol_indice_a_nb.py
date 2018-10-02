from SALib.sample import saltelli
from SALib.analyze import sobol
import os
import numpy as np

problem = {
	'num_vars': 2,
	'names': ['a', 'nb'],
	'bounds': [[0.0,1.0],     #'bounds':[a,b]
	           [0,2]
			   ]
}

#Y = np.empty([param_values.shape[0]])
Y = np.empty(6000)

dirList=os.listdir("./outputs/uasa_a_nb")
for i,f in enumerate(dirList):
	merit_fig = np.loadtxt("./outputs/uasa_a_nb/"+f, delimiter=' ')[4]
	Y[i] = float(merit_fig)   #error no index: i,

#param_values = saltelli.sample(problem, 1000, calc_second_order=True)  ##read first 4 columns

Si = sobol.analyze(problem, Y, print_to_console=False)
print Si['S1'],Si['ST']

file = open('./outputs/uasa_a_nb/SI.txt', 'w')
file.write(str(Si))
file.close()

#np.savetxt('./outputs' + '/LU_' + str(a) + '_' + str(b) + '_' + str(radius) + '_' + str(nb_type_index) + '.out', X, delimiter=',')
