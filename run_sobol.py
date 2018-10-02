import os
import numpy as np
import sys

block_num=sys.argv[1]  # index of block

args=np.loadtxt("arg_block_sobol" + block_num + ".txt",delimiter=" ")
#print args[120]

index=1
for arg in args:
	os.system("python ug_ff_sobol.py " + str(arg[0]) + " " + str(arg[1]) + " " +str(int(arg[2])) + " " + str(int(arg[3])) + " " + str(int(arg[4])) )
	print ("Simulation " + str(index) + " is finished.")
	index+=1

