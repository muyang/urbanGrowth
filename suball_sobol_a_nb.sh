#!/bin/bash
# Content:
#     This program aims to sub all.
# Example:
#     ./suball
#       from equilibrium add 0.1 each time ,total 30 times
#-----------------------------------------------

#for ((COUNTER=301; COUNTER<=400; ++COUNTER))
#	do
#		qsub ./pbsrun_short_sobol.sh -v step=${COUNTER}
#done

for ((COUNTER=1; COUNTER<=10; ++COUNTER))
	do
		qsub ./pbsrun_long_sobol.sh -v step=${COUNTER}
done

#for ((COUNTER=1; COUNTER<=100; ++COUNTER))
#	do
#		qsub ./pbsrun_med_sobol.sh -v step=${COUNTER}
#done