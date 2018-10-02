#!/bin/bash

###### PBS options for a serial job #############################################

### e.g. short,med,long,ultralong
#PBS -q long
###PBS -l walltime=24:00:00,nodes=1:ppn=1:ib
#PBS -l walltime=24:00:00,nodes=1:ppn=1:ib

###PBS -l walltime=10:00:00,nodes=4:ib

#PBS -N MustafaJob
### [b]egin, [a]bort, [e]nd
### Default: no notification
#PBS -m bae

#PBS -M mustafa.elmorshdy@yahoo.com

### If you do not care about standard output, use "PBS -o /dev/null"
### Default: Name of jobfile plus ".o" plus number of PBS job
#PBS -o output.$PBS_JOBID.block${step}.dat

### This option redirects stdout and stderr into the same output file
### (see PBS option -o).
#PBS -j oe

source /sysdata/shared/sfw/Modules/default/init/bash
### Load modules needed
### module load [compiler modules][MPI modules]
module load gcc/7.2.0 openblas/0.2.17
###module load openmpi/gcc7.2.x/2.1.1/nonthreaded/infiniband
module load openmpi/intel18.0.x/2.1.1/nonthreaded/infiniband

### The following command, if uncommented by deleting the hash sign in front of 'cat',
### saves the name of the compute node (to which the job is submitted by the batch system).
### This information may be useful when debugging.
### This information can also be retrieved while the job is being executed via "qstat -f jobid".
###
### Be sure to use a unique file name (!), though, to avoid concurrent write access
### which may happen when multiple jobs of yours are started simultaneously.
cat $PBS_NODEFILE > $HOME/pbs-machine.$PBS_JOBID.block${step}

### Go to the application's working directory
cd $HOME/simulation/sobol

### Start the application
###mpirun -np 94 ./batch_job.sh
###./batch_job.sh
python run_sobol.py ${step}