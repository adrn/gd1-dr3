#!/bin/bash   
#SBATCH -J Grid_Search      
#SBATCH -N 2
#SBATCH -p cca   
#SBATCH -t 3-00:00:00
#SBATCH -o ../ceph/log_spur_grid.out        # Output file name   
#SBATCH -e ../ceph/log_spur_grid.err        # Output file name
#SBATCH -C rome
# -C rome
# -C skylake

source ~/.bash_profile

cd /mnt/home/ktavangar/code/


srun python3 -m mpi4py.run -rc thread_level='funneled' spur_grid_search.py --mpi -c config.yaml


