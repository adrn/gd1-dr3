#!/bin/bash 
#SBATCH -o ../ceph/log_combine_fits.out        # Output file name   
#SBATCH -e ../ceph/log_combine_fits.err        # Output file name
#SBATCH -t 3-00:00:00

python combine_fits.py
