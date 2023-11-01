#!/bin/bash

#SBATCH --job-name="Python_test"
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --partition=compute
#SBATCH --account=research-tpm-mas

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-mpi4py
module load py-pip

python -m pip install --user -U -e git+https://github.com/quaquel/EMAworkbench@MPIEvaluator#egg=ema-workbench

mpiexec python -m mpi4py.futures benchmark.py
