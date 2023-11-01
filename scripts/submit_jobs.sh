#!/bin/bash

# Array of CPU counts to test: 12 for 0.25 node, 24 for 0.5 node, and so on...
declare -a core_counts=(48 96 192 384 768 1536)

echo "Installing EMAworkbench..."
python -m pip install --user -U -e git+https://github.com/quaquel/EMAworkbench@MPIEvaluator#egg=ema-workbench
echo "EMAworkbench installed!"

# Loop over core counts
for total_tasks in "${core_counts[@]}"; do

    # Calculate nodes based on core counts for naming purposes
    nodes=$(echo "scale=2; $total_tasks/48" | bc)

    # Generate the job script for the current scale
    cat << EOF > "test_script_${nodes}nodes.sh"
#!/bin/bash

#SBATCH --job-name="Bench_${nodes}nodes"
#SBATCH --time=06:00:00
#SBATCH --ntasks=${total_tasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --account=research-tpm-mas

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-mpi4py
module load py-pip

# Set the PYTHONPATH to include local user directory
export PYTHONPATH=/home/eterhoeven/.local/lib/python3.10/site-packages:$PYTHONPATH

# Export the NODE_MULTIPLIER for benchmark_scaling.py
export NODE_MULTIPLIER=${nodes}

mpiexec -n ${total_tasks} python -m mpi4py.futures benchmark_scaling.py
EOF

    # Submit the job
    sbatch "test_script_${nodes}nodes.sh"
done
