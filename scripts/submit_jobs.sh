#!/bin/bash

# Array of CPU counts to test: 12 for 0.25 node, 24 for 0.5 node, and so on...
declare -a core_counts=(12 24 48 96) # 192 384 768 1536)

# Loop over core counts
for total_tasks in "${core_counts[@]}"; do

    # Calculate nodes based on core counts for naming purposes
    nodes=$(echo "scale=2; $total_tasks/48" | bc)

    # Generate the job script for the current scale
    cat << EOF > "test_script_${nodes}nodes.sh"
#!/bin/bash

#SBATCH --job-name="Python_test_${nodes}nodes"
#SBATCH --time=00:06:00
#SBATCH --ntasks=${total_tasks}
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-tpm-mas

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-mpi4py
module load py-pip

# Set the PYTHONPATH to include local user directory
export PYTHONPATH=$HOME/.local/lib/python3.X/site-packages:$PYTHONPATH

mpiexec -n ${total_tasks} python -m mpi4py.futures benchmark_scaling.py ${nodes}
EOF

    # Modify the benchmark_scaling.py for the current node_multiplier
    sed "s/node_multiplier = [0-9]*\(\.[0-9]*\)\?/node_multiplier = ${nodes}/g" benchmark_scaling.py > "benchmark_scaling_${nodes}.py"
    mv "benchmark_scaling_${nodes}.py" benchmark_scaling.py

    # Submit the job
    sbatch "test_script_${nodes}nodes.sh"
done
