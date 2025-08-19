#!/bin/bash --login
### Choose ONE of the following partitions depending on your permitted access

#SBATCH -p gpuA              # A100 (80GB) GPUs  [up to 12 CPU cores per GPU permitted]

### Required flags
#SBATCH -G 1                 # (or --gpus=N) Number of GPUs 
#SBATCH -t 4-0               # Wallclock timelimit (1-0 is one day, 4-0 is max permitted)
### Optional flags
#SBATCH -n 8          # (or --ntasks=) Number of CPU (host) cores (default is 1)
                             # See above for number of cores per GPU you can request.
                             # Also affects host RAM allocated to job unless --mem=num used.

module purge
module load compilers/gcc/13.3.0
module load libs/cuda/11.7.0
module load libs/cuDNN/8.5.0
nvcc --version


echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"



source /mnt/iusers01/fatpou01/compsci01/x47085ha/scratch/venv/robot/bin/activate

python mpx/examples/train_quad.py
# python mpx/examples/train_talos.py
