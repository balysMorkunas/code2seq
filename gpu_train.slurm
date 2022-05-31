#!/bin/sh
#
#SBATCH --job-name="c2s_train"
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G

module load 2022r1
module load gpu
module load openmpi
module load python
module load miniconda3
module load openssh
module load git

echo $(module list)

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /beegfs/home/bmorkunas/.conda/c2s

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/beegfs/scratch/bmorkunas/TensorRT-6.0.1.5/lib

echo $(which python)
echo $CONDA_PREFIX

srun sh train_gpu.sh
