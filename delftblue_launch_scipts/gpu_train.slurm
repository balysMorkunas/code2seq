#!/bin/sh -l
#
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:2

module load 2022r1
module load gpu
# module load python/3.8.12-p6aunbm
module load py-pip
module load openmpi
#module load py-tensorflow
#module load openjdk/11.0.12_7-k7npudp
python -m pip install -r requirements.txt --user

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun sh train.sh
