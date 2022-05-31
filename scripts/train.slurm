#!/bin/sh -l
#
#SBATCH --job-name=preprocess
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3G

module load DefaultModules
module load 2022r1
module load compute
# module load python/3.8.12-p6aunbm
module load py-pip
module load openmpi
#module load py-tensorflow
#module load openjdk/11.0.12_7-k7npudp
python -m pip install -r requirements.txt --user

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun sh train_com.sh
