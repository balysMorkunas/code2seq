#!/bin/sh
#
#SBATCH --job-name="preprocess"
#SBATCH --partition=memory
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=1500G

module load DefaultModules
module load 2022r1
module load compute
module load py-pip
module load openjdk/11.0.12_7-k7npudp

srun bash preprocess.sh --unbuffered

