#!/bin/bash
#SBATCH --job-name=main 
#SBATCH --account=               
#SBATCH --nodes=1                      
#SBATCH --ntasks-per-node=1          
#SBATCH --mem=20G             
#SBATCH --time=0-00:05:00        
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=out/%x_%j.out 
#SBATCH --exclude=gpu-14,gpu-4

# Purge existing modules
module purge

module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2022b-Python-3.10.8
module load nlpl-llmtools/03-foss-2022b-Python-3.10.8

python3 -u process_to_imperative.py ${@}