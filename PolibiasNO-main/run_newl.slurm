#!/bin/bash
#SBATCH --job-name=main 
#SBATCH --account=               
#SBATCH --nodes=1                      
#SBATCH --ntasks-per-node=1          
#SBATCH --mem=40G  
#SBATCH --time=0-07:00:00         
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=out/%x_%j.out
#SBATCH --exclude=gpu-14,gpu-4

# Purge existing modules
module purge

module use -a /fp/projects01/ec30/software/easybuild/modules/all/

#module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
#module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8

module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8

module load nlpl-tokenizers/0.21.0-foss-2022b-Python-3.10.8

module load nlpl-accelerate/0.34.2-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8

#module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8

module load nlpl-llmtools/06-foss-2022b-Python-3.10.8

#module load nlpl-torchtext/0.15.2-foss-2022b-Python-3.10.8

python3 -u local_experiment_NOR.py ${@}