#!/bin/bash
#SBATCH --job-name=main
#SBATCH --account=
#SBATCH --time=0:45:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#
# by default, request two cores (NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on gpus
#
#SBATCH --cpus-per-task=2
#SBATCH --output=out/%x_%j.out


# Purge existing modules
module purge

module use -a /fp/projects01/ec30/software/easybuild/modules/all/

# Load necessary modules
module load JupyterLab/4.0.3-GCCcore-12.2.0

module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-nlptools/01-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8 
module load nlpl-accelerate/0.24.1-foss-2022b-Python-3.10.8

module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8 
module load nlpl-llmtools/03-foss-2022b-Python-3.10.8 
module load nlpl-torchtext/0.15.2-foss-2022b-Python-3.10.8

pip install --user pandas-read-xml

python3 -u download_storting_data.py ${@}