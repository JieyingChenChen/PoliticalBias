#!/bin/bash
#SBATCH --job-name=main 
#SBATCH --account=ec395                
#SBATCH --nodes=1                      
#SBATCH --ntasks-per-node=1          
#SBATCH --mem=20G             
#SBATCH --time=0-03:00:00        
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=out/%x_%j.out 
#SBATCH --exclude=gpu-14

# Purge existing modules
module purge

module use -a /fp/projects01/ec30/software/easybuild/modules/all/

module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-tokenizers/0.21.0-foss-2022b-Python-3.10.8
module load nlpl-accelerate/0.34.2-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
module load JupyterLab/4.0.3-GCCcore-12.2.0

# Define the parameters
# NOR NOR

MODELS=("Llama3-instruct" "Mistral-instruct" "Gemma2-instruct" "Falcon3-instruct")
PROMPTS=(1)
PROMPT_TEMPLATES=(0)
REPLACES=(2)

for model in "${MODELS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do

    for prompt_template in "${PROMPT_TEMPLATES[@]}"; do
      for replace in "${REPLACES[@]}"; do
        echo "Running with model=$model, prompt=$prompt, prompt_template=$prompt_template, replace=$replace"
        python3 -u local_experiment_NOR.py \
          --exp=ide \
          --model="$model" \
          --prompt="$prompt" \
          --replace="$replace" \
          --debug=1 \
          --datasize=200
      done
    done
  done
done


GPTMODELS=("gpt-3.5-turbo" "gpt-4o-mini")
PROMPTS=(1)


for model in "${GPTMODELS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do

    for prompt_template in "${PROMPT_TEMPLATES[@]}"; do
      for replace in "${REPLACES[@]}"; do
        echo "Running with model=$model, prompt=$prompt, prompt_template=$prompt_template, replace=$replace"
        python3 -u gpt_experiment_NOR.py \
          --exp=ide \
          --model="$model" \
          --prompt="$prompt" \
          --replace="$replace" \
          --debug=1 \
          --datasize=200
      done
    done
  done
done



PROMPTS=(1 2 3 11 12 13)

MODELS=("nor-mistral-instruct" "NorwAI")


# Loop over all combinations
for model in "${MODELS[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        for prompt_template in "${PROMPT_TEMPLATES[@]}"; do
            for replace in "${REPLACES[@]}"; do
                echo "Running with model=$model, prompt=$prompt, prompt_template=$prompt_template, replace=$replace"
                python3 -u local_experiment_NOR.py --exp=ide --model="$model" --prompt="$prompt" --replace="$replace" --debug=1 --datasize=200
            done
        done
    done
done


# Purge existing modules
module purge

# NORGPT
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8

#module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.38.2-foss-2022b-Python-3.10.8
#module load nlpl-transformers/4.43.4-foss-2022b-Python-3.10.8
#module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8

module load nlpl-accelerate/0.24.1-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8

module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8
module load nlpl-llmtools/03-foss-2022b-Python-3.10.8
module load nlpl-torchtext/0.15.2-foss-2022b-Python-3.10.8

python3 -u local_experiment_NOR.py ${@}

PROMPTS=(1 2)
MODELS=("NorskGPT")

# Loop over all combinations
for model in "${MODELS[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        for prompt_template in "${PROMPT_TEMPLATES[@]}"; do
            for replace in "${REPLACES[@]}"; do
                echo "Running with model=$model, prompt=$prompt, prompt_template=$prompt_template, replace=$replace"
                python3 -u local_experiment_NOR.py --exp=ide --model="$model" --prompt="$prompt" --replace="$replace" --debug=1 --datasize=200
            done
        done
    done
done
