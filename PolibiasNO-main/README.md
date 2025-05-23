# PolibiasNO

All these Experiments are designed to run on the `FOX` Cloud Computing service at UiO. Using `.slurm`-files.

To reproduce study:

Specify your `ec-group` in the `.slurm` files.

1. Download dataset
2. Process dataset
3. Run Prompt search
4. Run Category prediction (for motions that has no category in the downloaded dataset).
5. Run prompting experiments (LLMs votes)
6. Process Results


## Download dataset

`sbatch run_download_dataset.slurm --first=2018 --last=2023`

## Imperative form

`sbatch run_pros.slurm`



## Prompt search

Note: You must enter a valid OpenAI API-key in  `openai_apikey.txt`

Note: Some of the open source models may require a `Huggingface` access token. This must be entered in  `hf_accesstoken.txt`

`sbatch prompt_test.slurm`

## Categorize 

`sbatch run_gpt.slurm --exp=cat --model=gpt-4o-mini`

## Prompting

Note: `--cont=-1` will start running from scratch. For `N>0` `--cont=N` will start at the Nth (starting from 0) entity and append results to a partly filled results file. This file results file must exist. This is particularly useful for the Entity and Persona experiments in case runs crashes. 

## API models

Note: You must enter a valid OpenAI API-key in  `openai_apikey.txt`

### GPT-3.5-turbo

`sbatch run_gpt.slurm --exp=ide --model=gpt-3.5-turbo --prompt=1 --replace=2 --cont=-1`

`sbatch run_gpt.slurm --exp=ent --model=gpt-3.5-turbo --prompt=1 --replace=2 --cont=-1`

`sbatch run_gpt.slurm --exp=per --model=gpt-3.5-turbot --prompt=1 --replace=2 --cont=-1`

### GPT-4o-mini

`sbatch run_gpt.slurm --exp=ide --model=gpt-4o-mini --prompt=1 --replace=2 --cont=-1`

`sbatch run_gpt.slurm --exp=ent --model=gpt-4o-mini --prompt=1 --replace=2 --cont=-1`

`sbatch run_gpt.slurm --exp=per --model=gpt-4o-mini --prompt=1 --replace=2 --cont=-1`

### Open Models





### Llama3-instruct

`sbatch run_new.slurm --exp=ide --model=Llama3-instruct --prompt=1 --replace=2 --cont=-1`

`sbatch run_newl.slurm --exp=ent --model=Llama3-instruct --prompt=1 --replace=2 --cont=-1`

`sbatch run_new.slurm --exp=per --model=Llama3-instruct --prompt=1 --replace=2 --cont=-1`

### Mistral-instruct

`sbatch run_new.slurm --exp=ide --model=Mistral-instruct --prompt=11 --replace=2 --cont=-1`

`sbatch run_newl.slurm --exp=ent --model=Mistral-instruct --prompt=11 --replace=2 --cont=-1`

`sbatch run_new.slurm --exp=per --model=Mistral-instruct --prompt=11 --replace=2 --cont=-1`

### Gemma2-instruct

`sbatch run_new.slurm --exp=ide --model=Gemma2-instruct --prompt=1 --replace=2 --cont=-1`

`sbatch run_newl.slurm --exp=ent --model=Gemma2-instruct --prompt=1 --replace=2 --cont=-1`

`sbatch run_new.slurm --exp=per --model=Gemma2-instruct --prompt=1 --replace=2 --cont=-1`

### Falcon3-instruct

`sbatch run_new.slurm --exp=ide --model=Falcon3-instruct --prompt=1 --replace=2 --cont=-1`

`sbatch run_newl.slurm --exp=ent --model=Falcon3-instruct --prompt=1 --replace=2 --cont=-1`

`sbatch run_new.slurm --exp=per --model=Falcon3-instruct --prompt=1 --replace=2 --cont=-1`

### NorskGPT

`sbatch run.slurm --exp=ide --model=NorskGPT --prompt=2 --replace=2 --cont=-1`

`sbatch runl.slurm --exp=ent --model=NorskGPT --prompt=2 --replace=2 --cont=-1`

`sbatch run.slurm --exp=per --model=NorskGPT --prompt=2 --replace=2 --cont=-1`
