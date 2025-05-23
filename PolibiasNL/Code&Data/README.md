Overview
This repository contains all the necessary components for replicating the experiments discussed in our paper on Dutch Political Bias in Large Language Models.

The repository is structured as follows:

PoliBiasNL.csv
This file contains the entire benchmark dataset used in our experiments. This dataset is central to all the analyses conducted in the study.

Results
This folder contains the raw results from all the experiments. These results are further processed to generate the figures and final results presented in the paper.

Code/
This folder contains all the code used to conduct the experiments. The experiments are divided into two main categories: GPT-based experiments and Llama-based experiments. This folder contains the following files:

gpt_experiments.ipynb.
All GPT-related experiments are conducted within a single Jupyter notebook


llama persona experiment.py, llama entity experiment.py and llama ideology experiment.py
The Llama experiments are divided into separate Python files, each focusing on a specific aspect of the analysis

process_results.ipynb
This Jupyter notebook processes the raw results from the Results folder to produce the final figures and results included in the paper.

