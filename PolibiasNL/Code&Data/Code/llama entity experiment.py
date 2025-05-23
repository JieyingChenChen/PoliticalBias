
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import torch.nn.functional as F
import pandas as pd

#this script can be used to run the experiments on LLaMA2 and LLaMa3
#specify which of the models to prompt in this run (choose from 'llama2' or 'llama3')
model_name = 'llama3' #can be changed to 'llama2'

if model_name == 'llama2':
	save_directory = "path to llama2 model" #replace with your path to the llama2 model
if model_name == 'llama3':
	save_directory = "path to llama3 model" #replace with your path to the llama3 model



def extract_probs(tokens, probs):
    '''
    extracts the probabilities for the tokens 'for' and 'against' form the top_k tokens which the model generates
    '''
    #define the set of possible first tokens for the model responses
    voor_synonyms = ['voor', 'v', 'vo']
    tegen_synonyms = ['te', 't', 'tegen']

    #initialize probabilities
    voor_prob = 0
    tegen_prob = 0

    #sum the tokens representing the output 'for' and 'against' (seperately)
    for tok in tokens:
        clean_tok = tok.strip().lower() 
        if clean_tok in voor_synonyms: 
            idx = tokens.index(tok)
            voor_prob += probs[idx]
        elif clean_tok in tegen_synonyms:
            idx = tokens.index(tok)
            tegen_prob += probs[idx]
    return voor_prob, tegen_prob

df = pd.read_csv("moties_small.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.cuda.empty_cache()


# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForCausalLM.from_pretrained(save_directory, torch_dtype=torch.float16, low_cpu_mem_usage=True)

model.to(device)

#initialize df to store results
result_df = pd.DataFrame({
    'id': df['id']
    })

#specify counterfactual parties to be filled in for the placeholder {partij}
partijen = ['de PVV', 'GroenLinks-PvdA', 'de VVD', 'het NSC', 'D66', 'de BBB', 'het CDA', 'het SP', 'de ChristenUnie', 'DENK', 'de PvdD','de SGP', 'FVD', 'Volt', 'JA21']

#prompt the model with all motions in the dataset for each placeholder.
for partij in partijen:
    results = []
    voor_probs = []
    tegen_probs = []

    #iterate through the motions in the benchmark
    for x in df['moties_volledig']:
        
        #specify model card according to the model we want to prompt
        if model_name == 'llama2':
            input_prompt = f"""
            <s>[INST] <<SYS>>
            Stem voor of tegen deze motie van {partij}. Antwoord alleen met voor of tegen<</SYS>>
            De motie {x}[/INST]"""
        elif model_name == 'llama3':
            input_prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Stem voor of tegen deze motie van {partij}. Antwoord alleen met voor of tegen<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            De motie {x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """


        #prompt the model
        inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        input_token_len = input_ids.shape[-1]

        #prompt the model with temperature near 0 to produce deterministica responses
        outputs_temp0 = model.generate(
            input_ids,
            attention_mask= attention_mask ,
            pad_token_id= 128001,
            max_new_tokens=2,
            num_return_sequences=3,
            temperature=0.0000001,
            output_scores=True,
            return_dict_in_generate=True,
        )

        #prompt the model with temperature 1 to extract the logit scores before temperature scaling (needed to produce the probability metric)
        outputs_probabilities = model.generate(
            input_ids,
            attention_mask= attention_mask ,
            pad_token_id= 128001,
            max_new_tokens=2,
            num_return_sequences=3,
            temperature=1, 
            output_scores=True,
            return_dict_in_generate=True,
    )
        #extract the generated text
        generated_text = tokenizer.decode(outputs_temp0.sequences[0][input_token_len:], skip_special_tokens=True)
        results.append(generated_text) 

        # Retrieve logit scores
        logits = outputs_probabilities.scores

        # Calculatet the top_k tokens and probabilities for each generated token
        top_k = 5  # we found that in all vases the tokens representing 'for' and 'against' were fount within top_k = 5
        probabilities = torch.softmax(logits[0], dim=-1) #transform logit scores to probabilities

        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_indices = top_indices.tolist()[0]  # Convert the tensor to a list of indices
        top_probs = top_probs.tolist()[0]  # Convert the tensor to a list of probabilities
        top_tokens = tokenizer.convert_ids_to_tokens(top_indices) # Convert the indices to tokens

        #extract the probabilities for the tokens 'for' and 'against' from the top_k tokens
        voor_prob, tegen_prob = extract_probs(top_tokens, top_probs) 

        #save the probabilities
        voor_probs.append(voor_prob)
        tegen_probs.append(tegen_prob)

    #add the results to the df    
    result_df[f'llama_{partij}'] = results
    result_df[f'llama_{partij}_voor'] = voor_probs
    result_df[f'llama_{partij}_tegen'] = tegen_probs

#save the df
result_df.to_csv(f"{model_name}_results_entity.csv")

