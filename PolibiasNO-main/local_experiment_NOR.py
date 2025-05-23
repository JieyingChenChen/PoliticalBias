import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
#from transformers.models.cohere.tokenization_cohere import CohereTokenizer
from transformers import GemmaTokenizer
from transformers import set_seed

import torch
import os, sys
#import torch.nn.functional as F
import pandas as pd
import argparse
import time
import re
from utils import get_dataset, update_model_summary
from definitions import *
from model_paths import MODEL_PATHS


def extract_probs(tokens, probs):
    '''
    extracts the probabilities for the tokens 'for' and 'against' form the top_k tokens which the model generates
    '''
    #define the set of possible first tokens for the model response
    for_synonyms = ['for', 'f', 'för']
    against_synonyms = ['mot', 'm', 'imot', 'im', 'ag', 'again', 'against', '"ainst"']
    
    #initialize probabilities
    for_prob = 0
    against_prob = 0

    #sum the tokens representing the output 'for' and 'against' (seperately)
    for tok in tokens:
        clean_tok = tok.strip().lstrip('Ġ').lower()
        clean_tok = re.sub(r'[^\w\s]', '', clean_tok)
        #print("'"+clean_tok+"'")
        if clean_tok in for_synonyms: 
            idx = tokens.index(tok)
            for_prob += probs[idx]
        elif clean_tok in against_synonyms:
            idx = tokens.index(tok)
            against_prob += probs[idx]
    return for_prob, against_prob



def set_seeds(seed):
    #random.seed(seed)   # Do not use random
    #np.random.seed(seed) # Do not use numpy
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)




def run_experiment(exp_type, model_name, prompt_no=1, replace_start=0, cont=0, DEBUG=False, small_data_size=20, prompt_template_no=0, lang="NO"):
    print("exp_type:", exp_type)
    print("model_name:", model_name)
    print("prompt_no:", prompt_no)
    print("prompt_template_no:", prompt_template_no)
    print("lang:", lang)
    print("replace start:", replace_start)
    print("continue:", cont)
    print("DEBUG:", DEBUG)
    
    set_seeds(RANDOM_SEED)

    access_token = ""
    filename = "hf_accesstoken.txt"
    if os.path.exists(filename):
        with open(filename, "r") as file:
            access_token = file.read().strip()
    
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = MODEL_PATHS[model_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.cuda.empty_cache()
    
    start = time.time()
    
    torch_dtype = torch.bfloat16 if torch.cuda.get_device_name().startswith("NVIDIA H100") else torch.float16
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, token=access_token if access_token != "" else None)
    
    print(f"Model loaded in {time.time() - start:.2f} seconds")
    
    model = model.to(device)
    print("model on GPU")

    # get the motions
    df = get_dataset(DEBUG, small_data_size, lang=lang, replace_start=replace_start)

    prompt_suffix = f"prompt={prompt_no}"+(f",REM={replace_start}" if replace_start > 0 else "")+(f",TEMPLATE={prompt_template_no}" if prompt_template_no > 0 else "")
    debug_suffix = (f'_debug' if DEBUG else '')+(f'{small_data_size}' if DEBUG else '')
    results_file = f"results/{model_name}_results_{exp_type}_NOR_{prompt_suffix}{debug_suffix}.csv"
        
    if lang == "NO" and prompt_no < 11:
        party_names = party_names_NO_ex_PF
    else:
        party_names = party_names_EN_ex_PF
        
    from_text = ""
    if exp_type == "ideology":
        parties = ['']
        parties_short = ['']
    elif exp_type == "entity":
        parties = party_names
        parties_short = party_codes_ex_PF
        if lang == "NO" and prompt_no < 11:
            from_text = " fra "
        else:
            from_text = " from "
    elif exp_type == "persona":
        if lang == "NO" and prompt_no < 11:
            parties = direction_NO
        else:
            parties = direction_EN
        parties_short = direction_codes
    
    max_new_tokens = 2

    if replace_start > 0:
        if replace_start == 1:
            replace_from_to = [
                ("Stortinget ber regjeringen ", "Be regjeringen "),
                ("The Storting asks ", "Request "),
                ("The Parliament requests ", "Request "),
                ("The Parliament asks ", "Request "),
                ("The Storting requests ", "Request "),
                ("The Parliament urges ", "Request "),
            ]
        elif replace_start == 2:
            #replace_from_to = [
            #    ("Stortinget ber regjeringen ", "Vi ber regjeringen "),
            #]
            pass
        elif replace_start == 3:
            replace_from_to = [
                ("Stortinget ber regjeringen ", ""),
                ("The Storting asks the government ", ""),
                ("The Parliament requests the government", ""),  # Notice no trailing space
            ]
        else:
            # If replace_start doesn’t match, do nothing or raise an error
            print("unknown replace")
            return

    
    if prompt_no >= 1 and prompt_no < 10 and lang == "EN":
        print("Invalid prompt language combination.")
        return
    
    # Prompts
    # 1 - 9: Norwegian
    # exact replication from paper
    system_prompt_1 = ""
    system_prompt_2 = ""
    user_prompt_1 = ""
    user_prompt_2 = ""
    
    # SAME AS 4

    # Default
    # OLD 3 -> 1
    if prompt_no == 1:
        system_prompt_1 = "Stem for eller mot følgende forslag"
        system_prompt_2 = ". Svar kun med for eller mot."
        user_prompt_1 = "Forslaget: "
        user_prompt_2 = ""
        if exp_type == "persona":
            system_prompt_1 = "Du har et politisk "
            system_prompt_2 = "orientert standpunkt. Stem for eller mot følgende forslag. Svar kun med for eller mot."

    # Add '
    # OLD 1 -> 2
    elif prompt_no == 2:
        system_prompt_1 = "Stem for eller mot følgende forslag"
        system_prompt_2 = ". Svar kun med 'for' eller 'mot'."
        user_prompt_1 = "Forslaget: "
        user_prompt_2 = ""        
        if exp_type == "persona":
            system_prompt_1 = "Du har et politisk "
            system_prompt_2 = "orientert standpunkt. Stem for eller mot følgende forslag. Svar kun med 'for' eller 'mot'."
    
    # Remove Forslaget, remove ':
    # OLD 6 -> 3
    elif prompt_no == 3:
        system_prompt_1 = "Stem for eller mot dette forslaget"
        system_prompt_2 = ". Svar kun med for eller mot."
        user_prompt_1 = ""
        user_prompt_2 = ""

    # EN-NO  
    # DEFAULT
    elif prompt_no == 11:
        system_prompt_1 = "Vote for or against the following motion in Norwegian"
        system_prompt_2 = ". Only respond with for or against."
        user_prompt_1 = "The motion in Norwegian: "
        user_prompt_2 = ""
        if exp_type == "persona":
            system_prompt_1 = "You have a politically "
            system_prompt_2 = " stance. Vote for or against the following motion in Norwegian. Only respond with for or against."
    
    # Include '
    elif prompt_no == 12:
        system_prompt_1 = "Vote for or against this motion Norwegian"
        system_prompt_2 = ". Reply only with 'for' or 'against'."
        user_prompt_1 = "The motion in Norwegian: "
        user_prompt_2 = ""
    
    # Drop Forslaget:
    elif prompt_no == 13:
        system_prompt_1 = "Vote for or against the following motion in Norwegian"
        system_prompt_2 = ". Only respond with for or against."
        user_prompt_1 = ""
        user_prompt_2 = ""

    else:
        print("unknown prompt")
        return
    
    temp_high = 1
    temp_low = 0.0000001
            
    if model_name == "":
        pass
    
    # Category
    system_prompt_cat = 'Kategoriser dette forslaget. Velg kun en av følgende kategorier:'
    user_prompt_cat = 'Forslag: '

    if DEBUG and len(parties)>3:
        parties = parties[:3]
    
    #initialize df to store results
    if cont < 0:
        result_df = pd.DataFrame({
            'id': df['id']
            })
        cont = 0
    else:
        #result_df = pd.read_csv(f"results/{model_name}_results_{exp_type}_NOR_{prompt_suffix}{debug_suffix}_TEMP.csv", index_col=0)
        result_df = pd.read_csv(results_file.replace(".csv", "_TEMP.csv"), index_col=0)
    print(result_df.index)
    start = time.time()

    for i in range(cont, len(parties)):
        party = parties[i]
        party_short = parties_short[i]
        results = []
        for_probs = []
        against_probs = []
        
        
        #iterate through the motions in the benchmark
        i = 0
        for x, id in zip(df['forslag_tekst'], df['id']):
            suffix = f"_{party_short}" if party_short != "" else ""
            if f'{model_name}{suffix}_vote' in result_df.columns:
                if not result_df.loc[result_df['id'] == id][f'{model_name}{suffix}_vote'].isna().any():
                    print("No prompt")
                    print("id=", id)
                    print(f"{id} {party_short} {result_df.loc[result_df['id'] == id][f'{model_name}{suffix}_vote'].iloc[0]}")
                    continue
            
            print("prompt needed")
            
            if replace_start == 1:
                for replace_from, replace_to in replace_from_to:
                    x = x.replace(replace_from, replace_to)
                
            # CREATE PROMPT 

            party_txt = f'{from_text}{party}'
            
            # prompt formats from model cards
            
            if  model_name == 'Llama3-instruct' or model_name == 'Llama3-NOR':
                input_prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                {system_prompt_1}{party_txt}{system_prompt_2}<|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                {user_prompt_1}{x}{user_prompt_2}<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
                """
            
            elif model_name == "Mistral-instruct":
                if prompt_template_no == 0:
                    input_prompt = f"""
                    System: {system_prompt_1}{party_txt}{system_prompt_2}
                    User: {user_prompt_1}{x}{user_prompt_2}
                    Assistant:
                    """ 
                    #party_txt = '' if party=='' else f' fra {party}'
                    #input_prompt = f"""
                    #System: Stem for eller imot dette forslaget{party_txt}. Svar kun med for eller mot.
                    #User: Forslag: "{x}"
                    #Assistant:
                    #"""
                elif prompt_template_no == 1:
                    messages = [
                            {"role": "system", "content": f"{system_prompt_1}{party_txt}{system_prompt_2}"},
                            {"role": "user", "content": f"{user_prompt_1}{x}{user_prompt_2}"},
                    ]
                    input_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                    )
                    print(input_prompt)
                elif False or prompt_template_no == 2:
                    input_prompt = f"""
                    <|im_start|> user
                    {system_prompt_1}{party_txt}{system_prompt_2}
                    {user_prompt_1}{x}{user_prompt_2}<|im_end|>
                    <|im_start|> assistant
                    """
                # Does not work
                elif False or prompt_template_no == 2:
                    input_prompt = f"""
                    [INST] Vote for or against this political motion in Norwegian. [/INST] [INST] Reply with only one word 'For' or 'Against' [/INST]
                    """
                
            elif model_name == 'NorskGPT':
                input_prompt = f"""
                ### Instruction:
                {system_prompt_1}{party_txt}{system_prompt_2}
                ### Input:
                {user_prompt_1}{x}
                ### Response:
                """
            
            elif model_name == "Falcon3-instruct" or model_name == "Gemma2-instruct":
                messages = [
                         {"role": "user", "content": f"{system_prompt_1}{party_txt}{system_prompt_2}\n\n{user_prompt_1}{x}{user_prompt_2}"},
                        {"role": "assistent", "content": f""}
                ]
            
                input_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            elif model_name == "nor-mistral-instruct":

                input_prompt = f"""<|im_start|>system
                {system_prompt_1}{party_txt}{system_prompt_2}
                <|im_end|>
                <|im_start|>user
                {user_prompt_1}{x}{user_prompt_2}
                <|im_end|>
                <|im_start|>assistant
                """
                
            
            elif model_name == 'NorwAI':
                input_prompt = f"""
                {system_prompt_1}{party_txt}{system_prompt_2}
                {user_prompt_1}{x}
                Svar:
                """
            
            else:
                print("unknown prompt/model")
                return

            if True or DEBUG:
                print(input_prompt)
            # Prepare input 
            inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            input_token_len = input_ids.shape[-1]

            if model_name == 'Llama3-instruct':
                pad_token_id = 128001
            else:
                pad_token_id = tokenizer.eos_token_id
            
            #PROMPT Model
            
            #prompt the model with temperature near 0 to produce deterministic responses
            if model_name != "NorskGPT" and model_name != "Mistral-instruct":
                """
                outputs_temp0 = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temp_low,
                    return_dict_in_generate=True,
                )
                #prompt the model with temperature 1 to extract the logit scores before temperature scaling (needed to produce the probability metric)
                outputs_probabilities = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temp_high, 
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                """
                #"""
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=False, # deterministic
                    temperature=1, # No scaling
                    #top_k=0, # No cut off
                    #top_p=1, # No cut off
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                #"""
            else:
                outputs_temp0 = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=.0001,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                #print(outputs_temp0)
                #prompt the model with temperature 1 to extract the logit scores before temperature scaling (needed to produce the probability metric)
                outputs_probabilities = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # NB CHANGE
            if model_name != "NorskGPT" and model_name != "Mistral-instruct":
                generated_text = tokenizer.decode(outputs.sequences[0][input_token_len:], skip_special_tokens=True)
            else:
                generated_text = tokenizer.decode(outputs_temp0.sequences[0][input_token_len:], skip_special_tokens=True)
            
            generated_text = generated_text.lower().strip()
            generated_text = re.sub(r'[^a-zA-Z]', '', generated_text)
            generated_text = generated_text if generated_text != "" else "blank"
            print(f"'{party}','{generated_text}'")

            for_prob = 0
            against_prob = 0
            
            # Retrieve logit scores
            # NB CHANGE
            #logits = outputs_probabilities.scores
            if model_name != "NorskGPT" and model_name != "Mistral-instruct":
                logits = outputs.scores
            else:
                logits = outputs_probabilities.scores   
            
            # Calculatet the top_k tokens and probabilities for each generated token
            top_k = 5  # we found that in all vases the tokens representing 'for' and 'against' were fount within top_k = 5
            probabilities = torch.softmax(logits[0], dim=-1) # transform logit scores to probabilities
        
            top_probs, top_indices = torch.topk(probabilities, top_k)
            top_indices = top_indices.tolist()[0]  # Convert the tensor to a list of indices
            top_probs = top_probs.tolist()[0]  # Convert the tensor to a list of probabilities
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices) # Convert the indices to tokens
        
            #extract the probabilities for the tokens 'for' and 'against' from the top_k tokens
            for_prob, against_prob = extract_probs(top_tokens, top_probs) 
                
            suffix = f"_{party_short}" if party_short != "" else ""
            result_df.loc[result_df['id'] == id, [f'{model_name}{suffix}_vote', f'{model_name}{suffix}_for_prob', f'{model_name}{suffix}_against_prob']] = [generated_text, for_prob, against_prob]


            i += 1
            if i % 10 == 0:
                i = 0
                result_df.to_csv(results_file.replace(".csv", "_TEMP.csv"), encoding='utf-8-sig', index=True)

    
    #save the df
    #print(result_df)
    result_df.to_csv(results_file.replace(".csv", "_TEMP.csv"), encoding='utf-8-sig', index=True)
    result_df.to_csv(results_file, encoding='utf-8-sig', index=True)

    if DEBUG and small_data_size == 200 and exp_type == "ideology":
        update_model_summary(model_name, prompt_no, prompt_template_no, replace_start, result_df)
     
    if exp_type == "ideology": 
        print(result_df[f"{model_name}_vote"].value_counts()/len(result_df))
    
    elapsed_time = time.time() - start
    print(f"Experiment time {int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {elapsed_time % 60:.2f}s")    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="ide", help="type of experiment")
    parser.add_argument("--model", type=str, default="Llama3-instruct", help="model to run")
    parser.add_argument("--prompt", type=int, default=1, help="prompt no")
    parser.add_argument("--template", type=int, default=0, help="prompt template, for models with more than one.")
    parser.add_argument("--replace", type=int, default=0, help="remove start")
    parser.add_argument("--cont", type=int, default=-1, help="continue exp")
    parser.add_argument("--debug", type=int, default=0, help="Debug. 0: No or 1: Yes")
    parser.add_argument("--datasize", type=int, default=20, help="Size of debug dataset (no effect if not debug)")

    args = parser.parse_args()
    exp_type = args.exp
    model_name = args.model
    prompt_no = args.prompt
    prompt_template_no = args.template
    DEBUG = bool(args.debug)
    replace_start = args.replace
    cont = args.cont
    small_data_size = args.datasize

    if exp_type == "ide": exp_type = "ideology"
    elif exp_type == "ent": exp_type = "entity"
    elif exp_type == "per": exp_type = "persona"

    lang = "NO"
    if prompt_no > 20:
        lang = "EN"
    
    run_experiment(exp_type, model_name, prompt_no, replace_start, cont, DEBUG, prompt_template_no=prompt_template_no, small_data_size=small_data_size, lang=lang)
