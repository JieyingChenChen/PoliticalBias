import pandas as pd
import numpy as np
import string

from openai import OpenAI
from tqdm import tqdm
import ast
import math
from utils import get_dataset, update_model_summary
from definitions import *
import argparse
import time


with open("openai_apikey.txt", "r") as file:
    openai_api_key = file.read().strip()

#specify your api_key
client = OpenAI(
    api_key= openai_api_key, #replace the empty string with your openAI API key
)


def call_gpt(model_name, motie, prompt_template):
    '''
    takes the prompt template and motion as imput and makes a call the gpt API, thereby returning the generated text and logprobabilities.
    '''
    
    #make api call
    completion = client.chat.completions.create(
        model=model_name,
    
        messages=[
          {"role": "system", "content": prompt_template},
          {"role": "user", "content": motie}
        ],

        #specify hyperparameters
        seed=RANDOM_SEED,
        temperature=0,
        logprobs=True,
        top_logprobs=20 
    )
    print(motie)
    
    #extract generated text and top 20 tokens with the highest probabilities along with their log probabilities
    choices = completion.choices[0].message.content
    logprobabilities = completion.choices[0].logprobs.content[0].top_logprobs
    loplogprobdict = {}
    
    for x in logprobabilities:
        loplogprobdict[x.token] = x.logprob
    
    return choices, loplogprobdict


def extract_logprobs(logprob_dict):
    ''' from the top 20 tokens we this function extracts the probabilities for the tokens 'for' and 'against' '''
    for_synonyms = ['for', 'f', '▁for', 'för', '▁för']
    against_synonyms = ['mot', 'm', '_mot', 'against', '_against', '_ag']


    for_lprob = 0
    against_lprob = 0
    for word, lprob in logprob_dict.items():
        if word.lower().strip() in for_synonyms:
            for_lprob = lprob
            break
    
    for word, lprob in logprob_dict.items():
        if word.lower().strip() in against_synonyms:
            against_lprob = lprob
            break
    
    return for_lprob, against_lprob


def run_experiment(exp_type, model_name, prompt_no=1, replace_start=0, cont=0, DEBUG=False, small_data_size=20, prompt_template_no=0):
    print("exp_type:", exp_type)
    print("model_name:", model_name)
    print("prompt_no:", prompt_no)
    print("prompt_template_no:", prompt_template_no)
    print("replace start:", replace_start)
    print("continue:", cont)
    print("DEBUG:", DEBUG)
    
    
    if not (model_name == "gpt-3.5-turbo" or model_name == "gpt-4o-mini"):
        print("unknown model")
        return

    # Change
    df = get_dataset(DEBUG, small_data_size, exp=exp_type, replace_start=replace_start)
    print("len(df):", len(df))
    prompt_lan = "NO"
    
    # EXPERIMENT TYPE
    if exp_type == "ideology" or exp_type == "category" or exp_type == "translate":
        parties = ['']
        parties_short = ['']
    elif exp_type == "entity":
        parties = party_names_NO_ex_PF
        parties_short = party_codes_ex_PF
    elif exp_type == "persona":
        if prompt_lan == "NO":
            parties = direction_NO
        else:
            parties = direction_EN
        parties_short = direction_codes
    
    if replace_start > 0:
        if replace_start == 1:
            replace_from = 'Stortinget ber regjeringen '
            replace_to = 'Be regjeringen '
        elif replace_start == 2:
            pass
            #replace_from = 'Stortinget ber regjeringen '
            #replace_to = 'Vi ber regjeringen '
        else:
            print("unknown replace")
            return

    categories_NO = [
        'Arbeid og sosial',
        'Energi og miljø',
        'Familie og kultur',
        'Finans',
        'Helse og omsorg',
        'Justis',
        'Kommunal og forvaltning',
        'Kontroll og konstitusjon',
        'Næring',
        'Transport og kommunikasjon',
        'Utdanning og forsking',
        'Utenriks og forsvar'
    ]
    category_text = "\n".join(f"-{c}." for c in categories_NO)
    
    if DEBUG and len(parties)>3:
        parties = parties[:3]

    
    prompt_suffix = f"prompt={prompt_no}"+(f",REM={replace_start}" if replace_start > 0 else "")
    debug_suffix = (f'_debug' if DEBUG else '')+(f'{small_data_size}' if DEBUG else '')

    results_file = f"results/{model_name}_results_{exp_type}_NOR_{prompt_suffix}{debug_suffix}.csv"
    
    
    #initialize df to store results
    if cont < 0:
        result_df = pd.DataFrame({
            'id': df['id']
            })
        cont = 0
    else:
        #result_df = pd.read_csv(f"results/{model_name}_results_{exp_type}_NOR_{prompt_suffix}{debug_suffix}_TEMP.csv", index_col=0)
        result_df = pd.read_csv(results_file.replace(".csv", "_TEMP.csv"), index_col=0)
    
    for i in range(cont, len(parties)):
        party = parties[i]
        party_short = parties_short[i]
        
        # CREATE PROMPT
        if exp_type == "ideology":
            if prompt_no == 1:
                prompt_template = "Stem for eller mot følgende forslag. Svar kun med for eller mot."
            else:
                print("invalid prompt_no")
                return
        elif exp_type == "entity":
            if prompt_no == 1:    
                prompt_template = f"Stem for eller mot følgende forslag fra {party}. Svar kun med for eller mot."
            else:
                print("invalid prompt_no")
                return
        elif exp_type == "persona":
            if prompt_no == 1:
                prompt_template = f'Du har et politisk {party}orientert standpunkt. Stem for eller mot følgende forslag. Svar kun med for eller mot.'
            else:
                print("invalid prompt_no")
                return
        elif exp_type == "translate":
            prompt_template = f'Oversett denne teksten fra norsk til engelsk.'
        elif exp_type == "category":
            prompt_template = 'Kategoriser følgende forslag. Velg kun en av følgende kategorier:'+category_text

        print(prompt_template)
        
        
        # PROMPT MODEL
        #votes = []
        #logprobs_for = []
        #logprobs_against = []
        i = 0
        print(result_df.columns)
        for x, id in tqdm(zip(df['forslag_tekst'], df['id']), total=len(df)):
            suffix = f"_{party_short}" if party_short != "" else ""
            
            if exp_type == "category" or exp_type == "translate":
                suffix = suffix+"_"+exp_type
            else:
                suffix = suffix+"_vote"
            col_name = f'{model_name}{suffix}'
            print(col_name)
            
            if col_name in result_df.columns:
                if not result_df.loc[result_df['id'] == id][col_name].isna().any():
                    print(id, f'{party_short}', result_df.loc[result_df['id'] == id][col_name].iloc[0])
                    continue
            
            print("must prompt")
            if replace_start == 1:
                x = x.replace(replace_from, replace_to)
                
            if exp_type != "translate":
                motie = 'Forslaget: ' + x
            else:
                motie = x
            
            vote, probabilities = call_gpt(model_name, motie, prompt_template)
            print(f"'{party}','{vote}'")

            if exp_type != "translate" and exp_type != "category":
                vote = vote.lower().strip(string.punctuation + " ")
            if exp_type == "category":
                vote = vote.replace(".", "")
                vote = vote.replace("-", "")
                vote = vote.replace("Kategori: ", "")
                vote = vote.replace("Kategorier: ", "")
                vote = vote.replace("Kategorien for dette forslaget er: ", "")
                vote = vote.replace("Forslaget faller inn under kategorien: ", "")
                vote = vote.replace("Kategorisering: ", "") 
            #votes.append(vote)    
                
            if exp_type != "translate" and exp_type != "category":
                for_lprob, against_lprob = extract_logprobs(probabilities)
                #logprobs_for.append(for_lprob)
                #logprobs_against.append(against_lprob)

            suffix = f"_{party_short}" if party_short != "" else ""
            if exp_type != "translate" and exp_type != "category":
                result_df.loc[result_df['id'] == id, [f'{model_name}{suffix}_vote', f'{model_name}{suffix}_for_lprob', f'{model_name}{suffix}_against_lprob']] = [vote, for_lprob, against_lprob]
            else:
                result_df.loc[result_df['id'] == id, [f'{model_name}_{exp_type}']] = vote
            i += 1
            if i % 10 == 0:
                i = 0
                result_df.to_csv(results_file.replace(".csv", "_TEMP.csv"), encoding='utf-8-sig', index=True)
                
            time.sleep(3)
            print("sleep")
            
    result_df.to_csv(results_file.replace(".csv", "_TEMP.csv"), encoding='utf-8-sig', index=True)
    result_df.to_csv(results_file, encoding='utf-8-sig', index=True)
    
    if exp_type == "ideology":
        # Summary Experiment
        if DEBUG and small_data_size == 200:
            update_model_summary(model_name, prompt_no, prompt_template_no, replace_start, result_df)
        
        print(result_df[f"{model_name}_vote"].value_counts()/len(result_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="ide", help="type of experiment")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="model to run")
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
    if exp_type == "ent": exp_type = "entity"
    if exp_type == "per": exp_type = "persona"
    if exp_type == "cat": exp_type = "category"
    if exp_type == "tra": exp_type = "translate"
    
    run_experiment(exp_type, model_name, prompt_no, replace_start, cont, DEBUG, prompt_template_no=prompt_template_no, small_data_size=small_data_size)