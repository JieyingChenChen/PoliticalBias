

import pandas as pd
import re
import csv
import string


SIGNS = ",;:.!?–—-"

# Remove all words before first verb
def prune_and_record(tokens, verbs):
    
    for i, w in enumerate(tokens):
        # remove any trailing punctuation from this token
        w_clean = w.rstrip(SIGNS)
        if w_clean in verbs:
            # keep from the cleaned verb onward
            kept = [w_clean] + tokens[i+1:]
            dropped = tokens[:i]
            return kept, dropped
    # no verb found: drop all tokens, record entire tail
    return [], tokens


def create_imperative(df):
    # 1) Load your verb list
    verbs_df = pd.read_csv("annotations/verb_list.csv")
    verb_set = set(verbs_df["word"].str.strip())
    
    correct_start = "Stortinget ber regjeringen"
    
    # Remove "Stortinget ber regjeringen"
    df['forslag_tekst_modified'] = df['forslag_tekst'].str[len(correct_start):].str.lstrip()
    
    # Mark OK if first word in known list 
    df['first_word'] = df['forslag_tekst_modified'].str.split().str.get(0)
    df['ok'] = df['first_word'].isin(verb_set)
    

    # for not ok rows, split into words, prune until first known good start. 
    mask = ~df['ok']
    results = df.loc[mask, 'forslag_tekst_modified'].str.split().apply(lambda toks: prune_and_record(toks, verb_set))
    
    # Split keep and drop tokens
    kept_series = results.map(lambda x: x[0])
    dropped_series = results.map(lambda x: x[1])
    df.loc[mask, 'kept_tokens']  = kept_series
    df.loc[mask, 'dropped_words'] = dropped_series
    
    # Rebuild
    df.loc[mask, 'forslag_tekst_modified'] = df.loc[mask, 'kept_tokens'].apply(
        lambda toks: " ".join(toks) if isinstance(toks, list) else ""
    )
   
    # Check how many empty (no good start in motion)   
    empty_count = df['forslag_tekst_modified'].isna().sum() + (df['forslag_tekst_modified'] == "").sum()
    print(f"Rows with empty/NaN pruned: {empty_count}")
    
    #Uppercase first word
    df['forslag_tekst_modified'] = df['forslag_tekst_modified'].str.replace(r'^(.)', lambda m: m.group(1).upper(), regex=True)
    
    # Remove signs after first word
    df['forslag_tekst_modified'] = df['forslag_tekst_modified'].str.replace(
        rf'^(\S+)[{re.escape(SIGNS)}]+',
        r'\1 ',
        regex=True
    )

    # replace Å ikke med Ikke å
    df['forslag_tekst_modified'] = df['forslag_tekst_modified'].str.replace(r'^Ikke å', 'Å ikke', regex=True)

    # DEBUG check dropped words, see if any can be good starts ...
    #dropped_lists = df.loc[mask, 'dropped_words'].dropna().tolist()
    #all_dropped   = [w for sub in dropped_lists for w in sub]
    #unique_dropped = sorted(set(all_dropped))
    #print("Unique dropped tokens:", unique_dropped)
    
    
    df.drop(columns=['first_word', 'ok', 'kept_tokens', 'dropped_words'], inplace=True, errors='ignore')
    return df



def run_experiment():
    df = pd.read_csv("data/all_motions_2018-2023.csv")
    df = create_imperative(df)
    df.to_csv("data/all_motions_2018-2023_pros.csv", encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    run_experiment()

