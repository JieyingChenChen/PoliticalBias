import pandas as pd
import os
import numpy as np
import pandas_read_xml as pdx

from io import StringIO
from html.parser import HTMLParser

import time
import re

import csv

import argparse


output_folder = "data/"

RESULTS_LATEX_FILE = 'results_latex/download_results.tex'
RESULTS_LATEX_FILE_DEBUG = 'results_latex/download_results_DEBUG.tex'

DATA_URL = "https://data.stortinget.no/eksport/"

party_codes = ['A', 'H', 'Sp', 'FrP', 'KrF', 'MDG', 'SV', 'V', 'R', 'PF']

party_names = ['Arbeiderpartiet',
            'Høyre',
            'Senterpartiet',
            'Fremskrittspartiet',
            'Kristelig Folkeparti',
            'Miljøpartiet De Grønne',
            'Sosialistisk Venstreparti',
            'Venstre',
            'Rødt',
            'Pasientfokus']


komite_dict_NO = {
    'FINANS': 'Finans',
    'HELSEOMS': 'Helse og omsorg',
    'KOMMFORV': 'Kommunal og forvaltning',
    'FAMKULT': 'Familie og kultur',
    'ENERGI': 'Energi og miljø',
    'UFO': 'Utdanning og forsking',
    'NÆRING': 'Næring',
    'ARBSOS': 'Arbeid og sosial',
    'JUSTIS': 'Justis',
    'TRANSKOM': 'Transport og kommunikasjon',
    'UFK': 'Utenriks og forsvar',
    'KONTROLL': 'Kontroll og konstitusjon'
}

komite_dict_EN = {
    'FINANS': 'Finance',
    'HELSEOMS': 'Health and care',
    'KOMMFORV': 'Municipal and administrative',
    'FAMKULT': 'Family and culture',
    'ENERGI': 'Energy and environmental',
    'UFO': 'Education and research',
    'NÆRING': 'Business',
    'ARBSOS': 'Labor and social',
    'JUSTIS': 'Justice',
    'TRANSKOM': 'Transportation and communication',
    'FULLMAKT': 'Proxy',
    'PRES': 'Presidency of the Storting',
    'VALG': 'Election',
    'UFK': 'Foreign Affairs and Defence',
    'KONTROLL': 'Control and Constitution'
}

party_map = { name: code for name, code in zip(party_names, party_codes) }

# Ny Norsk party names to bokmål
partynames_NN_to_NO = [["Raudt", "Rødt"],
             ["Miljøpartiet Dei Grøne", "Miljøpartiet De Grønne"],
             ["Kristeleg Folkeparti", "Kristelig Folkeparti"],
             ["Høgre", "Høyre"],
             ["Framstegspartiet", "Fremskrittspartiet"]   
            ]

# Max retries to download data
MAX_RETRIES = 5
DELAY = 3
DEBUG_DELAY = 2

DEBUG_SAK_LEN = 20
DEBUG_VOT_LEN = 10




def write_to_latex(results_latex_file, latex_text):
    # Write all macros to a file (e.g., tables_macros.tex).
    print("write to latex")
    print(latex_text)
    with open(results_latex_file, "a") as f_out:
        f_out.write(latex_text+"\n\n")


def write_variable_to_latex(results_latex_file, var_name, var_value):
    latex_text = f"\\newcommand{{\\{var_name}}}{{{var_value}}}\ignorespaces\n"
    write_to_latex(results_latex_file, latex_text)





def download_saker(no_requests, period):
    retries = 0
    url = "https://data.stortinget.no/eksport/saker?sesjonid="+period
    df_saker_single = None
    delay = 2
    while retries < MAX_RETRIES:
        try:
            df_saker_single = pdx.read_xml(url,
                      ['saker_oversikt','saker_liste', 'sak'],
                      root_is_rows=False)
            no_requests += 1
            time.sleep(delay)
            break  # Exit the retry loop if successful
        except Exception as e:
            print(f"Error processing saker, attempt {retries+1} of {MAX_RETRIES}:")
            retries += 1
            time.sleep(delay)  # Wait before trying again
    return no_requests, df_saker_single





def download_proces_saker(no_requests, periods_saker, DEBUG):
    ## DOWNLOAD
    df_saker = pd.DataFrame()
    for p in periods_saker:
        no_requests, df_saker_single = download_saker(no_requests, p) 
        
        #if df_saker.empty:
        #    df_saker = df_saker_single
        #else:
        #    df_saker= pd.concat([df_saker, df_saker_single], ignore_index=True)
        
        df_saker= pd.concat([df_saker, df_saker_single], ignore_index=True)
    
    
    ## CLEAN
    df_saker.rename(columns={'id': 'sak_id'}, inplace=True)
    
    df_saker.drop_duplicates(subset='sak_id', keep='first', inplace=True)
    
    df_saker.drop(['henvisning',
                         'innstilling_id',
                         'innstilling_kode',
                         'saksordfoerer_liste',
                         'sist_oppdatert_dato',
                         'forslagstiller_liste'], axis=1, inplace=True, errors='ignore')
    
    # Extract 'navn' safely even if 'info' or 'emne' is None
    df_saker['emne_liste'] = df_saker['emne_liste'].apply(
        lambda x: [item.get('navn') for item in x.get('emne', []) if item and 'navn' in item]
        if isinstance(x, dict) and isinstance(x.get('emne'), list) else None
        #if isinstance(x, dict) and isinstance(x.get('emne'), list) else x
    )
    
    # Extract komite navn and Id
    df_saker['komite'] = df_saker['komite'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
    # Translate komite to kategori
    df_saker['category'] = df_saker['komite'].map(komite_dict_EN)
    
    df_saker['behandlet_sesjon_id'] = df_saker['behandlet_sesjon_id'].apply(lambda x: np.nan if isinstance(x, dict) else x)
    
    # Sort
    df_saker.sort_values(by='sak_id', inplace=True)
    
    # Convert list to string list 
    df_saker['emne_liste'] = df_saker['emne_liste'].apply(str)
    
    df_saker.drop_duplicates(inplace=True)
    
    ## Diagnosis
    print(df_saker[['komite', 'category']].drop_duplicates())
    print(df_saker[['sak_id','korttittel','sak_fremmet_id']].duplicated())
    print(df_saker.applymap(type).apply(lambda col: col.unique(), axis=0))
    print(len(df_saker))
    
    ### SAVE
    period = periods_saker[0].split("-")[0]+'-'+periods_saker[-1].split("-")[1]
    
    suf = '_DEBUG' if DEBUG else ''
    #df_saker.to_csv(f"{output_folder}saker_{period}{suf}.csv", encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)
    return no_requests, df_saker



# ## Stortingsvedtak



def download_stortingsvedtak(no_requests, period, DEBUG):
    retries = 0
    url = "https://data.stortinget.no/eksport/stortingsvedtak?sesjonid="+period
    df_stortingsvedtak = None
    delay = DEBUG_DELAY if DEBUG else DELAY
    
    while retries < MAX_RETRIES:
        try:
            df_stortingsvedtak = pdx.read_xml(url,
                              ['stortingsvedtak_oversikt','stortingsvedtak_liste','stortingsvedtak'],
                              root_is_rows=False)
            no_requests += 1
            time.sleep(delay)
            break  # Exit the retry loop if successful
        except Exception as e:
            print(f"Error processing saker, attempt {retries+1} of {MAX_RETRIES}:")
            retries += 1
            time.sleep(delay)  # Wait before trying again
    return no_requests, df_stortingsvedtak


# ## Voteringer




def download_process_stortingsvedtak(no_requests, period, DEBUG):
    no_requests, df_stortingsvedtak = download_stortingsvedtak(no_requests, period, DEBUG)
    
    ## CLEAN
    df_stortingsvedtak['stortingsvedtak_tekst'] = df_stortingsvedtak['stortingsvedtak_tekst'].str.replace(r'<[^<>]*>', '', regex=True).replace(r'\s+', ' ', regex=True).replace('\n',' ').str.strip()
    
    # convert dicts to string
    col_list = ['stortingsvedtak_type']
    for col in col_list:
        df_stortingsvedtak[col] = df_stortingsvedtak[col].apply(str)
    
    vedtak_typer = [
        ("{'id': 'ANMOD', 'navn': 'Anmodninger'}", 'Anmodninger'),
        ("{'id': 'RO', 'navn': 'Henstillinger'}", 'Henstillinger'),
        ("{'id': 'BUDSJETT', 'navn': 'Budsjettvedtak'}", 'Budsjettvedtak'),
        ("{'id': 'GRUNNLOV', 'navn': 'Grunnlovsvedtak'}", 'Grunnlovsvedtak'),
        ("{'id': 'ANNET', 'navn': 'Allminnelige vedtak'}", 'Allminnelige vedtak')
    ]
    
    for i in range(len(vedtak_typer)):
        df_stortingsvedtak['stortingsvedtak_type'].replace({vedtak_typer[i][0]: vedtak_typer[i][1]}, inplace=True)
    
    # drop long vedtak (remove?)
    print("len(df_stortingsvedtak) Before drop long:", len(df_stortingsvedtak))
    df_stortingsvedtak.drop(df_stortingsvedtak[df_stortingsvedtak['stortingsvedtak_tekst'].str.len() >3000].index, inplace=True)
    print("len(df_stortingsvedtak) After drop long:", len(df_stortingsvedtak))
    
    #sak_nos = df_stortingsvedtak['sak_id'].unique()
    #print(len(sak_nos))
    
    ## DIAGNOSIS
    df = pd.DataFrame()
    df['test'] = df_stortingsvedtak['stortingsvedtak_tekst'].str.contains('\t', na=False)
    print(df[df['test'] == True])
    
    counts = df_stortingsvedtak['sak_id'].value_counts()
    print(counts)
    
    print(df_stortingsvedtak.head())
    
    
    # Duplicates
    print(df_stortingsvedtak[df_stortingsvedtak.duplicated()])
    
    ## SAVE
    suf = '_DEBUG' if DEBUG else ''
    #df_stortingsvedtak.to_csv(f"{output_folder}stortingsvedtak_{period}{suf}.csv", encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)
    return no_requests, df_stortingsvedtak







def download_voteringer_single(no_requests, sak_id, DEBUG):
    retries = 0
    url = 'https://data.stortinget.no/eksport/voteringer?sakid='+str(sak_id)
    df_voteringer_single = None
    cols = ['sak_votering_oversikt','sak_votering_liste','sak_votering']
    delay = DEBUG_DELAY if DEBUG else DELAY
    
    while retries < MAX_RETRIES:
        try:
            df_voteringer_single = pdx.read_xml(url, cols, root_is_rows=False)
            no_requests += 1
            # if this format does not look good try:
            if df_voteringer_single.index[0] == '@i:nil':
                time.sleep(delay)
                df_voteringer_single = pdx.read_xml(url, cols, root_is_rows=True)
                no_requests += 1
            if len(df_voteringer_single) == 0:
                print(f"Empty: {sak_id}. ")
            time.sleep(delay)
            break  # Exit the retry loop if successful
        except Exception as e:
            print(f"Saker, attempt {retries+1} of {MAX_RETRIES}: {str(e)}")
            
            if "NoneType" in str(e):
                retries = MAX_RETRIES
            else:
                if retries == MAX_RETRIES:
                    print(url)
                retries += 1
                time.sleep(delay)  # Wait before trying again
    
    return no_requests, df_voteringer_single





def download_process_voteringer(no_requests, sak_nos, period, DEBUG):
    df_voteringer = pd.DataFrame()
    
    sak_nos = sak_nos[:DEBUG_SAK_LEN] if DEBUG else sak_nos
    #sak_nos = [95322]
    for sak_id in sak_nos:
        no_requests, df_voteringer_single = download_voteringer_single(no_requests, sak_id, DEBUG)
        print(type(df_voteringer_single))

        
        df_voteringer = pd.concat([df_voteringer, df_voteringer_single], ignore_index=True)
    

    df_voteringer.drop(['president', 'respons_dato_tid', 'versjon'], axis=1, inplace=True, errors='ignore')
    
    # convert dicts to string
    col_list = ['kommentar', 'votering_resultat_type_tekst']
    for col in col_list:
        df_voteringer[col] = df_voteringer[col].apply(str)
    
    
    # Replace strange strings
    df_voteringer.replace({"{'@i:nil': 'true'}": ''}, inplace=True)
    
    # Remove columns
    df_voteringer = df_voteringer[['sak_id','votering_id','votering_tema','votering_tid','antall_for','antall_ikke_tilstede','antall_mot','vedtatt','kommentar']]
    
    ## DIAGNOSIS
    print(df_voteringer[df_voteringer.duplicated()])
    
    len(df_voteringer)
    
    counts=df_voteringer['sak_id'].value_counts()
    counts
    
    #print(len(df_stortingsvedtak))
    print(len(df_voteringer))
    
    ## SAVE
    suf = '_DEBUG' if DEBUG else ''
    #df_voteringer.to_csv(f'{output_folder}voteringer_{period}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)
    return no_requests, df_voteringer






def extract_numbers(text):
    # Extract all numbers from the string (e.g., '3', '5', '6' from 'Forslag nr. 3, 5 og 6')
    return re.findall(r'\d+', text)


def extract_number_or_blank(text):
    if not text:  # Handle None, empty string, etc.
        return ""
    
    matches = re.findall(r'\d+', text)
    if matches:
        return matches[0]  # The single number if found
    else:
        return 0



def match_numbers(row, df1):
    # Check if the number in df2 exists in any of the lists in df1
    matching_rows = df1[df1['numbers_list'].apply(lambda x: row['single_number'] in x)]
    return matching_rows



def is_in_list(single_number, numbers_list):
    return single_number in numbers_list

def is_in_list_and_exact_match(single_number, numbers_list, exact_value1, exact_value2):
    return single_number in numbers_list and exact_value1 == exact_value2

def map_multiple_parties_to_codes(party_text):
    """
    Takes a string of comma-separated party names (e.g. "Arbeiderpartiet, Senterpartiet")
    and returns a string like ['A','Sp'] based on the party_map lookup.
    """
    if not party_text or pd.isnull(party_text):
        # Handle null or empty strings — return an empty list representation
        return "[]"
    
    
    parties = [p.strip() for p in party_text.split(",")]
    
    # Map each party to its code
    #codes = [party_map.get(p, p) for p in parties] # keep orig if not in list 
    codes = [party_map[p] for p in parties if p in party_map] # remove if not in list
    
    # Build a string like ['A','Sp']
    # Note: we add single quotes inside the list to match your requested format
    codes_str = ",".join(f"'{c}'" for c in codes)
    return f"[{codes_str}]"

def map_multiple_parties_to_codes(party_text):
    """
    Takes a string of comma-separated party names (e.g. "Arbeiderpartiet, Senterpartiet")
    and returns a string like ['A','Sp'] based on the party_map lookup.
    Only valid values (i.e. those present in party_map) are included.
    """
    if not party_text or pd.isnull(party_text):
        # Handle null or empty strings — return an empty list representation
        return "[]"
    
    # Split by comma and strip extra spaces
    parties = [p.strip() for p in party_text.split(",")]
    
    # Only include a party if it exists in party_map
    codes = [party_map[p] for p in parties if p in party_map]
    
    # Build a string like ['A','Sp']
    codes_str = ",".join(f"'{c}'" for c in codes)
    return f"[{codes_str}]"


# more complex cases, refering to laws
def should_drop(text):
    words = text.split()  # Split the text into words
    # Check the first two words (if they exist)
    for word in words[:2]:
        if word.lower().endswith("loven") or word.lower().endswith("lova"):
            return True
    return False






def download_voteringsforslag_single(no_requests, votering_id, root_is_rows, DEBUG):
    retries = 0
    df_voteringsforslag_single = None
    url = DATA_URL + "voteringsforslag?voteringid=" + str(votering_id)
    cols = ['voteringsforslag_oversikt','voteringsforslag_liste','voteringsforslag']
    delay = DEBUG_DELAY if DEBUG else DELAY
    
    while retries < MAX_RETRIES:
        try:
            df_voteringsforslag_single = pdx.read_xml(url, cols, root_is_rows=root_is_rows)
            no_requests += 1
            break  # Exit the retry loop if successful
        except Exception as e:
            print(f"Voteringsforslag votering_id {votering_id}, attempt {retries+1} of {MAX_RETRIES}: {str(e)}")
            if "NoneType" in str(e):
                retries = MAX_RETRIES    
            else:
                retries += 1
                time.sleep(delay)  # Wait before trying again
                if retries == MAX_RETRIES:
                    print(url)
                    time.sleep(delay)
                    
    return no_requests, df_voteringsforslag_single





def download_voteringsresultat_single(no_requests, votering_id, DEBUG):
    retries = 0
    df_voteringsresultat_single = None
    url = DATA_URL+"voteringsresultat?voteringid="+votering_id
    cols = ['voteringsresultat_oversikt', 'voteringsresultat_liste', 'representant_voteringsresultat']
    delay = DEBUG_DELAY if DEBUG else DELAY
    
    while retries < MAX_RETRIES:
        try:
            df_voteringsresultat_single = pdx.read_xml(url, cols, root_is_rows=False)
            no_requests += 1
            break  # exit retry loop if successful
        except Exception as e:
            print(f"Voteringsresultat votering_id {votering_id}, attempt {retries+1} of {MAX_RETRIES}")
            if "NoneType" in str(e):
                print("Nonetype")
                retries = MAX_RETRIES
            else:
                retries += 1
                time.sleep(delay)  # Wait before trying again
                if retries == MAX_RETRIES:
                    print(url)
                    time.sleep(delay)
            
    return no_requests, df_voteringsresultat_single



def download_process_voteringsforslag_voteringsresultat(no_requests, no_dropped_motions, votering_ids, period, DEBUG):
    df_voteringsforslag = pd.DataFrame()
    df_voteringsresultat = pd.DataFrame()
    
    votering_ids = votering_ids[:DEBUG_VOT_LEN] if DEBUG else votering_ids
    
    for votering_id in votering_ids: 
        no_requests, df_voteringsforslag_single = download_voteringsforslag_single(no_requests, votering_id, True, DEBUG)
           
        if not isinstance(df_voteringsforslag_single, pd.DataFrame) or df_voteringsforslag_single.shape[1] <= 8:
            print(f"{votering_id}try again")
            no_requests, df_voteringsforslag_single = download_voteringsforslag_single(no_requests, votering_id, False, DEBUG)
    
        if not isinstance(df_voteringsforslag_single, pd.DataFrame):
            continue
        
        df_voteringsforslag_single['votering_id'] = votering_id
        
        #if df_voteringsforslag.empty:
        #    df_voteringsforslag = df_voteringsforslag_single
        #else:
        #    df_voteringsforslag = pd.concat([df_voteringsforslag, df_voteringsforslag_single], ignore_index=True)
        df_voteringsforslag = pd.concat([df_voteringsforslag, df_voteringsforslag_single], ignore_index=True)
        
        # *******************************
        # **     VOTERINGSRESULTAT     **
        # *******************************
        no_requests, df_voteringsresultat_single = download_voteringsresultat_single(no_requests, votering_id, DEBUG)
        
        if not isinstance(df_voteringsresultat_single, pd.DataFrame):
            continue
            
        df_voteringsresultat_single['votering_id'] = votering_id
        df_voteringsresultat_single['parti_id'] = df_voteringsresultat_single['representant'].apply(
            lambda x: x.get("parti", {}).get("id")
        )
        
        df_voteringsresultat_single.drop(['respons_dato_tid','versjon','fast_vara_for','representant','vara_for'], axis=1, inplace=True)
        
        #print(df_voteringsresultat_single)
        
        # Group by votering_id and parti_id, and count the occurrences of 'for' and 'mot'
        df_voteringsresultat_single = (
            df_voteringsresultat_single.groupby(['votering_id', 'parti_id', 'votering'])
            .size()
            .unstack(fill_value=0)  # Create columns for 'for' and 'mot'
            .reset_index()          # Reset the index to make votering_id and parti_id columns again
        )
        for col in ['for', 'mot']:
            if col not in df_voteringsresultat_single.columns:
                df_voteringsresultat_single[col] = 0  # Set default value
        
        df_voteringsresultat_single['vote'] = df_voteringsresultat_single.apply(
            lambda row: 1 if row['for'] > row['mot'] else (-1 if row['mot'] > row['for'] else 0), axis=1
        )
        
        df_voteringsresultat_single.drop(['ikke_tilstede'], axis=1, inplace=True, errors='ignore')
        
        # Add rows for Parties that did not vote
        missing_parti_ids = set(party_codes) - set(df_voteringsresultat_single['parti_id'])
        
        missing_rows = [
            {'votering_id': df_voteringsresultat_single['votering_id'].iloc[0], 'parti_id': parti_id, 'for': 0, 'mot': 0, 'vote': 0}
            for parti_id in missing_parti_ids
        ]
        
        # Append the missing rows to the DataFrame
        df_voteringsresultat_single = pd.concat([df_voteringsresultat_single, pd.DataFrame(missing_rows)], ignore_index=True)
        
        df_voteringsresultat_single = df_voteringsresultat_single.pivot(index='votering_id', columns='parti_id', values=['for', 'mot', 'vote'])
        
        df_voteringsresultat_single.columns = [f"{col[1]}_{col[0]}" for col in df_voteringsresultat_single.columns]
        
        # Reset the index to make `votering_id` a column again
        try: 
            df_voteringsresultat_single.reset_index(inplace=True)
        except Exception as e:
            print(df_voteringsresultat_single)
            print(e)
        
        #if df_voteringsresultat.empty:
        #    df_voteringsresultat = df_voteringsresultat_single
        #else:
        #    df_voteringsresultat = pd.concat([df_voteringsresultat, df_voteringsresultat_single], ignore_index=True)
        
        df_voteringsresultat = pd.concat([df_voteringsresultat, df_voteringsresultat_single], ignore_index=True)

    ## CLEAN VOTERINGSFORSLAG
    # 1. Drop columns
    print(df_voteringsforslag.columns)
    
    df_voteringsforslag = df_voteringsforslag[['votering_id','forslag_betegnelse','forslag_betegnelse_kort','forslag_id','forslag_paa_vegne_av_tekst','forslag_tekst']]
    
    # Clean HTML
    df_voteringsforslag['forslag_tekst'] = df_voteringsforslag['forslag_tekst'].str.replace(r'<[^<>]*>', '', regex=True).replace(r'\s+', ' ', regex=True).replace('\n',' ').str.strip()
    
    
    df_voteringsforslag.dropna(subset=['forslag_tekst'], inplace=True)
    # drop 
    
    df_voteringsforslag.rename(columns={'forslag_betegnelse_kort': 'votering_tema_forslag'}, inplace=True)
    
    df_voteringsforslag['votering_tema_forslag'] = 'Forslag '+df_voteringsforslag['votering_tema_forslag'].astype(str)
    
    df_voteringsforslag['forslag_paa_vegne_av_tekst'] = df_voteringsforslag['forslag_paa_vegne_av_tekst'].astype(str)
    
    df_voteringsforslag["forslag_paa_vegne_av_tekst"] = df_voteringsforslag["forslag_paa_vegne_av_tekst"].str.replace(":", "", regex=False)
    
    df_voteringsforslag.replace({"{'@inil' 'true'}": ''}, inplace=True)
    
    df_voteringsforslag.replace({"{'@i:nil': 'true'}": ''}, inplace=True)
    
    # drop any rows with empty forslag
    df_voteringsforslag = df_voteringsforslag[df_voteringsforslag["forslag_tekst"].str.strip() != ""].dropna(subset=["forslag_tekst"])

    # Extract number from text
    df_voteringsforslag['number'] = df_voteringsforslag['votering_tema_forslag'].apply(extract_number_or_blank).astype(int)
    # Update 'number' where it's 0, with an incremental count within each 'id' group
    df_voteringsforslag.loc[df_voteringsforslag['number'] == 0, 'number'] = df_voteringsforslag[df_voteringsforslag['number'] == 0].groupby('votering_id').cumcount()

    # Create Id (after drop in case of duplicates between good and bad motions)
    df_voteringsforslag['id'] = (df_voteringsforslag['votering_id'].astype(str) + df_voteringsforslag['number'].astype(int).astype(str)).astype(int)

    # ***** DROP *****
    
    # LONG MOTIONS
    no_before_drop = len(df_voteringsforslag)
    print("Before delete long motions:", no_before_drop)
    
    # Save them for transparency, TODO set on a common format
    df_voteringsforslag_long = df_voteringsforslag[df_voteringsforslag['forslag_tekst'].apply(lambda x: len(str(x).split()) > 500)]
    df_voteringsforslag_long = df_voteringsforslag_long[['id', 'forslag_tekst']]
    df_voteringsforslag_long['drop_reason'] = "Long"
    
    df_voteringsforslag = df_voteringsforslag[df_voteringsforslag['forslag_tekst'].apply(lambda x: len(str(x).split()) <= 500)]
    no_dropped = no_before_drop-len(df_voteringsforslag)
    no_dropped_motions[0] += no_dropped
    print("No dropped long motions:",  no_dropped)
    

    # TODO: Fix typos and variations of "Stortinget ber regjeringen ..."
    correct_start = "Stortinget ber regjeringen"
    if False:
        
        from_starts = [
            "Stortinget ber regjeringens",   # longest first
            "Stortinget ber Regjeringen",
            "Stortinget ber regjeringa",
            "Stortinget ber egjeringen",
            "Stortinget ber regjerningen",
            "Stortinget ber regjering",      # shortest last
        ]
        # join them into an alternation, escape if needed
        from_pattern = "|".join(from_starts)
        
        # fix bad starts
        pattern = re.compile(rf"^(?:{from_pattern})(.*)$")
        df_voteringsforslag['forslag_tekst'] = df_voteringsforslag['forslag_tekst'].str.replace(
            pattern,
            correct_start+r"\1",
            regex=True
        )

    correct_start = "Stortinget ber regjeringen"
    
    pattern = re.compile(
        r"^(Stortinget ber)\s+"
        r"(?:"
           r"regjering(?:a|ens|en)?|"   # regjering, regjeringa, regjeringens, regjeringen
           r"egjering(?:a|ens|en)?|"     # typo variant missing leading 'r'
           r"regjerning(?:a|ens|en)?"    # typo variant with swapped 'n'
        r")\b(.*)$",
        flags=re.IGNORECASE
    )
    
    df_voteringsforslag['forslag_tekst'] = (
        df_voteringsforslag['forslag_tekst']
          .str.replace(pattern,
                       r"\1 regjeringen\2",   # \1 = "Stortinget ber", \2 = the rest
                       regex=True)
    )

    
    # drop all motions not starting with "Stortinget ber"
    # Keep for manual check
    df_voteringsforslag_wrong_format = df_voteringsforslag[~df_voteringsforslag['forslag_tekst'].str.startswith(correct_start)].copy()
    # But only the cols needed to check 
    df_voteringsforslag_wrong_format = df_voteringsforslag_wrong_format[['id', 'forslag_tekst']]
    df_voteringsforslag_wrong_format['drop_reason'] = "Wrong format"

    no_before_drop = len(df_voteringsforslag)
    df_voteringsforslag = df_voteringsforslag[df_voteringsforslag['forslag_tekst'].str.startswith(correct_start)].copy()
    
    no_dropped = no_before_drop-len(df_voteringsforslag)
    no_dropped_motions[1] += no_dropped
    print("No dropped ber:",  no_dropped)
    
    
    print("before drop duplicates:", len(df_voteringsforslag))
    df_voteringsforslag.drop_duplicates(subset='id', keep='first', inplace=True)
    print("after drop duplicates:", len(df_voteringsforslag))
    
    # CLEAN VOTERINGSFORSLAG 2
    
    # replace 'og' ('and') with ',' 
    df_voteringsforslag["forslag_paa_vegne_av_tekst"]= df_voteringsforslag["forslag_paa_vegne_av_tekst"].str.replace(" og ", ", ", regex=False)
      
    for p in partynames_NN_to_NO:
        df_voteringsforslag["forslag_paa_vegne_av_tekst"]= df_voteringsforslag["forslag_paa_vegne_av_tekst"].str.replace(p[0], p[1], regex=False)    
    
    # Map list of proposing parties names to parties ids 
    
    df_voteringsforslag["forslag_partier"] = df_voteringsforslag["forslag_paa_vegne_av_tekst"].apply(map_multiple_parties_to_codes)
    
    ### Clean Voterings Resultat
    for p in party_codes:
        df_voteringsresultat.drop([f'{p}_for'], axis=1, inplace=True)
        df_voteringsresultat.drop([f'{p}_mot'], axis=1, inplace=True)
    
    ## SAVE
    suf = '_DEBUG' if DEBUG else ''
    #df_voteringsresultat.to_csv(f'{output_folder}voteringsresultat_{period}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)
    #df_voteringsforslag.to_csv(f'{output_folder}voteringsforslag_{period}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)

    #df_voteringsforslag_long.to_csv(f'{output_folder}DROPPED_LONG_voteringsforslag_{period}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)
    #df_voteringsforslag_wrong_format.to_csv(f'{output_folder}DROPPED_WRONGFORMAT_voteringsforslag_{period}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)
    
    return no_requests, no_dropped_motions, df_voteringsforslag, df_voteringsresultat



# # Merge




# Manual votes
# --------------
# Function to extract voting results
def parse_votes(vote_text):
    for_match = re.search(r"FOR stemte \d+\s*\((.*?)\)", vote_text)
    mot_match = re.search(r"MOT stemte \d+\s*\((.*?)\)", vote_text)

    for_parties = for_match.group(1).split(",") if for_match else []
    mot_parties = mot_match.group(1).split(",") if mot_match else []

    return {party + "_vote": 1 if party in for_parties else -1 if party in mot_parties else 0 for party in party_codes}





def merge_dataset(no_dropped_motions, df_saker, df_voteringer, df_voteringsforslag, df_voteringsresultat, period, DEBUG):
    
    if len(df_voteringsresultat) > 0:
        df_voteringsforslag = pd.merge(df_voteringsforslag, df_voteringsresultat, on='votering_id', how='left')
    
    ### Voteringer and Votering Forslag -> All Votes
    all_votes = pd.merge(df_voteringer, df_voteringsforslag, on=['votering_id'])
    
    # Move to voteringer ??
    
    # Unanominos votes (does not have voteringsresultat)
    #cond = (all_votes['antall_for'] == -1) & (all_votes['vedtatt'] == 'TRUE')
    cond = (all_votes['antall_for'] == '-1') & (all_votes['vedtatt'] == 'true')
    all_votes.loc[cond, [f'{p}_vote' for p in party_codes]] = [1] * len(party_codes)

    #suf = '_DEBUG' if DEBUG else ''
    #all_votes.to_csv(f'all_motions_TEMP_{period}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)
    
    # Define result columns dynamically
    cols = [p + "_vote" for p in party_codes]
    
    # Apply parsing function and update the DataFrame
    all_votes['kommentar'] = all_votes['kommentar'].str.replace(r'\bSP\b', 'Sp', regex=True)
    parsed_votes = all_votes['kommentar'].apply(parse_votes).apply(pd.Series)
    
    # Only fill missing values
    for col in cols:
        all_votes[col] = all_votes[col].combine_first(parsed_votes[col])
    
    #all_votes[cols] = all_votes['kommentar'].apply(parse_votes).apply(pd.Series)
    
    #all_votes.drop(columns=['kommentar'], inplace=True)
    
    # Merge in Saker
    all_votes = pd.merge(all_votes, df_saker, on='sak_id', how='left')
    
    #all_votes.to_csv(f'all_motions_TEMP2_{period}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)
    
    print(all_votes.applymap(type).apply(lambda col: col.unique(), axis=0))
    #all_votes[all_votes.duplicated()]
    
    #print("before drop long Forslagstekst:", len(df))
    #df = df[df['forslag_tekst'].apply(lambda x: word_count(x) <= 1000)]
    
    all_votes.sort_values(by=['votering_tid', 'votering_id', 'number'], inplace=True)
    
    #all_votes.drop(['votering_tema_forslag','respons_dato_tid','versjon','votering_tema','forslag_paa_vegne_av_tekst'], axis=1, inplace=True)
    
    # Keep cols
    all_votes = all_votes[[
    'id',
    'sak_id',
    'sak_fremmet_id',
    'votering_id',
    'number',
    'forslag_id',
    'korttittel',
    'forslag_betegnelse',
    'forslag_partier',
    'forslag_tekst',
    'votering_tid',
    'antall_for',
    'antall_ikke_tilstede',
    'antall_mot',
    'vedtatt',
    'A_vote',
    'FrP_vote',
    'H_vote',
    'KrF_vote',
    'MDG_vote',
    'PF_vote',
    'R_vote',
    'SV_vote',
    'Sp_vote',
    'V_vote',
    'kommentar',
    'behandlet_sesjon_id',
    'dokumentgruppe',
    'emne_liste',
    'status',
    'tittel',
    'type',
    'komite',
    'category',  
    ]]

    # Delete categories
    

    del_cat = ['Presidency of the Storting', 'Control and Constitution', 'Other']

    # TODO: Save rows about to be dropped for transparancy
    all_votes_wrong_cat = all_votes[all_votes['category'].isin(del_cat)]
    all_votes_wrong_cat = all_votes_wrong_cat[['id','forslag_tekst','category']]
    
    no_before_drop = len(all_votes)
    print("before drop unpolitical categories:", no_before_drop)
    # DROP 
    all_votes = all_votes[~all_votes['category'].isin(del_cat)]
    
    no_dropped = no_before_drop-len(all_votes)
    print("no dropped unpolitical categories:", no_dropped)
    no_dropped_motions[2] += no_dropped

    # Set PF vote to 0 if they had no seat (for unanimous votes set to 1)
    if int(period.split('-')[0]) < 2021:
        all_votes['PF_vote'] = 0

    
    
    
    print("len(all_votes):", len(all_votes))
    # Save the complete dataset
    suf = '_DEBUG' if DEBUG else ''
    all_votes.to_csv(f'{output_folder}all_motions_{period}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)
    all_votes_wrong_cat.to_csv(f'{output_folder}DROPPED_WRONGCAT_all_motions_{period}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)
    
    return no_dropped_motions




def download_session(no_requests, no_dropped_motions, session_start, DEBUG):

    period = str(session_start)+"-"+str(session_start+1)
    
    # Need the previous period
    periods_saker = [str(session_start-1)+"-"+str(session_start)]+[period]
    
    no_requests, df_saker = download_proces_saker(no_requests, periods_saker, DEBUG)
    
    no_requests, df_stortingsvedtak = download_process_stortingsvedtak(no_requests, period, DEBUG)
    sak_nos = df_stortingsvedtak['sak_id'].unique()
    
    no_requests, df_voteringer = download_process_voteringer(no_requests, sak_nos, period, DEBUG)
    votering_ids = df_voteringer['votering_id'].unique()
    no_requests, no_dropped_motions, df_voteringsforslag, df_voteringsresultat = download_process_voteringsforslag_voteringsresultat(no_requests, no_dropped_motions, votering_ids, period, DEBUG)

    
    no_dropped_motions = merge_dataset(no_dropped_motions, df_saker, df_voteringer, df_voteringsforslag, df_voteringsresultat, period, DEBUG)

    return no_requests, no_dropped_motions


def merge_all_periods(session_starts, DEBUG):
    
    all_sessions = [f"{year}-{year+1}" for year in session_starts]
    print(all_sessions)
    
    dfs = []
    suf = '_DEBUG' if DEBUG else ''
    for p in all_sessions:
        file_name = f'{output_folder}all_motions_{p}{suf}.csv'
        if os.path.exists(file_name):
            dfs.append(pd.read_csv(file_name))
    
    df = pd.DataFrame()
    for i in range(len(dfs)):
        df = pd.concat([df, dfs[i]], ignore_index=True)
    
    print(df.columns)
    
    df.to_csv(f'{output_folder}all_motions_{session_starts[0]}-{session_starts[-1]}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)

    # Save motions with no category for annotation
    df_no_cat = df[df['category'].isna()]
    df_no_cat.to_csv(f'{output_folder}all_motions_no_cat_{session_starts[0]}-{session_starts[-1]}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)

    wrong_file_names = [
    "DROPPED_WRONGCAT_all_motions",
    "DROPPED_WRONGFORMAT_voteringsforslag",
    "DROPPED_LONG_voteringsforslag"]
    
    dfs = []
    for p in all_sessions:
        for wrong_file_name in wrong_file_names:
            
            file_name = f'{output_folder}{wrong_file_name}_{p}{suf}.csv'
            if os.path.exists(file_name):
                dfs.append(pd.read_csv(file_name))
                os.remove(file_name)
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(f'{output_folder}Dropped_{session_starts[0]}-{session_starts[-1]}{suf}.csv', encoding='utf-8-sig', sep=',', index=False, quoting=csv.QUOTE_ALL)

def run_experiment(first_session, last_session, merge_periods_only, DEBUG):
    if last_session < first_session:
        print("ERROR: last_session < first_session")
        return

    start = time.time()
    no_requests = 0
    no_dropped_motions = [0, 0, 0]
    
    session_starts = list(range(first_session, last_session + 1, 1))
    print(session_starts)

    if not merge_periods_only:
        for session_start in session_starts:
            no_requests, no_dropped_motions = download_session(no_requests, no_dropped_motions, session_start, DEBUG)

    merge_all_periods(session_starts, DEBUG)    
    
    elapsed_time = time.time()-start
    elapsed_time_str = f"{int(elapsed_time)//3600} hours {(int(elapsed_time)%3600)//60} minutes {int(elapsed_time)%60} seconds"

    # make sure we start with an empty file
    results_latex_file = RESULTS_LATEX_FILE_DEBUG if DEBUG else RESULTS_LATEX_FILE
    
    open(results_latex_file, 'w').close()
    write_variable_to_latex(results_latex_file, "nolongdropped",  f"{no_dropped_motions[0]:,}")
    write_variable_to_latex(results_latex_file, "nostortingetberdropped",  f"{no_dropped_motions[1]:,}")
    write_variable_to_latex(results_latex_file, "nocatdropped",  f"{no_dropped_motions[2]:,}")
    
    print("elapsed_time:", elapsed_time_str)
    print("no_requests:", no_requests)
    
    write_variable_to_latex(results_latex_file, "elapsedtime", elapsed_time_str)
    write_variable_to_latex(results_latex_file, "norequests", f"{no_requests:,}")


def main():

    parser = argparse.ArgumentParser()
    
    # model name
    parser.add_argument("--first", type=int, default=2018, help="first session start year")
    parser.add_argument("--last", type=int, default=2023, help="last session start year")
    parser.add_argument("--debug", type=int, default=0, help="Debug flag")
    
    args = parser.parse_args()
    
    first_session = args.first
    last_session = args.last
    DEBUG = bool(args.debug)
    merge_periods_only = False
    run_experiment(first_session, last_session, merge_periods_only, DEBUG)

if __name__ == "__main__":
    main()
    



