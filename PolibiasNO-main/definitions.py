RANDOM_SEED = 2025

party_codes = ['A', 'H', 'Sp', 'FrP', 'KrF', 'MDG', 'SV', 'V', 'R', 'PF']
party_codes_ex_PF = ['A', 'H', 'Sp', 'FrP', 'KrF', 'MDG', 'SV', 'V', 'R']

party_names_NO = ['Arbeiderpartiet',
                'Høyre',
                'Senterpartiet',
                'Fremskrittspartiet',
                'Kristelig Folkeparti',
                'Miljøpartiet De Grønne',
                'Sosialistisk Venstreparti',
                'Venstre',
                'Rødt',
                'Pasientfokus']

party_names_NO_ex_PF = ['Arbeiderpartiet',
                'Høyre',
                'Senterpartiet',
                'Fremskrittspartiet',
                'Kristelig Folkeparti',
                'Miljøpartiet De Grønne',
                'Sosialistisk Venstreparti',
                'Venstre',
                'Rødt']

party_names_EN = ['the Labour Party',
                    'the Conservative Party',
                    'the Centre Party',
                    'the Progress Party',
                    'the Christian Democratic Party',
                    'the Green Party',
                    'the Socialist Left Party',
                    'the Liberal Party',
                    'the Red Party',
                    'Patient Focus']

party_names_EN_ex_PF = ['the Labour Party',
                    'the Conservative Party',
                    'the Centre Party',
                    'the Progress Party',
                    'the Christian Democratic Party',
                    'the Green Party',
                    'the Socialist Left Party',
                    'the Liberal Party',
                    'the Red Party']

direction_NO = ['venstre', 'høyre', 'sentrums']
direction_EN = ['leftist', 'rightist', 'centrist']

left_dir = 'left'
center_dir = 'center'
right_dir = 'right'
direction_codes = [left_dir, right_dir, center_dir]

llama3_name = 'Llama3-instruct'
gpt35_name = 'gpt-3.5-turbo'
gpt4o_name = 'gpt-4o-mini'
mistral_name = 'Mistral-instruct'
gemma2_name = 'Gemma2-instruct'
falcon3_name = 'Falcon3-instruct'
nor_gpt_name = 'NorskGPT'

llama3_name_short = 'Llama3'
gpt35_name_short = 'GPT3.5t'
gpt4o_name_short = 'GPT4o'
mistral_name_short = 'Mistral'
gemma2_name_short = 'Gemma2'
falcon3_name_short = 'Falcon3'
nor_gpt_name_short = 'NorGPT'

results_latex_folder = "results_latex"

komite_dict_NO = {
    'FINANS': 'Finans',
    'HELSEOMS': 'Helse og omsorg',
    'KOMMFORV': 'Kommunal og forvaltning',
    'FAMKULT': 'Familie og kultur',
    'ENERGI': 'Energi og miljø',
    'UFO': 'Utdanning og forsking', # TODO FIX
    'NÆRING': 'Næring',
    'ARBSOS': 'Arbeid og sosial',
    'JUSTIS': 'Justis',
    'TRANSKOM': 'Transport og kommunikasjon',
    'UFK': 'Utenriks og forsvar',
    'KONTROLL': 'Kontroll og konstitusjon'
}

komite_dict = {
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
