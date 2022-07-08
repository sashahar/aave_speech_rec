import csv
import os
import re
from os import path

import numpy as np
import pandas as pd
from text_cleaning_utils import *

MANIFEST_FILE = "ROC_manifest_transcribed_new.csv"
OUTFILE = "transcriptions/ROC_manifest_wer_new.csv"

#globally Accessible lists
swear_words = ['nigga', 'niggas', 'shit', 'bitch', 'damn', 'fuck', 'fuckin', 'fucking', 'motherfuckin', 'motherfucking']
filler_words = ['um', 'uh', 'mm', 'hm', 'ooh', 'woo', 'mhm', 'mm-hm', 'huh', 'ha']

pre_cardinal = ['N', 'E', 'S', 'W', 'NE', 'NW', 'SE', 'SW']
post_cardinal = ['North', 'East', 'South', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest']

pre_list = ['cuz', 'ok', 'o', 'till', 'yup', 'ima', 'mister', 'dr',
           'carryout', 'sawmill', 'highschool', 'worldclass', 'rd', 'blvd',
           'theatre', 'neighbour', 'neighbours', 'neighbourhood', 'programme']
post_list = ['cause', 'okay', 'oh', 'til', 'yep', 'imma', 'mr', 'doctor',
            'carry out', 'saw mill', 'high school', 'world class', 'road', 'boulevard',
             'theater', 'neighbor', 'neighbors', 'neighborhood', 'program']

def clean_coraal_lambda(text):
    '''
    Applies several important transformations to raw CORAAL transcripts.
    Changes spelling on some words that are out of vocab for speech APIs.
    i.e. "aks" -> "ask"
    Removes CORAAL flags like unintelligible and redacted words.
    '''

    # Relabel CORAAL words. For consideration: aks -> ask?
    split_words = text.split()
    split_words = [x if x != 'busses' else 'buses' for x in split_words]
    split_words = [x if x != 'aks' else 'ask' for x in split_words]
    split_words = [x if x != 'aksing' else 'asking' for x in split_words]
    split_words = [x if x != 'aksed' else 'asked' for x in split_words]
    text = ' '.join(split_words)

    # remove CORAAL unintelligible flags
    text = re.sub("\/unintelligible\/",'',''.join(text))
    text = re.sub("\/inaudible\/",'',''.join(text))
    text = re.sub('\/RD(.*?)\/', '',''.join(text))
    text = re.sub('\/(\?)\1*\/', '',''.join(text))
    text = re.sub('\[', '',text) # remove square brackets from text
    text = re.sub('\]', '',text) # remove square brackets from text

    # remove nonlinguistic markers
    text = remove_markers(text, ['<>', '()', '{}'])

    return text

def clean_voc_lambda(text):
    '''
    Applies several important transformations to raw VoC transcripts.
    '''

    text = re.sub('\(', '',text) # remove brackets from text
    text = re.sub('\)', '',text) # remove brackets from text

    # remove nonlinguistic markers
    text = remove_markers(text, ['[]', '{}'])

    return text

def general_string_cleaning(text):
    re.sub(r"([a-z])\-([a-z])", r"\1 \2", text , 0, re.IGNORECASE) # replace inter-word hyphen with space
    text =re.sub(r'[^\s\w$\']|_', ' ',text) # replace special characters with space, except $
    text = re.sub("\s+"," ",''.join(text)) # standardize whitespace
    #
    # # update numeric numbers to strings and remove $
    text = re.sub("ft ²", "square feet", ''.join(text))
    text = fix_numbers(text)
    text = re.sub("\$",'dollars',''.join(text))
    text = re.sub("\£",'pounds',''.join(text))
    return text

def clean_coraal(df):
    df.loc[:,'clean_text'] = df.loc[:,'groundtruth_text'].copy()
    # Replace original unmatched CORAAL transcript square brackets with squiggly bracket
    # df.loc[:,'clean_text'] = df['clean_text'].str.replace('\[','\{')
    # df.loc[:,'clean_text'] = df['clean_text'].str.replace('\]','\}')

    df['clean_text'] = df.apply(lambda x: clean_coraal_lambda(x['clean_text']), axis=1)
    return df

def clean_within_all(text):
    # fix spacing in certain spellings
    text = re.sub('T V','TV',''.join(text))
    text = re.sub('D C','DC',''.join(text))

    # remove remaining floating non-linguistic words
    single_paren = ['<','>', '(',')', '{','}','[',']']
    for paren in single_paren:
        linguistic_words  = [word for word in text.split() if paren not in word]
        text = ' '.join(linguistic_words)

    # general string cleaning
    text = general_string_cleaning(text)

    # standardize spellings
    split_words = text.split()
    for i in range(len(pre_list)):
        split_words = [x if x.lower() != pre_list[i] else post_list[i] for x in split_words]
    text = ' '.join(split_words)

    # deal with cardinal directions
    split_words_dir = text.split()
    for i in range(len(pre_cardinal)):
        split_words_dir = [x if x != pre_cardinal[i] else post_cardinal[i] for x in split_words_dir]
    text = ' '.join(split_words_dir)

    # deal with state abbreviations
    #text = fix_state_abbrevs(text)
    text = text.lower()

    # update spacing in certain spellings
    spacing_list_pre = ['north east', 'north west', 'south east', 'south west', 'all right']
    spacing_list_post = ['northeast', 'northwest', 'southeast', 'southwest', 'alright']
    for i in range(len(spacing_list_pre)):
        text = re.sub(spacing_list_pre[i], spacing_list_post[i],''.join(text))

    # remove filler words and swear words
    #remove_words = swear_words + filler_words
    #resultwords  = [word for word in text.split() if word not in remove_words]
    #result = ' '.join(resultwords)

    result = text

    return result

def clean_all_transcripts(df, cols_to_clean):
    clean_df = df.copy()
    for col in cols_to_clean:
        clean_df[col] = clean_df[col].replace(np.nan, '', regex=True)

    for col in cols_to_clean:
        clean_df['clean_text'] = df.apply(lambda x: clean_within_all(x['clean_text']), axis=1)
        clean_df['clean_' + col] = clean_df.apply(lambda x: clean_within_all(x[col]), axis=1)

    return clean_df

def wer_calc(transcripts, human_clean_col, asr_clean_col):
    # Calculate WER
    new_transcripts = transcripts.copy()
    ground_truth = transcripts[human_clean_col].tolist()
    for col in asr_clean_col:
        new_transcripts[col] = new_transcripts[col].replace(np.nan, '', regex=True)
        asr_trans = new_transcripts[col].tolist()
        wer_list = []
        for i in range(len(ground_truth)):
            wer_list.append(wer(ground_truth[i], asr_trans[i]))
        new_transcripts[col+"_wer"] = wer_list
    return new_transcripts

def apply_cleaning_rules(df, cols_to_clean):
    #Cleans everything from CORAAL transcript
    all_usable = clean_coraal(df)
    clean_all = clean_all_transcripts(all_usable, cols_to_clean)
    return clean_all


if __name__ == '__main__':
    #init manifest file
    print("starting...")
    all_snippets = pd.read_csv(MANIFEST_FILE)

    cols_to_clean = ['google_transcription', 'mod9_transcription']

    clean_snippets = apply_cleaning_rules(all_snippets, cols_to_clean)

    old_len = len(clean_snippets)
    print("Removing short snippets...")

    clean_snippets['wordcount'] = clean_snippets['clean_text'].str.split().str.len()
    clean_snippets = clean_snippets[clean_snippets['wordcount']>=5]
    print("Num Removed = ", old_len - len(clean_snippets))

        # Create ASR list for WER calculations
    clean_asr_trans_list = ['clean_google_transcription', 'clean_mod9_transcription']

    # Run WER calculations on all usable snippets, with cleaning
    print("calculating word error rate...")
    clean_transcripts_wer = wer_calc(clean_snippets, 'clean_text', clean_asr_trans_list)

    print("overall Google WER: ", clean_transcripts_wer['clean_google_transcription_wer'].mean())
    print("overall Mod9 WER: ", clean_transcripts_wer['clean_mod9_transcription_wer'].mean())

    clean_transcripts_wer.to_csv(OUTFILE, index = False)
