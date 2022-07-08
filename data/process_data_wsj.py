#Process WSJ text transcriptions to fit our format
import os
import re

import numpy as np
import pandas as pd

OUTPUT_DIR = 'wsj/train/txt'
BASE_PATH_SLURM = '/juice/scr/aharris6/'
INPUT_MANIFEST_PATH = 'manifests_wsj/temp/train_all.csv' #'manifests_wsj/temp/val.csv'
OUTPUT_MANIFEST_PATH = 'manifests_wsj/train_manifest_wsj.csv'

TRANSCRIPTION_DIRS = ['wsj/13-34.1/wsj1/trans/wsj1', 'wsj/11-10.1/wsj0/transcrp/dots']
EXCLUDE_LIST = ['48wc0304']

def create_output_dir(dir_id):
    if not os.path.exists(OUTPUT_DIR + "/" + dir_id):
        os.mkdir(OUTPUT_DIR + "/" + dir_id)

def get_transcript_components(line):
    component_list = line.split("(")
    transcript = component_list[0].strip()
    #Remove special characters other than apostrophe
    transcript =re.sub(r'[^\s\w$\']|_', '',transcript) # replace special characters except apstrophe with empty string
    transcript = re.sub("\s+"," ",''.join(transcript)) # standardize whitespace
    id = component_list[1].strip()[:-1] #remove trailing close paren
    return transcript.upper(), id.lower()

def read_dot_file(filepath, dir_id, id_set):
    #read individual lines of transcription file
    output_dict = {'txt_file': [], 'id': [], 'transcript': []}
    file = open(filepath, 'r')
    for line in file.readlines():
        transcript, id = get_transcript_components(line)
        if id in EXCLUDE_LIST or id not in id_set: continue #or id not in id_set:
        destination_path = OUTPUT_DIR + "/" + dir_id + "/" + id + ".txt"
        create_output_dir(dir_id)
        with open(destination_path, "w") as txt_file:
            txt_file.write(transcript)
        output_dict['txt_file'].append(destination_path)
        output_dict['id'].append(id)
        output_dict['transcript'].append(transcript)
    return output_dict

def get_all_wav_files_in_train(index_path):
    index_file = open(index_path, 'r')
    df_dict = {'wav_file': [], 'id': []}
    for line in index_file.readlines():
        #skip comments in index file, which are denoted using the character ';'
        if line[0] == ";": continue
        disc_id, filepath = [item.strip() for item in line.split(":")]
        components = filepath.split("/")
        identifier = components[-1].split(".")[0]
        df_dict['wav_file'].append(filepath)
        df_dict['id'].append(identifier)
    return pd.DataFrame(df_dict)

def process_transcript_directory(id_set):
    #iterate through all directories in transcript directory
    all_files_dict = {'txt_file': [], 'id': [], 'transcript': []}
    for dir in TRANSCRIPTION_DIRS:
        for dataset_id in os.listdir(dir):
            print(dataset_id)
            for dir_entry in os.listdir(dir + "/" + dataset_id):
                if not os.path.isdir(dir + "/" + dataset_id + "/" + dir_entry): continue
                for file in os.listdir(dir + "/" + dataset_id + "/" + dir_entry):
                    if file.split(".")[-1] == 'dot':
                        output_dict = read_dot_file(dir + "/" + dataset_id + "/" + dir_entry + "/" + file, dir_entry, id_set)
                        all_files_dict['txt_file'].extend(output_dict['txt_file'])
                        all_files_dict['id'].extend(output_dict['id'])
                        all_files_dict['transcript'].extend(output_dict['transcript'])
    return pd.DataFrame(all_files_dict)

if __name__ == '__main__':
    print("Processing WSJ txt files, writing results to: {}".format(OUTPUT_DIR))

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    wav_df = pd.read_csv(INPUT_MANIFEST_PATH, names = ['input_file','wav_file','id'])#get_all_wav_files_in_train(INDEX_PATH)
    original_length = len(wav_df)
    id_set = set(wav_df.id)
    txt_df = process_transcript_directory(id_set)
    txt_df['txt_file'] = txt_df['txt_file'].apply(lambda x: BASE_PATH_SLURM + x)
    txt_df.to_csv("txt_df.csv", index = False)
    txt_df.set_index('id', inplace = True)
    final_manifest = wav_df.join(txt_df, on = 'id', how = 'inner')
    final_len = len(final_manifest)
    cols_new_order = ['wav_file', 'txt_file', 'transcript', 'id']
    final_manifest = final_manifest[cols_new_order]
    print("Input has length {}, output has length {}".format(original_length, final_len))
    final_manifest.to_csv(OUTPUT_MANIFEST_PATH, index = False)
