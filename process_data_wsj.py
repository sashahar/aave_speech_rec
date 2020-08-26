#Process WSJ text transcriptions to fit our format
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = 'wsj_txt/si_tr_s'
INPUT_MANIFEST_PATH = 'manifests_wsj/train_manifest_wsj_wav_only.csv'
OUTPUT_MANIFEST_PATH = 'wsj/manifests_wsj/train_manifest.csv'

TRANSCRIPTION_DIR = 'wsj/13-34.1/wsj1/trans/wsj1/si_tr_s'
EXCLUDE_LIST = ['48wc0304']

def create_output_dir(dir_id):
    if not os.path.exists(OUTPUT_DIR + "/" + dir_id):
        os.mkdir(OUTPUT_DIR + "/" + dir_id)

def get_transcript_components(line):
    component_list = line.split("(")
    transcript = component_list[0].strip()
    #Remove special characters other than apostrophe
    transcript =re.sub(r'[^\s\w$\']|_', ' ',transcript) # replace special characters with space, except $
    transcript = re.sub("\s+"," ",''.join(transcript)) # standardize whitespace
    id = component_list[1].strip()[:-1] #remove trailing close paren
    return transcript, id.lower()

def read_lsn_file(filepath, dir_id, id_set):
    #read individual lines of transcription file
    output_dict = {'txt_file': [], 'id': [], 'transcript': []}
    file = open(filepath, 'r')
    create_output_dir(dir_id)
    for line in file.readlines():
        transcript, id = get_transcript_components(line)
        if id in EXCLUDE_LIST or id not in id_set: continue
        destination_path = OUTPUT_DIR + "/" + dir_id + "/" + id + ".txt"
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
    all_files_dict = {'txt_file': [], 'id': [], 'transcription': []}
    for dir_entry in os.listdir(TRANSCRIPTION_DIR):
        for file in os.listdir(TRANSCRIPTION_DIR + "/" + dir_entry):
            if file.split(".")[-1] == 'lsn':
                output_dict = read_lsn_file(TRANSCRIPTION_DIR + "/" + dir_entry + "/" + file, dir_entry, id_set)
                all_files_dict['txt_file'].extend(output_dict['txt_file'])
                all_files_dict['id'].extend(output_dict['id'])
                all_files_dict['transcription'].extend(output_dict['transcription'])
    return pd.DataFrame(all_files_dict)

if __name__ == '__main__':
    print("Processing WSJ txt files, writing results to: {}".format(OUTPUT_DIR))

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    wav_df = pd.read_csv(INPUT_MANIFEST_PATH)#get_all_wav_files_in_train(INDEX_PATH)
    id_set = set(wav_df.id)
    txt_df = process_transcript_directory(id_set)
    txt_df.set_index('id', inplace = True)
    final_manifest = wav_df.join(txt_df, on = 'id', how = 'inner')
    cols_new_order = ['wav_file', 'txt_file', 'groundtruth_text', 'id']
    final_manifest = final_manifest[cols_new_order]
    print(len(final_manifest))
    print(final_manifest.head())
    final_manifest.to_csv(OUTPUT_MANIFEST_PATH, index = False)
