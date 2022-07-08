import csv
import os
import re
import threading
from os import path

import numpy as np
import pandas as pd
import pydub
from process_transcriptions import *
from pydub import AudioSegment

AUDIO_DIRS = [
#DC DATA
'data/DCB/DCB_audio_part01_2018.10.06','data/DCB/DCB_audio_part02_2018.10.06',\
'data/DCB/DCB_audio_part03_2018.10.06','data/DCB/DCB_audio_part04_2018.10.06',\
'data/DCB/DCB_audio_part05_2018.10.06','data/DCB/DCB_audio_part06_2018.10.06',\
'data/DCB/DCB_audio_part07_2018.10.06','data/DCB/DCB_audio_part08_2018.10.06',\
'data/DCB/DCB_audio_part09_2018.10.06','data/DCB/DCB_audio_part10_2018.10.06',\
'data/DCB/DCB_audio_part11_2018.10.06','data/DCB/DCB_audio_part12_2018.10.06',\
'data/DCB/DCB_audio_part13_2018.10.06','data/DCB/DCB_audio_part14_2018.10.06',\
#ROC DATA
# 'data/ROC/ROC_audio_part01_2020.05','data/ROC/ROC_audio_part02_2020.05',\
# 'data/ROC/ROC_audio_part03_2020.05','data/ROC/ROC_audio_part04_2020.05',\
# 'data/ROC/ROC_audio_part05_2020.05',\
# #ATL DATA
# 'data/ATL/ATL_audio_part01_2020.05','data/ATL/ATL_audio_part02_2020.05',\
# 'data/ATL/ATL_audio_part03_2020.05','data/ATL/ATL_audio_part04_2020.05',\
# #PRV DATA
# 'data/PRV/PRV_audio_part01_2018.10.06','data/PRV/PRV_audio_part02_2018.10.06',\
# 'data/PRV/PRV_audio_part03_2018.10.06','data/PRV/PRV_audio_part04_2018.10.06'
]

# AUDIO_DIRS = ['data/PRV_audio_part01_2018.10.06','data/PRV_audio_part02_2018.10.06',\
#     'data/PRV_audio_part03_2018.10.06','data/PRV_audio_part04_2018.10.06']

TXT_DIR = 'data/DCB/DCB_textfiles_2018.10.06' #'data/ROC_textfiles_2020.05'
RESULT_DIR = 'data_processed_DCB'
MANIFEST_FILE = 'DCB_manifest.csv'
METADATA_FILE = 'data/DCB/DCB_metadata_2018.10.06.txt'
MIN_AUDIO_LENGTH = 5000 #unit millisections: 5 seconds is min length for audio segment
MAX_AUDIO_LENGTH = 20000 #unit millisections: 20 seconds max length for audio segment

def is_interviewer(str):
    '''
    determines if speaker is the interviwer by searching for string 'int'
    this signals the interviewer in the marked up transcripts.
    '''
    return str.find("int") != -1

def include_file(filename):
    '''
    Expects file format to be "<region>_<socio-economic group>_<age group>_<gender>_<id>_<part>.wav"
    Filters out files from age group 1 (participants < 19 yo)
    '''
    components = filename.split("_")
    age, gender = get_metadata(filename)
    if age < 18:
        return False
    return True

def get_metadata(filepath):
    metadata = pd.read_csv(METADATA_FILE, delimiter = '\t')
    row = metadata[metadata['CORAAL.File'] == filepath]
    age = row.Age.values[0]
    gender = row.Gender.values[0]
    return age, gender

def write_files(filepath, num_segments, result, curr_text):
    '''
    Takes a list of audio segments and corresponding transcripts
    and writes them to the RESULT_DIR.
    Args:
    filepath: filename of original WAV file
    num_segments:
    result: AudioSegment to write to new WAV file
    curr_text: text transcript corresponding to the utterances in result.

    '''
    #WRITE_WAV_FILE
    result_path = path.join(RESULT_DIR + "/wav", \
        filepath + '_part_{}'.format(num_segments) + '.wav')
    result.export(result_path, format = "wav")


    #WRITE RAW TEXT
    result_text_path = path.join(RESULT_DIR + "/txt_raw", \
        filepath + '_part_{}'.format(num_segments) + '.txt')
    groundtruth_text = ' '.join(curr_text)
    with open(result_text_path, "w") as txt_file:
        txt_file.write(groundtruth_text)

    #WRITE ADJUSTED TEXT
    result_text_path = path.join(RESULT_DIR + "/txt", \
        filepath + '_part_{}'.format(num_segments) + '.txt')
    groundtruth_text = clean_coraal_lambda(groundtruth_text)
    groundtruth_text = clean_within_all(groundtruth_text)
    groundtruth_text = groundtruth_text.upper()
    with open(result_text_path, "w") as txt_file:
        txt_file.write(groundtruth_text)

    #WRITE ROW IN MANIFEST FILE
    age, gender = get_metadata(filepath)
    writer = csv.writer(open(MANIFEST_FILE, "a"))
    writer.writerow([result_path, result_text_path, ' '.join(curr_text), groundtruth_text, len(result)/1000, age, gender])

def is_useful_content(str):
    x = re.search("\(pause [0-9]\.[0-9]{2}\)", str)
    if x:
        return False
    return True


def process_single_audio_file(root_dir, filename):
    '''
    Expects file at path <root_dir>/<filename> to be a wav file.
    Breaks this file into chunks approx 20 seconds in length.
    Excludes utterances from the interviewer.  Writes resulting snippets
    to RESULT_DIR/wav
    '''
    #open audio file
    audio_filepath = path.join(root_dir, filename + '.wav')
    audio_segment = AudioSegment.from_wav(audio_filepath)
    #open text file (text grid?)
    txt_filepath = path.join(TXT_DIR, filename + '.txt')
    txt_df = pd.read_csv(txt_filepath, sep = '\t')

    # Create an empty AudioSegment
    result = AudioSegment.silent(duration=0)
    total_time = 0
    num_segments = 1
    curr_text = []
    for i, row in txt_df.iterrows():
        # print(total_time)
        if not is_interviewer(row.Spkr):
            t1 = row.StTime * 1000 #convert sec to millisec
            t2 = row.EnTime * 1000 #convert sec to millisec
            result += audio_segment[t1:t2]
            total_time += t2 - t1
            if is_useful_content(row.Content):
                curr_text.append(row.Content)
        if is_interviewer(row.Spkr) or total_time >= MAX_AUDIO_LENGTH or (i == len(txt_df) - 1):
            if total_time >= MIN_AUDIO_LENGTH:
                write_files(filename, num_segments, result, curr_text)
                num_segments += 1
            curr_text = []
            total_time = 0
            result = AudioSegment.silent(duration=0)

def process_all_audio_files(dir_list):
    for dir in dir_list:
        for file in os.listdir(dir):
            full_path = path.join(dir, file)
            #skip directories
            if not path.isfile(full_path) or file == '.DS_Store':
                continue
            print("file: ", file)
            file = file.split(".")[0]
            if include_file(file):
                process_single_audio_file(dir, file)

if __name__ == '__main__':
    #init manifest file
    np.savetxt(MANIFEST_FILE, np.array(
        ['wav_file,txt_file,groundtruth_text_raw, groundtruth_text_train ,duration,age,gender']), fmt="%s", delimiter=",")
    process_all_audio_files(AUDIO_DIRS)
