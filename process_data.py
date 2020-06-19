import pydub
from pydub import AudioSegment
import pandas as pd
import numpy as np
import os
from os import path
import threading
import re
import csv

# AUDIO_DIRS = ['data/ATL_audio_part01_2020.05','data/ATL_audio_part02_2020.05',\
#     'data/ATL_audio_part03_2020.05','data/ATL_audio_part04_2020.05']

# AUDIO_DIRS = ['data/DCA_audio_part02_2018.10.06', \
#     'data/DCA_audio_part03_2018.10.06', 'data/DCA_audio_part04_2018.10.06', \
#     'data/DCA_audio_part05_2018.10.06','data/DCA_audio_part06_2018.10.06', \
#     'data/DCA_audio_part07_2018.10.06', 'data/DCA_audio_part08_2018.10.06' \
#     'data/DCA_audio_part09_2018.10.06', 'data/DCA_audio_part10_2018.10.06'],\

AUDIO_DIRS = ['data/ROC_audio_part01_2020.05','data/ROC_audio_part02_2020.05',\
    'data/ROC_audio_part03_2020.05','data/ROC_audio_part04_2020.05',\
    'data/ROC_audio_part05_2020.05']

TXT_DIR = 'data/ROC_textfiles_2020.05'
RESULT_DIR = 'data_processed_ROC'
MANIFEST_FILE = 'ROC_manifest.csv'
METADATA_FILE = 'data/ROC_metadata_2020.05.txt'
MIN_AUDIO_LENGTH = 5000
MAX_AUDIO_LENGTH = 50000 #50 seconds max length for audio segment

def is_interviewer(str):
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

    result_path = path.join(RESULT_DIR + "/wav", \
        filepath + '_part_{}'.format(num_segments) + '.wav')
    result.export(result_path, format = "wav")


    result_text_path = path.join(RESULT_DIR + "/txt", \
        filepath + '_part_{}'.format(num_segments) + '.txt')
    groundtruth_text = ' '.join(curr_text)
    with open(result_text_path, "w") as txt_file:
        txt_file.write(' '.join(curr_text))

    #write to manifest
    # with open(MANIFEST_FILE, "a") as manifest_file:
    #     print("writing to manifest: ", MANIFEST_FILE)
    #     print("{},{},{},{}\n".format(
    #                 result_path, result_text_path, groundtruth_text, len(result)/1000))
    age, gender = get_metadata(filepath)
    writer = csv.writer(open(MANIFEST_FILE, "a"))
    writer.writerow([result_path, result_text_path, groundtruth_text, len(result)/1000, age, gender])
        # manifest_file.write("\{},{},{},{}\n".format(
        #             result_path, result_text_path, groundtruth_text, len(result)/1000))

def is_useful_content(str):
    x = re.search("\(pause [0-9]\.[0-9]{2}\)", str)
    if x:
        return False
    return True


#TODO: this function will eventually be a thread routine.
#good candidate for multiprocessing because of letency of reading files
def process_single_audio_file(root_dir, filename):
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

# np.savetxt(MANIFEST_FILE, np.array(
#     ['wav_file,txt_file,groundtruth_text,duration']), fmt="%s", delimiter=",")
# FILE_PATH = 'DCA_se1_ag1_f_01_1'
# process_single_audio_file(AUDIO_DIRS[0], FILE_PATH)

#TODO: limit to ag groups excluding age group 1
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
        ['wav_file,txt_file,groundtruth_text,duration,age,gender']), fmt="%s", delimiter=",")
    process_all_audio_files(AUDIO_DIRS)
