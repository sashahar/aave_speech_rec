import pydub
from pydub import AudioSegment
import pandas as pd
from os import path
import threading
import re

AUDIO_DIR ='data/DCA_audio_part01_2018.10.06'
TXT_DIR = 'data/DCA_textfiles_2018.10.06'
RESULT_DIR = 'data_processed'
MAX_SEGMENT_LENGTH = 50000 #50 seconds max length for audio segment

def is_interviewer(str):
    return str.find("int") != -1

def write_files(filepath, num_segments, result, curr_text):
    result_path = path.join(RESULT_DIR + "/wav", \
        filepath + '_part_{}'.format(num_segments) + '.wav')
    result.export(result_path, format = "wav")


    result_text_path = path.join(RESULT_DIR + "/txt", \
        filepath + '_part_{}'.format(num_segments) + '.txt')
    with open(result_text_path, "w") as txt_file:
        txt_file.write(' '.join(curr_text))

def is_useful_content(str):
    x = re.search("\(pause [0-9]\.[0-9]{2}\)", str)
    if x:
        return False
    return True


#TODO: this function will eventually be a thread routine.
#good candidate for multiprocessing because of letency of reading files
def process_single_audio_file(filepath):
    #open audio file
    audio_filepath = path.join(AUDIO_DIR, filepath + '.wav')
    audio_segment = AudioSegment.from_wav(audio_filepath)
    #open text file (text grid?)
    txt_filepath = path.join(TXT_DIR, filepath + '.txt')
    txt_df = pd.read_csv(txt_filepath, sep = '\t')

    # Create an empty AudioSegment
    result = AudioSegment.silent(duration=0)
    total_time = 0
    num_segments = 1
    curr_text = []
    for i, row in txt_df.iterrows():
        print(total_time)
        if not is_interviewer(row.Spkr) and is_useful_content(row.Content):
            t1 = row.StTime * 1000 #convert sec to millisec
            t2 = row.EnTime * 1000 #convert sec to millisec
            result += audio_segment[t1:t2]
            total_time += t2 - t1
            curr_text.append(row.Content)
            if (total_time >= MAX_SEGMENT_LENGTH) or (i == len(txt_df) - 1):
                write_files(filepath, num_segments, result, curr_text)
                curr_text = []
                total_time = 0
                result = AudioSegment.silent(duration=0)
                num_segments += 1

FILE_PATH = 'DCA_se1_ag1_f_01_1'
process_single_audio_file(FILE_PATH)
