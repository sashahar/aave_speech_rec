import pydub
from pydub import AudioSegment
import pandas as pd
from os import path

AUDIO_DIR ='data/DCA_audio_part01_2018.10.06'
TXT_DIR = 'data/DCA_textfiles_2018.10.06'
RESULT_DIR = 'data_processed'

def is_interviewer(str):
    return str.find("int") != -1

def process_single_audio_file(filepath):
    #open audio file
    audio_filepath = path.join(AUDIO_DIR, filepath + '.wav')
    audio_segment = AudioSegment.from_wav(audio_filepath)
    #open text file (text grid?)
    txt_filepath = path.join(TXT_DIR, filepath + '.txt')
    txt_df = pd.read_csv(txt_filepath, sep = '\t')
    print(txt_df.head())

    # Create an empty AudioSegment
    result = AudioSegment.silent(duration=0)

    for i, row in txt_df.iterrows():
        if not is_interviewer(row.Spkr):
            t1 = row.StTime * 1000 #convert sec to millisec
            t2 = row.EnTime * 1000 #convert sec to millisec
            result += audio_segment[t1:t2]

    result_path = path.join(RESULT_DIR, filepath + '.wav')
    result.export(result_path, format = "wav")
    # newAudio = newAudio[t1:t2]
    # newAudio.export('newSong.wav', format="wav") #Exports to a wav file in the current path.

FILE_PATH = 'DCA_se1_ag1_f_01_1'
process_single_audio_file(FILE_PATH)
