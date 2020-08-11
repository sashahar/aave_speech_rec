#!/usr/bin/env python3

import argparse
import csv
import os
from os import path
import re

import numpy as np
import pandas as pd
from pydub import AudioSegment

from process_transcriptions import *

REGIONS = ['DCB', 'PRV', 'ROC', 'ATL']
# Kinda ugly way of retrieving the metadata...
METADATA_FILES = {'ATL': '/mnt/disk1/data/coraal/tars/ATL_metadata_2020.05.txt',
                  'DCB': '/mnt/disk1/data/coraal/tars/DCB_metadata_2018.10.06.txt',
                  'PRV': '/mnt/disk1/data/coraal/tars/PRV_metadata_2018.10.06.txt',
                  'ROC': '/mnt/disk1/data/coraal/tars/ROC_metadata_2020.05.txt'}

MIN_AUDIO_LENGTH = 5000  # unit millisections: 5 seconds is min length for audio segment
MAX_AUDIO_LENGTH = 20000  # unit millisections: 20 seconds max length for audio segment


class Global:
    '''Stores globals, like args. Not meant to be instantiated.'''
    args = None


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
    age, gender = get_metadata(filename)
    if age < 18:
        return False
    return True


def get_metadata(filepath):
    '''
    Expects file format to be "<region>_<socio-economic group>_<age group>_<gender>_<id>_<part>.wav"
    Filters out files from age group 1 (participants < 19 yo)
    '''
    region = filepath.split('_')[0]
    metadata = pd.read_csv(METADATA_FILES[region], delimiter = '\t')
    row = metadata[metadata['CORAAL.File'] == filepath]
    age = row.Age.values[0]
    gender = row.Gender.values[0]
    return age, gender


def write_files(filepath, num_segments, result, curr_text):
    '''
    Takes a list of audio segments and corresponding transcripts
    and writes them to ${Global.args.output_dir}.
    Args:
    filepath: filename of original WAV file
    num_segments:
    result: AudioSegment to write to new WAV file
    curr_text: text transcript corresponding to the utterances in result.

    '''
    #WRITE_WAV_FILE
    result_path = path.join(Global.args.output_dir + "/wav", \
        filepath + '_part_{}'.format(num_segments) + '.wav')
    result.export(result_path, format = "wav")


    #WRITE RAW TEXT
    result_text_path = path.join(Global.args.output_dir + "/txt_raw", \
        filepath + '_part_{}'.format(num_segments) + '.txt')
    groundtruth_text = ' '.join(curr_text)
    with open(result_text_path, "w") as txt_file:
        txt_file.write(groundtruth_text)

    #WRITE ADJUSTED TEXT
    result_text_path = path.join(Global.args.output_dir + "/txt", \
        filepath + '_part_{}'.format(num_segments) + '.txt')
    groundtruth_text = clean_coraal_lambda(groundtruth_text)
    groundtruth_text = clean_within_all(groundtruth_text)
    groundtruth_text = groundtruth_text.upper()
    with open(result_text_path, "w") as txt_file:
        txt_file.write(groundtruth_text)

    #WRITE ROW IN MANIFEST FILE
    age, gender = get_metadata(filepath)
    writer = csv.writer(open(Global.args.output_manifest, "a"))
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
    to ${Global.args.output_dir}/wav
    '''
    #open audio file
    audio_filepath = path.join(root_dir, filename + '.wav')
    audio_segment = AudioSegment.from_wav(audio_filepath)
    #open text file (text grid?)
    txt_filepath = path.join(root_dir, filename + '.txt')
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


def process_all_audio_files(data_dir):
    for region in REGIONS:
        region_dir = path.join(data_dir, region)
        for file in os.listdir(region_dir):
            full_path = path.join(region_dir, file)
            # skip directories and hidden files and non-wave files
            if not path.isfile(full_path) or file[0] == '.' or file[-4:] != '.wav':
                continue
            print("file: ", file)
            file = file.split(".")[0]
            if include_file(file):
                process_single_audio_file(region_dir, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_dir', default='/mnt/disk1/data/coraal',
                        help='Base directory where all the CORAAL data lives. Assumes directories DCB, PRV, ROC, and ATL exist.')
    parser.add_argument('output_dir', default='data_processed',
                        help='Output of audio segments and resultant text files.')
    parser.add_argument('--output-manifest', default='coraal_manifest.csv')
    Global.args = parser.parse_args()
    # init manifest file
    np.savetxt(Global.args.output_manifest, np.array(
        ['wav_file,txt_file,groundtruth_text_raw, groundtruth_text_train ,duration,age,gender']), fmt="%s", delimiter=",")
    process_all_audio_files(Global.args.data_base_dir)
