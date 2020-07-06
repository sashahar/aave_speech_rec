import pydub
from pydub import AudioSegment
import pandas as pd
import numpy as np
import os
from os import path
import threading
import re
import csv
import xml.etree.ElementTree as ET
import re

AUDIO_DIRS = ['data_voc']

TXT_DIR = 'data_voc'
RESULT_DIR = 'data_processed_voc'
MANIFEST_FILE = 'voc_manifest.csv'
ID_FILE = 'voc_ids.csv'
MIN_AUDIO_LENGTH = 5000
MAX_AUDIO_LENGTH = 20000 #20 seconds max length for audio segment

def get_name(filename):
	'''
    Expects file format to be "<region>_<lastname>_<firstname>.wav"
    '''
	components = filename.split("_")
	return components[1], components[2]

def write_files(identifier, num_segments, result, curr_text):

    result_path = path.join(RESULT_DIR + "/wav", 'voc_' + str(identifier) + '_part_{}'.format(num_segments) + '.wav')
    result.export(result_path, format = "wav")

    result_text_path = path.join(RESULT_DIR + "/txt", 'voc_' + str(identifier) + '_part_{}'.format(num_segments) + '.txt')
    groundtruth_text = ' '.join(curr_text)
    with open(result_text_path, "w") as txt_file:
        txt_file.write(' '.join(curr_text))

    writer = csv.writer(open(MANIFEST_FILE, "a"))
    writer.writerow([result_path, result_text_path, groundtruth_text, len(result)/1000])

def find_interviewee(root, transcript):
	components = transcript.split("/")[1].split(".")[0].split("_")
	first = components[2]

	for speaker in root.findall("Speakers/Speaker"):
		if "interviewer" not in speaker.attrib['name'].lower():
			if first.lower() in speaker.attrib['name'].lower():
				print(speaker.attrib['name'])
				return speaker.attrib['id']
	return "error"

def is_interviewee(turn, interviewee):
	if 'speaker' in turn.attrib:
		return turn.attrib['speaker'] == interviewee
	else:
		return False

def process_single_audio_file_trs(root_dir, filename, identifier, transcript):
	#open audio file
	audio_filepath = path.join(root_dir, filename + '.wav')
	audio_segment = AudioSegment.from_wav(audio_filepath)

	#parse transcript
	tree = ET.parse(transcript)
	root = tree.getroot()
	interviewee = find_interviewee(root, transcript)
	num_turns = len(root.findall("Episode/Section/Turn"))

	num_segments = 1
	curr_text = []
	total_time = 0
	result = AudioSegment.silent(duration=0)
	
	for i, turn in enumerate(root.findall("Episode/Section/Turn")):
		if is_interviewee(turn, interviewee):
			t1 = float(turn.attrib['startTime']) * 1000 #convert sec to millisec
			t2 = float(turn.attrib['endTime']) * 1000 #convert sec to millisec
			result += audio_segment[t1:t2]
			total_time += t2 - t1
			text = "".join(turn.itertext()).strip().replace('\n', ' ')
			text = re.sub('\[.*?\]', '', text)
			text = re.sub(' +', ' ', text) #Do we want to make them lower case and strip punctuation? Also how should we deal with disfluencies in the transcripts?
			if text != "":
				curr_text.append(text)
		if not is_interviewee(turn, interviewee) or total_time >= MAX_AUDIO_LENGTH or (i == num_turns - 1):
			if total_time >= MIN_AUDIO_LENGTH:
				write_files(identifier, num_segments, result, curr_text)
				num_segments += 1
			curr_text = []
			total_time = 0
			result = AudioSegment.silent(duration=0)

def create_time_slot_dictionary(root):
	time_slots = {}
	for time_slot in root.findall("TIME_ORDER/TIME_SLOT"):
		time_slots[time_slot.attrib["TIME_SLOT_ID"]] = int(time_slot.attrib["TIME_VALUE"])
	return time_slots

def find_interviewee_tier(root, transcript):
	components = transcript.split("/")[1].split(".")[0].split("_")
	first = components[2]

	for tier in root.findall("TIER"):
		if first.lower() in tier.attrib["TIER_ID"].lower():
			return tier
		elif "speaker" in tier.attrib["TIER_ID"].lower():
			return tier
		elif "interviewee" in tier.attrib["TIER_ID"].lower():
			return tier
	return "error"

def process_single_audio_file_eaf(root_dir, filename, identifier, transcript):
	#open audio file
	audio_filepath = path.join(root_dir, filename + '.wav')
	audio_segment = AudioSegment.from_wav(audio_filepath)

	#parse transcript
	tree = ET.parse(transcript)
	root = tree.getroot()
	time_slots = create_time_slot_dictionary(root)
	tier = find_interviewee_tier(root, transcript)
	print(tier.attrib["TIER_ID"])
	num_turns = len(tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION"))

	num_segments = 1
	curr_text = []
	total_time = 0
	last_end = 0
	result = AudioSegment.silent(duration=0)
	cut_off = 600 #in milliseconds

	for i, turn in enumerate(tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION")):
		start_time = time_slots[turn.attrib["TIME_SLOT_REF1"]]
		end_time = time_slots[turn.attrib["TIME_SLOT_REF2"]]

		if not (start_time - last_end < cut_off) or total_time >= MAX_AUDIO_LENGTH or (i == num_turns - 1):
			if total_time >= MIN_AUDIO_LENGTH:
				write_files(identifier, num_segments, result, curr_text)
				num_segments += 1
			curr_text = []
			total_time = 0
			result = AudioSegment.silent(duration=0)
			last_end = start_time

		result += audio_segment[last_end:end_time]
		total_time += end_time - last_end
		text = "".join(turn.itertext()).strip().replace('\n', ' ')
		text = re.sub('\[.*?\]', '', text)
		text = re.sub('\\{.*?\}', '', text)
		text = re.sub(' +', ' ', text) #Do we want to make them lower case and strip punctuation? Also how should we deal with disfluencies in the transcripts?
		if text != "":
			curr_text.append(text)
		last_end = end_time

		#Write out if last turn
		if i == num_turns - 1:
			if total_time >= MIN_AUDIO_LENGTH:
				write_files(identifier, num_segments, result, curr_text)

def process_single_audio_file(root_dir, filename, identifier):
    #open text file (text grid?)
    txt_filepath = path.join(TXT_DIR, filename + '.trs')
    if path.exists(txt_filepath):
    	process_single_audio_file_trs(root_dir, filename, identifier, txt_filepath)
    else:
    	txt_filepath = path.join(TXT_DIR, filename + '.eaf')
    	process_single_audio_file_eaf(root_dir, filename, identifier, txt_filepath)

def process_all_audio_files(dir_list):
	identifier = 0
	for dir in dir_list:
		for file in os.listdir(dir):
			ext = file.split(".")[1]
			#Skip transcripts
			if ext != "wav":
				continue
			full_path = path.join(dir, file)
			#skip directories
			if not path.isfile(full_path) or file == '.DS_Store':
			    continue
			print("file: ", file)
			file = file.split(".")[0]
			first, last = get_name(file)

			writer = csv.writer(open(ID_FILE, "a"))
			writer.writerow([identifier, first, last])

			process_single_audio_file(dir, file, identifier)
			identifier += 1

if __name__ == '__main__':
    #init manifest file
    np.savetxt(MANIFEST_FILE, np.array(
        ['wav_file,txt_file,groundtruth_text,duration']), fmt="%s", delimiter=",")
    #init ids file
    np.savetxt(ID_FILE, np.array(
        ['id,first,last']), fmt="%s", delimiter=",")
    process_all_audio_files(AUDIO_DIRS)
