import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

audio_conf = {
    'sample_rate': 44100,  # The sample rate for the data/model features
    'window_size': 0.02,  # Window size for spectrogram generation (seconds)
    'window_stride': 0.01,  # Window stride for spectrogram generation (seconds)
    'n_fft': 1024, #sample_rate*window_size rounded to nearest power of 2, for efficiency
    'window':scipy.signal.hamming
}

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super().__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def _collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = batch[0][0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_lengths = torch.IntTensor(minibatch_size)
    target_lengths = torch.IntTensor(minibatch_size)
    targets = []
    files = []
    for i in range(minibatch_size):
        example = batch[i]
        spectrogram = example[0]
        transcript = example[1]
        files.append(example[2])
        seq_length = spectrogram.size(1)
        inputs[i][0].narrow(1, 0, seq_length).copy_(spectrogram)
        input_lengths[i] = seq_length
        target_lengths[i] = len(transcript)
        targets.extend(transcript)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_lengths, target_lengths, files

class SpectrogramDataset(Dataset):
    def __init__(self, manifest_filepath, char_vocab_path, audio_conf = audio_conf):
        super().__init__()

        with open(manifest_filepath) as f:
            data_filepaths = f.readlines()
        data_filepaths = [x.strip().split(',') for x in data_filepaths]
        self.data_filepaths = data_filepaths
        self.size = len(data_filepaths)
        self.char2ind = self.create_char2ind_map(char_vocab_path)
        self.audio_conf = audio_conf

    '''Takes in a file path to the character vocab and returns a dictionary
    mapping characters to indices'''
    def create_char2ind_map(self, char_vocab_path):
        with open(char_vocab_path) as char_vocab_file:
            characters = str(''.join(json.load(char_vocab_file)))
        char2ind = dict([(characters[i], i) for i in range(len(characters))])
        return char2ind

    def clean_coraal_transcript(text):
        #convert to upper case
        text = clean_coraal_lambda(text)
        text = clean_within_all_lambda(text)
        return text

    #Takes in an audio path and returns a normalized array representing the audio
    def load_audio(self, audio_path):
        #returns audio time series with length duration*sample_rate
        sound, sample_rate = librosa.load(audio_path, sr = self.audio_conf['sample_rate'])
        return sound

    def get_mfcc_features(self, sound):
        mfcc = librosa.feature.mfcc(sound, sr = self.audio_conf['sample_rate'], \
            n_mfcc = 40, n_fft = self.audio_conf['n_fft'], window = self.audio_conf['window'])
        return mfcc

    #Takes in a sound array and returns a spectrogram
    def get_spectrogram(self, sound):
        n_fft = self.audio_conf['n_fft']
        win_length = n_fft
        hop_length = int(self.audio_conf['sample_rate'] * self.audio_conf['window_stride'])
        #Return complex values spectrogram (D) - Short Time Fourier Transform
        D = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
                                 win_length=win_length, window=self.audio_conf['window'])
        #Transofrms D into its magnitude and phase components
        spect, phase = librosa.magphase(D)
        # S = log(S + 1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        #Normalize spectrogram
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)
        return spect

    #Takes in an audio path and returns a spectrogram
    #Spectrogram is a num_seconds * max_frequency list of lists
    def parse_audio(self, audio_path):
        sound = self.load_audio(audio_path)
        spect = self.get_spectrogram(sound)
        return spect

    '''Takes in a transcript path and the char2ind map and returns an array of
    character indices representing the transcript'''
    def parse_transcript(self, transcript_path, char2ind):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = self.clean_coraal_transcript(transcript)
        print("TRANSCRIPT: ", transcript)
        transcript = list(filter(None, [char2ind.get(x) for x in list(transcript)]))
        return transcript

    def __getitem__(self, index):
        example = self.data_filepaths[index]
        audio_path, transcript_path = example[0], example[1]
        spectrogram = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path, self.char2ind)
        return spectrogram, transcript, transcript_path

    def __len__(self):
        return self.size

class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)

#globally Accessible lists
swear_words = ['nigga', 'niggas', 'shit', 'bitch', 'damn', 'fuck', 'fuckin', 'fucking', 'motherfuckin', 'motherfucking']
filler_words = ['um', 'uh', 'mm', 'hm', 'ooh', 'woo', 'mhm', 'mm-hm''huh', 'ha']

pre_cardinal = ['N', 'E', 'S', 'W', 'NE', 'NW', 'SE', 'SW']
post_cardinal = ['North', 'East', 'South', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest']

pre_list = ['cuz', 'ok', 'o', 'till', 'yup', 'imma', 'mister', 'doctor',
            'gonna', 'tryna',
           'carryout', 'sawmill', 'highschool', 'worldclass',
           'saint', 'street', 'state',
            'avenue', 'road', 'boulevard',
           'theatre', 'neighbour', 'neighbours', 'neighbourhood', 'programme']
post_list = ['cause', 'okay', 'oh', 'til', 'yep', 'ima', 'mr', 'dr',
             'going to', 'trying to',
            'carry out', 'saw mill', 'high school', 'world class',
             'st', 'st', 'st',
             'ave', 'rd', 'blvd',
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

def clean_within_all_lambda(text):
    # fix spacing in certain spellings
    text = re.sub('T V','TV',''.join(text))
    text = re.sub('D C','DC',''.join(text))

    # remove remaining floating non-linguistic words
    single_paren = ['<','>', '(',')', '{','}','[',']']
    for paren in single_paren:
        linguistic_words  = [word for word in text.split() if paren not in word]
        text = ' '.join(linguistic_words)

    # general string cleaning
    text = re.sub(r"([a-z])\-([a-z])", r"\1 \2", text , 0, re.IGNORECASE) # replace inter-word hyphen with space
    #DO NO REMOVE APOSTROPHES
    text =re.sub(r'[^\s\w$]|_', ' ',text) # replace special characters with space, except $
    text = re.sub("\s+"," ",''.join(text)) # standardize whitespace

    # update numeric numbers to strings and remove $
    text = re.sub("ft ²", "square feet", ''.join(text))
    text = fix_numbers(text)
    text = re.sub("\$",'dollars',''.join(text))
    text = re.sub("\£",'pounds',''.join(text))

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
    text = fix_state_abbrevs(text)
    text = text.lower()

    # update spacing in certain spellings
    spacing_list_pre = ['north east', 'north west', 'south east', 'south west', 'all right']
    spacing_list_post = ['northeast', 'northwest', 'southeast', 'southwest', 'alright']
    for i in range(len(spacing_list_pre)):
        text = re.sub(spacing_list_pre[i], spacing_list_post[i],''.join(text))

    # remove filler words and swear words
    remove_words = swear_words + filler_words
    resultwords  = [word for word in text.split() if word not in remove_words]
    result = ' '.join(resultwords)

    return result.upper()
