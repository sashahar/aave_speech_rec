import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy
import json
import re
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

class AudioDataset(Dataset):
    def __init__(self, manifest_filepath, char_vocab_path, use_mfcc_features = False, audio_conf = audio_conf):
        super().__init__()

        with open(manifest_filepath) as f:
            data_filepaths = f.readlines()
        data_filepaths = [x.strip().split(',') for x in data_filepaths]
        self.data_filepaths = data_filepaths
        self.size = len(data_filepaths)
        self.char2ind = self.create_char2ind_map(char_vocab_path)
        self.audio_conf = audio_conf
        self.use_mfcc_features = use_mfcc_features

    '''Takes in a file path to the character vocab and returns a dictionary
    mapping characters to indices'''
    def create_char2ind_map(self, char_vocab_path):
        with open(char_vocab_path) as char_vocab_file:
            characters = str(''.join(json.load(char_vocab_file)))
        char2ind = dict([(characters[i], i) for i in range(len(characters))])
        return char2ind

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

    #From https://github.com/MyrtleSoftware/deepspeech/blob/master/src/deepspeech/data/preprocess.py
    """Add context frames to each step in the original signal.
    Args:
        n_context: Number of context frames to add to frame in the original
            signal.
    Example:
        >>> signal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> n_context = 2
        >>> print(add_context_frames(signal, n_context))
        [[0 0 0 0 0 0 1 2 3 4 5 6 7 8 9]
         [0 0 0 1 2 3 4 5 6 7 8 9 0 0 0]
         [1 2 3 4 5 6 7 8 9 0 0 0 0 0 0]]
    """

    def add_context_frames(self, signal, n_context):
        """
        Args:
            signal: numpy ndarray with shape (steps, features).
        Returns:
            numpy ndarray with shape:
                (steps, features * (n_context + 1 + n_context))
        """
        # Pad to ensure first and last n_context frames in original signal have
        # at least n_context frames to their left and right respectively.
        steps, features = signal.shape
        padding = np.zeros((n_context, features), dtype=signal.dtype)
        signal = np.concatenate((padding, signal, padding))

        window_size = n_context + 1 + n_context
        strided_signal = np.lib.stride_tricks.as_strided(
            signal,
            # Shape of the new array.
            (steps, window_size, features),
            # Strides of the new array (bytes to step in each dim).
            (signal.strides[0], signal.strides[0], signal.strides[1]),
            # Disable write to prevent accidental errors as elems share memory.
            writeable=False)

        # Flatten last dim and return a copy to permit writes.
        return strided_signal.reshape(steps, -1).copy()

    #Takes in an audio path and returns a spectrogram
    #Spectrogram is a num_seconds * max_frequency list of lists (Not sure this is correct?)
    def parse_audio(self, audio_path):
        sound = self.load_audio(audio_path)
        if self.use_mfcc_features:
            features = self.get_mfcc_features(sound) # n_mfcc * T
            features = np.transpose(features)
            features = self.add_context_frames(features, 9) #Context window of length 9
            features = (features - features.mean()) / features.std() #Normalize
            features = torch.FloatTensor(np.transpose(features))
        else:
            features = self.get_spectrogram(sound) # D * T
        return features

    '''Takes in a transcript path and the char2ind map and returns an array of
    character indices representing the transcript'''
    def parse_transcript(self, transcript_path, char2ind):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
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

