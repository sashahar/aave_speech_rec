import argparse

import numpy as np
import torch
from tqdm import tqdm
import json
import sys
from multiprocessing.pool import Pool

from decoder import GreedyDecoder, BeamCTCDecoder

parser = argparse.ArgumentParser(description='Deepspeech LM Tuning')
parser.add_argument('--saved-output', default="", type=str, help='Path to output from test.py')
parser.add_argument('--lm-alpha-from', default=0.0, type=float, help='Language model weight start tuning')
parser.add_argument('--lm-alpha', default=0.5, type=float, help='Language model weight end tuning')
parser.add_argument('--lm-beta', default=0.5, type=float, help='Language model word bonus (all words)')
parser.add_argument('--lm-path', help='path to ARPA-format language model', default=None)
parser.add_argument('--char-vocab-path', default="character_vocab.json", help='Contains all characters for transcription')

args = parser.parse_args()

if args.lm_path is None:
	print("error: LM must be provided")
	sys.exit(1)

with open(args.char_vocab_path) as label_file:
	characters = str(''.join(json.load(label_file)))

saved_output = torch.load(args.saved_output)

def decode_dataset(lm_alpha, lm_beta, decoder):

	total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
	for out, sizes, target_strings in saved_output:
		decoded_output, _, = decoder.decode(torch.from_numpy(out), torch.from_numpy(sizes))
		for x in range(len(target_strings)):
			transcript, reference = decoded_output[x][0], target_strings[x][0]
			wer_inst = decoder.wer(transcript, reference)
			cer_inst = decoder.cer(transcript, reference)
			total_cer += cer_inst
			total_wer += wer_inst
			num_tokens += len(reference.split())
			num_chars += len(reference.replace(' ', ''))

	wer = float(total_wer) / num_tokens
	cer = float(total_cer) / num_chars

	return [lm_alpha, lm_beta, wer * 100, cer * 100]

if __name__ == '__main__':
	
	decoder = BeamCTCDecoder(characters, lm_path=args.lm_path,  blank_index=characters.index('_'), alpha = args.lm_alpha, beta=args.lm_beta)
	results = decode_dataset(args.lm_alpha, args.lm_beta, decoder)

	print(results)
	print("Alpha: %f \nBeta: %f \nWER: %f\nCER: %f" % tuple(results))