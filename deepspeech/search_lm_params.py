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
parser.add_argument('--lm-alpha-to', default=3.0, type=float, help='Language model weight end tuning')
parser.add_argument('--lm-beta-from', default=0.0, type=float, help='Language model word bonus (all words) start tuning')
parser.add_argument('--lm-beta-to', default=0.5, type=float, help='Language model word bonus (all words) end tuning')
parser.add_argument('--lm-num-alphas', default=45, type=int, help='Number of alpha candidates for tuning')
parser.add_argument('--lm-num-betas', default=8, type=int, help='Number of beta candidates for tuning')
parser.add_argument('--output-path', help="Where to save tuning results")
parser.add_argument('--lm-path', help='path to ARPA-format language model', default=None)
parser.add_argument('--char-vocab-path', default="character_vocab.json", help='Contains all characters for transcription')
parser.add_argument('--num-workers', default=16, type=int, help='Number of parallel decodes to run')

args = parser.parse_args()

if args.lm_path is None:
	print("error: LM must be provided for tuning")
	sys.exit(1)

with open(args.char_vocab_path) as label_file:
	characters = str(''.join(json.load(label_file)))

saved_output = torch.load(args.saved_output)

def init(characters, blank_index, lm_path):
	global decoder
	decoder = BeamCTCDecoder(characters, lm_path=lm_path,  blank_index=blank_index)

def decode_dataset(params):
	lm_alpha, lm_beta = params
	global decoder
	decoder._decoder.reset_params(lm_alpha, lm_beta)

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

	p = Pool(args.num_workers, init, [characters, characters.index('_'), args.lm_path])

	cand_alphas = np.linspace(args.lm_alpha_from, args.lm_alpha_to, args.lm_num_alphas)
	cand_betas = np.linspace(args.lm_beta_from, args.lm_beta_to, args.lm_num_betas)
	params_grid = [(float(alpha), float(beta)) for alpha in cand_alphas
		for beta in cand_betas]

	scores = []

	for params in tqdm(p.imap(decode_dataset, params_grid), total=len(params_grid)):
		scores.append(list(params))

	print("Saving tuning results to: {}".format(args.output_path))
	with open(args.output_path, "w") as output_file:
		json.dump(scores, output_file)