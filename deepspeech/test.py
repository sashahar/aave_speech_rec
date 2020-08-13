import argparse

import numpy as np
import torch
from tqdm import tqdm
import json
import os

from dataloader import AudioDataLoader, AudioDataset, BucketingSampler
from decoder import GreedyDecoder, BeamCTCDecoder

parser = argparse.ArgumentParser(description='DeepSpeech testing')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--lm-path',
                    help='path to ARPA-format language model', default=None)
parser.add_argument('--alpha', default=0.0, type=float, help='Language model weight')
parser.add_argument('--beta', default=0.0, type=float, help='Bonus weight for words')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for testing')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--model-path', default='models/deepspeech_final.pth', help='Path to model file created by training')
parser.add_argument('--cuda', action="store_true", help='Use cuda')
parser.add_argument('--id', type=str, help='Unique identifier')
parser.add_argument('--eval-id', type=str, help='Unique identifier for evaluation run')
parser.add_argument('--char-vocab-path', default="character_vocab.json", help='Contains all characters for transcription')
parser.add_argument('--beam-decode', action='store_true',
                    help='Type of decoder to use in model evaluation: Options are greedy decoding and beam search decoding.')
parser.add_argument('--adversarial', action='store_true',
                    help='Type of decoder to use in model evaluation: Options are greedy decoding and beam search decoding.')
parser.add_argument('--log-dir', default='logs',
                    help='Specify absolute path to log directory.  Relative paths will originate in deepspeech dir.')
parser.add_argument('--mfcc', action="store_true", help='Use cuda')

SAVE_TXT_FILE = 'word_preds'
SAVE_SUMMARY_FILE = 'summary_stats'
SAVE_OUTPUT_FILE = 'output_data'
RESULTS_DIR = 'results'

def evaluate(test_loader, device, model, decoder, target_decoder, save_output=False):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    output_text = ['filepath,prediction,target'] #adds header
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):

        #wrap in try/catch statement
        #try:
        inputs, targets, input_sizes, target_sizes, filenames = data

        total_length = max(input_sizes).item()

        inputs = inputs.to(device)
        targets = targets.to(device)

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out, output_sizes = model(inputs, input_sizes, total_length)

        decoded_output, _ = decoder.decode(out, output_sizes)
        #print(decoded_output)
        target_strings = target_decoder.convert_to_strings(split_targets)
        #print(target_strings)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.detach().cpu().numpy(), output_sizes.detach().cpu().numpy(), target_strings))
            #adjustment: add decoded output to list in format ("model output", "target output")
            for j in range(len(target_strings)):
                curr_file = filenames[j].split("/")[-1]
                output_text.append("{},{},{}".format(curr_file, decoded_output[j][0], target_strings[j][0]))

        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))
        # except:
        #     print("Encountered CUDA memory error")
        #     print(torch.cuda.memory_summary())

    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    #adjustment: added output_text as an additional return value
    return wer * 100, cer * 100, output_data, output_text

def evaluate_adversarial(test_loader, device, model, decoder, target_decoder):
    model.eval()
    total_cer, total_wer, generic_cer, generic_wer, accent_cer, accent_wer, num_tokens, num_chars, generic_num_tokens, generic_num_chars, accent_num_tokens, accent_num_chars = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    generic_cer_vec, generic_wer_vec, accent_cer_vec, accent_wer_vec = [], [], [], []
    output_data = []
    output_text = ['filepath,prediction,target'] #adds header
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):

        try:

            inputs, targets, accent_label, input_sizes, target_sizes, filenames = data

            inputs = inputs.to(device)
            targets = targets.to(device)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            out, output_sizes = model(inputs, input_sizes)

            decoded_output, _ = decoder.decode(out, output_sizes)
            #print(decoded_output)
            target_strings = target_decoder.convert_to_strings(split_targets)
            #print(target_strings)

            # add output to data array, and continue
            output_data.append((out.cpu().numpy(), output_sizes.numpy(), target_strings))
            #adjustment: add decoded output to list in format ("model output", "target output")
            for j in range(len(target_strings)):
                curr_file = filenames[j].split("/")[-1]
                output_text.append("{},{},{}".format(curr_file, decoded_output[j][0], target_strings[j][0]))

            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                label_inst = accent_label[x]
                wer_inst = decoder.wer(transcript, reference)
                cer_inst = decoder.cer(transcript, reference)
                total_wer += wer_inst
                total_cer += cer_inst
                num_tokens += len(reference.split())
                num_chars += len(reference.replace(' ', ''))
                if label_inst == 0:
                    generic_wer += wer_inst
                    generic_cer += cer_inst
                    generic_num_tokens += len(reference.split())
                    generic_num_chars += len(reference.replace(' ', ''))
                    generic_wer_vec.append(float(wer_inst)/len(reference.split()))
                    generic_cer_vec.append(float(cer_inst)/len(reference.replace(' ', '')))
                else:
                    accent_wer += wer_inst
                    accent_cer += cer_inst
                    accent_num_tokens += len(reference.split())
                    accent_num_chars += len(reference.replace(' ', ''))
                    accent_wer_vec.append(float(wer_inst)/len(reference.split()))
                    accent_cer_vec.append(float(cer_inst)/len(reference.replace(' ', '')))
        except:
            print("Encountered CUDA memory error")
            print(torch.cuda.memory_summary())

    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    generic_wer = float(generic_wer) / generic_num_tokens
    generic_cer = float(generic_cer) / generic_num_chars
    accent_wer = float(accent_wer) / accent_num_tokens
    accent_cer = float(accent_cer) / accent_num_chars
    print("Inside test func:")
    generic_n = len(generic_wer_vec)
    accent_n = len(accent_wer_vec)
    print('Generic WER confidence interval:({}, {}) mean: {}'.format(np.mean(np.array(generic_wer_vec)) - 1.96*np.std(np.array(generic_wer_vec))/np.sqrt(generic_n), np.mean(np.array(generic_wer_vec)) + 1.96*np.std(np.array(generic_wer_vec))/np.sqrt(generic_n), np.mean(np.array(generic_wer_vec))))
    print('Generic CER confidence interval:({}, {}) mean: {}'.format(np.mean(np.array(generic_cer_vec)) - 1.96*np.std(np.array(generic_cer_vec))/np.sqrt(generic_n), np.mean(np.array(generic_cer_vec)) + 1.96*np.std(np.array(generic_cer_vec))/np.sqrt(generic_n), np.mean(np.array(generic_cer_vec))))
    print('Accent WER confidence interval:({}, {}) mean: {}'.format(np.mean(np.array(accent_wer_vec)) - 1.96*np.std(np.array(accent_wer_vec))/np.sqrt(accent_n), np.mean(np.array(accent_wer_vec)) + 1.96*np.std(np.array(accent_wer_vec))/np.sqrt(accent_n), np.mean(np.array(accent_wer_vec))))
    print('Accent CER confidence interval:({}, {}) mean: {}'.format(np.mean(np.array(accent_cer_vec)) - 1.96*np.std(np.array(accent_cer_vec))/np.sqrt(accent_n), np.mean(np.array(accent_cer_vec)) + 1.96*np.std(np.array(accent_cer_vec))/np.sqrt(accent_n), np.mean(np.array(accent_cer_vec))))

    #adjustment: added output_text as an additional return value
    return wer * 100, cer * 100, generic_wer * 100, generic_cer * 100, accent_wer * 100, accent_cer * 100,  output_data, output_text

def load_saved_model(args):
    print("Loading model %s" % args.model_path)
    package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)

    #if not args.finetune:  # Don't want to restart training
    optim_state = package['optim_dict']
    start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
    start_iter = package.get('iteration', None)
    if start_iter is None:
        start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
        start_iter = 0
    else:
        start_iter += 1
    avg_loss = int(package.get('avg_loss', 0))
    return model, optim_state, start_epoch, start_iter, avg_loss

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mfcc:
        from model_mfcc import DeepSpeech
    else:
        from model import DeepSpeech
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Using device: ", device)
    model, optim_state, start_epoch, _, avg_loss = load_saved_model(args)
    model = model.to(device)

    with open(args.char_vocab_path) as label_file:
        characters = str(''.join(json.load(label_file)))

    if not args.beam_decode:
        decoder = GreedyDecoder(characters)
    else:
        decoder = BeamCTCDecoder(characters, lm_path = args.lm_path, alpha = args.alpha, beta = args.beta)

    target_decoder = GreedyDecoder(characters)

    test_dataset = AudioDataset(manifest_filepath=args.test_manifest, char_vocab_path=args.char_vocab_path)
    test_sampler = BucketingSampler(test_dataset, batch_size=args.batch_size)
    test_loader = AudioDataLoader(test_dataset, batch_sampler=test_sampler)

    if args.adversarial:
        wer, cer, generic_wer, generic_cer, accent_wer, accent_cer, output_data, output_text = evaluate_adversarial(test_loader=test_loader, device=device, model=model, decoder=decoder, target_decoder=target_decoder)
        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'
              'Generic WER {generic_wer:.3f}\t'
              'Generic CER {generic_cer:.3f}\t'
              'Accent WER {accent_wer:.3f}\t'
              'Accent CER {accent_cer:.3f}\t'.format(wer=wer, cer=cer, generic_wer=generic_wer, generic_cer=generic_cer, accent_wer=accent_wer, accent_cer=accent_cer))
    else:
        wer, cer, output_data, output_text = evaluate(test_loader=test_loader, device=device, model=model, decoder=decoder, target_decoder=target_decoder)
        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))


    args.log_dir += "/" + args.id

    save_word_preds_file = args.log_dir + "/" + RESULTS_DIR  + "/"+ SAVE_TXT_FILE + "_" + args.eval_id + ".csv"
    save_summary_file = args.log_dir + "/" + RESULTS_DIR  + "/"+ SAVE_SUMMARY_FILE + "_" + args.eval_id + ".csv"
    save_output_file = args.log_dir + "/" + RESULTS_DIR  + "/"+ SAVE_OUTPUT_FILE + "_" + args.eval_id + ".pt"

    if not os.path.exists(args.log_dir + "/" + RESULTS_DIR):
        os.mkdir(args.log_dir  + "/" + RESULTS_DIR)

    print('Saving predictions to: {}'.format(save_word_preds_file))
    np.savetxt(save_word_preds_file, output_text, fmt="%s", delimiter=",")
    print('Saving summary stats to: {}'.format(save_summary_file))
    with open(save_summary_file, "w") as file:
        file.write('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
    print('Saving output data to: {}'.format(save_output_file))
    torch.save(output_data, save_output_file)
