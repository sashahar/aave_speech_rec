#!/usr/bin/env python3

import argparse
import csv
import logging

import Levenshtein as Lev


def wer(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return Lev.distance(''.join(w1), ''.join(w2))


def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)


if __name__ == '__main__':
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0

    parser = argparse.ArgumentParser()
    parser.add_argument('results', help='Input CSV file containing the segment filename, reference transcript, and decoded results. Should have a header.')
    parser.add_argument('--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Display log messages at and above this level.')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format="%(levelname)s: %(message)s")

    with open(args.results, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Code copied from deepspeech/test.py evaluate() function.
            transcript = row['mod9_transcript']
            # Upper case mod9 transcript to match reference casing.
            transcript = transcript.upper()
            reference = row['reference_transcript']
            wer_inst = wer(transcript, reference)
            cer_inst = cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))
    logging.debug("Total WER: %d\nTotal CER: %d\nNumber of tokens: %d\nNumber of chars: %d",
                  total_wer,
                  total_cer,
                  num_tokens,
                  num_chars)
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    logging.info("Average WER: %f\tAverage CER: %f", wer, cer)
