#!/usr/bin/env python3
#
# Generates a CSV file with the segment filename, reference transcript, and mod9 onebest.

import argparse
import csv
import json
import logging
import os

FIELDNAMES = ['segment_filename', 'reference_transcript', 'mod9_transcript']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Data directory of processed CORAAL segments. Should have the subdirs wav/ and txt/.')
    parser.add_argument('mod9_results', type=str,
                        help='Mod9 results jsons file.')
    parser.add_argument('output_file', nargs='?', type=str, default='out.csv',
                        help='Output csv filename csv with file name, reference transcript, and mod9 transcript.')
    parser.add_argument('--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='WARNING', help='Display log messages at and above this level.')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format="%(levelname)s: %(message)s")

    with open(args.mod9_results, 'r', newline='') as f:
        with open(args.output_file, 'a') as out:
            writer = csv.DictWriter(out, FIELDNAMES, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for line in f:
                result = json.loads(line)
                segment = os.path.split(result['segment_filename'])[1].split('.')[0]
                row = dict.fromkeys(FIELDNAMES)
                row['segment_filename'] = result['segment_filename']
                row['mod9_transcript'] = result['transcript']
                # Get onebest text.
                with open(os.path.join(args.data_dir, 'txt', f"{segment}.txt"), 'r') as ref:
                    row['reference_transcript'] = ref.read()
                # Write row to the output csv file.
                writer.writerow(row)
