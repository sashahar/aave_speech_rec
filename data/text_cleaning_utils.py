import os
# import glob
import pandas as pd
import numpy as np
import re
import inflect
from text2digits import text2digits
from jiwer import wer
# from nltk import ngrams
import math
import collections

t2d = text2digits.Text2Digits()
p = inflect.engine()

def remove_markers(line, markers):
    # Remove any text within markers, e.g. 'We(BR) went' -> 'We went'
    # markers = list of pairs, e.g. ['()', '[]'] denoting breath or noise in transcripts
    for s, e in markers:
         line = re.sub(" ?\\" + s + "[^" + e + "]+\\" + e, "", line)
    return line

# Standardize state abbreviations
states = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}

def fix_state_abbrevs(text):
    # Standardize state abbreviations
    ix = 0
    state_result = []
    wordlist = text.split()
    while ix < len(wordlist):
        word = wordlist[ix].lower().capitalize()
        if word in states.keys(): # is this correct check?
            new_word = states[word]
        elif (ix < len(wordlist)-1) and ((word + ' ' + wordlist[ix+1].lower().capitalize()) in states.keys()):
            new_word = states[(word + ' ' + wordlist[ix+1].lower().capitalize())]
            ix += 1
        else:
            new_word = word
        state_result.append(new_word)
        ix += 1
    text = ' '.join(state_result)
    return text

def fix_numbers(text):
    # Standardize number parsing and dollars
    split_words_num = text.split()
    new_list = []
    for i in range(len(split_words_num)):
        x = split_words_num[i]

        # deal with years
        if x.isdigit():
            if (1100 <= int(x) < 2000) or (2010 <= int(x) < 2100) or (int(x) == 5050):
                # deal with years as colloquially spoken
                new_word = p.number_to_words(x[:2]) + " " + p.number_to_words(x[2:])
            elif "and" in p.number_to_words(x):
                # remove 'and' from e.g. 'four hundred and ninety five'
                output = p.number_to_words(x)
                resultwords  = [word for word in output.split() if word not in ['and']]
                new_word = ' '.join(resultwords)
            else:
                new_word = p.number_to_words(x)

        # deal with cases like 1st, 2nd, etc.
        elif re.match(r"(\d+)(\w+)", x, re.I):
            single_digits = ['1st', '2nd', '3rd', '5th', '8th', '9th']
            double_digits = ['12th']
            single_num = ['1', '2', '3', '5', '8', '9']
            double_num = ['12']
            single_digit_labels = ['first', 'second', 'third', 'fifth', 'eighth', 'ninth']
            double_digit_labels = ['twelfth']
            all_digits = single_digits + double_digits
            all_labels = single_digit_labels + double_digit_labels
            if x in all_digits:
                new_word = all_labels[all_digits.index(x)]
            else:
                items = re.match(r"(\d+)(\w+)", x, re.I).groups()
                if (items[1] not in ['s', 'th', 'st', 'nd', 'rd']):
                    new_word = fix_numbers(items[0]) + " " + items[1]
                elif (items[0][-2:] in double_num):
                    new_word = fix_numbers(str(100*int(items[0][:-2]))) + " " + fix_numbers(items[0][-2:]+items[1])
                elif ((items[0][-1:] in single_num) and items[0][-2:-1] != '1'):
                    try:
                        new_word = fix_numbers(str(10*int(items[0][:-1]))) + " " + fix_numbers(items[0][-1:]+items[1])
                    except:
                        new_word = fix_numbers(items[0]) + items[1]
                # deal with case e.g. 80s
                elif (items[1] in ['s', 'th']) and (p.number_to_words(items[0])[-1] == 'y'):
                    new_word = fix_numbers(items[0])[:-1] + "ie" + items[1]
                else:
                    new_word = fix_numbers(items[0]) + items[1]

        # deal with dollars
        elif re.match(r"\$[^\]]+", x, re.I):
            # deal with $ to 'dollars'
            money = fix_numbers(x[1:])
            if x[1:] in ["1", "a"]:
                new_word = money + " dollar"
            else:
                new_word = money + " dollars"

        elif re.match(r"\£[^\]]+", x, re.I):
            # deal with £ to 'pounds'
            money = fix_numbers(x[1:])
            if x[1:] in ["1", "a"]:
                new_word = money + " pound"
            else:
                new_word = money + " pounds"

        else:
            new_word = x

        new_list.append(new_word)

    text = ' '.join(new_list)

    # Deal with written out years (two thousand and ten -> twenty ten)
    for double_dig in range(10, 100):
        double_dig_str = p.number_to_words(double_dig)
        text = re.sub('two thousand and ' + double_dig_str, 'twenty ' + double_dig_str, text.lower())
        text = re.sub('two thousand ' + double_dig_str, 'twenty ' + double_dig_str, text.lower())

    # Change e.g. 101 to 'one oh one' -- good for area codes
    single_dig_list = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for j in single_dig_list:
        text = re.sub('thousand and ' + j, 'thousand ' + j, text.lower())
        for k in single_dig_list:
            #print(j + ' hundred ' + k)
            text = re.sub(j + ' hundred ' + k + ' ', j + ' oh ' + k + ' ', text.lower())
            text = re.sub(j + ' hundred ' + k + '$', j + ' oh ' + k, text.lower())

    text = re.sub("\s+"," ",''.join(text)) # standardize whitespace

    return text
