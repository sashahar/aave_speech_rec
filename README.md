# AAVE Speech Recognition - Summer 2020
This repository contains the source code for a summer research project on deep learning based speech recognition algorithms.

Inspired by the 2020 paper [Racial disparities in automated speech recognition](https://www.pnas.org/doi/10.1073/pnas.1915768117),
we trained our speech recognition algorithm on the Coraal dataset to see if we could surpass the performance of commercial ASR systems examined by Koenecke et al.


## Datasets
[CORRAL](https://oraal.uoregon.edu/coraal) - Corpus of Regional African American Language
[Voices of California](http://web.stanford.edu/dept/linguistics/VoCal/) - Corpus of speech samples collected across the state of California

Summary of file structure and usage:

## Deepspeech

**train.py**
the main script for training a DeepSpeech model.  Specify a train and test manifest as command line arguments.  
Example usage:
```
python3 train.py --checkpoint --train-manifest ../manifests/train_1000_coraal.csv  --val-manifest ../manifests/val_1000_coraal.csv --batch-size 10  --epochs 20  --cuda  --id v_1
```

**test.py**
Script for conducting inference using a previously trained model.  In order to use, specify path to test manifest, and path to saved model through the command line arguments.  Example usage:
```
python test.py --test-manifest ../manifests/test_manifest.csv  --cuda --id final_results --model-path logs/v_1/saved_models/deepspeech_checkpoint_v1_epoch_91.pth
```

**model.py**
File in which model architecture is defined.  Includes definitions for custom layers, including BatchRNN, MaskConv.  Includes SimpleNN and Simple LSTM, used as adversaries in adversarial architecture.

## Manifests
Folder containing manifest files for model training.  Each manifest is a CSV file where one line contains the file path to a wav_file and corresponding txt_file containing a transcription.  The Deepspeech model reads from these manifests in order to characterize the training and validation datasets.  i.e. to craft a small training dataset of 1000 examples requires creating a manifest with 1000 rows.

Relevant files:

**coraal_manifest.csv**
Comprehensive list of all CORAAL trianing examples

**voc_manifest.csv**
Comprehensive list of all VOC training examples

### process_data_coraal.py
Used to process raw data files from CORAAL dataset.  To use, adjust the constants at the top of the file to point at file path to which data files have been downloaded.  For each CORAAL interview, omits utterances from the interviewer, and splits interviewee utterances into chunks of 20 seconds or less. 

### process_data_voc.py
script for processing raw VOC data from the Stanford Linguistics department.  Supports two kinds of transcript makup schemas.  Splices out utterances from interviewer, and creates speech snippers approximately 20 seconds in length.

### process_transcriptions.py
Used to perform post-processing on interview transcripts for the purpose of calculating Word Error Rate (WER).  Uses regexes to correct irregular spacing, standardize to lowercase, remove non-alphanumeric characters, and correct certain spellings.  Reads transcriptions from a manifest CSV file, and writes a new manifest with added cloumns for the clean text.  Adjust constants at top of file to use.

## Sources
The model architecture used in this project is inspired by the 2014 paper [Deepspeech] (https://arxiv.org/pdf/1412.5567.pdf)
and some of the code (particularly, the CTCDecoders) borrows from this repository: https://github.com/SeanNaren/deepspeech.pytorch
