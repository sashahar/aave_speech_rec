# AAVE Speech Recognition - Summer 2020
Summer research project on speech recognition.

Summary of file structure and usage:

## Deepspeech

**train.py**
the main script for training a DeepSpeech model.  Specify a train and test manifest as command line arguments.  Example command:
`python3 train.py --checkpoint --train-manifest ../manifests/train_1000_coraal.csv  --val-manifest ../manifests/val_1000_coraal.csv --batch-size 10  --epochs 20  --cuda  --id v_1`

## Manifests
Folder containing manifest files for model training.  Each manifest is a CSV file where one line contains the file path to a wav_file and corresponding txt_file containing a transcription.  The Deepspeech model reads from these manifests in roder to characterize the training and validation datasets.  i.e. to craft a small training dataset of 1000 examples requires creating a manifest with 1000 rows.

Relevant files:

**coraal_manifest.csv**
Comprehensive list of all CORAAL trianing examples

**voc_manifest.csv**
Comprehensive list of all VOC training examples

### process_data_coraal.py
Used to process raw data files from CORAAL dataset.  To use, adjust the constants at the top of the file to point at file path to which data files have been downloaded.  For each CORAAL interview, omits utterances from the interviewer, and splits interviewee utterances into chunks of 20 seconds or less. 

### process_data_voc.py

### process_transcriptions.py
Used to performn post-processing on interview transcripts fior the purpose of calculating Word Error Rate (WER).  Uses regexes to correct irregular spacing, standardize to lowercase, remove non-alphanumeric characters, and correct certain spellings.  Reads transcriptions from a manifest CSV file, and writes a new manifest with added cloumns for the clean text.  Adjust constants at top of file to use.
