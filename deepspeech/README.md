# Deepspeech

This folder contains the code related to defining, training, and running inference with the deep learning model.

# Dataset and Data Loader
The input data is formatted as .wav files - the Dataset class processes these wave files into spectrograms,
    which in turn are fed as input to the network.

# Model
At its core, the Deepspeech architecture is a deep, bidirectional LSTM.  It takes spectrograms as input, extracts features
from the spectrograms through a series of convolutions, and then uses stacked LSTM layers to predict a character at each time
interval.

# Training
During training, the model is optimized with CTC Loss, a standard choice in deep speech recognition literature
because phonemes can take up several windows depending on the speaker.

Example command to run the training script:
```shell
python3 train.py --checkpoint --train-manifest ../manifests_slurm/coraal_train_manifest.csv  --val-manifest ../manifests_slurm/coraal_val_manifest.csv --batch-size 10  —hidden—dim 548 epochs 2 --cuda --id coraal_hidden_548_lr1e-3
```

# Testing & Inference
During test time, the raw output of the model is also processes by a `CTCBeamDecoder` which uses a combination of beam search
and a word-level language model to find the most probable sequence of characters during inference.  The strength of the
language model is controlled by two hyperparameters alpha and beta, which are tuned using the scripts `search_lm_params.py`
`select_lm_params.py`

The main model performance metrics reported are Character Error Rate (CER) and Word Error Rate (WER)

# References/Resources
(Beam Search)[https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7]
(Intuitive Guide to Deep Learning for Speech Recognition)[https://towardsdatascience.com/audio-deep-learning-made-simple-automatic-speech-recognition-asr-how-it-works-716cfce4c706]
