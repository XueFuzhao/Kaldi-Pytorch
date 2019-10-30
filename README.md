
## This project aims to simplify using Kaldi and pytorch for rnn based speech recognition. I'm still working on complete it
Following the general design of pytorch, our lib package can be extended easily to implement other new RNN models.

## Installation

### Prerequisites

* Compiled Kaldi instance ([instructions](https://github.com/kaldi-asr/kaldi/blob/master/INSTALL))
* Install Anaconda and create the environment with python 3.7, pytorch 1.3.

## Major Scripts 

* **`steps_sru/lib`**: package for basic functionality of neural networks and the implementations of RNN models
* **`steps_sru/dataGenSequences_cxt_nolap_fbank.py`**:  data iterater
* **`steps_sru/train_sru_nolap_12layer_1024_fbank.py`**: acoustic modeling
* **`steps_sru/decode_myseq_nolap_sru_dnn_fbank.sh`**:  decoder
* **`steps_sru/nnet-forward-myseq_nolap_sru_dnn.py`**:  HMM state posterior probability estimator

## Usage

1. Call the script from kaldi first: egs/dataset/s5/run.sh
2. Call the script to generate Fbank features: steps/make_fbank.sh
3. Call the script to perform speaker level MVN: steps/compute_cmvn_stats.sh
4. Call the script: rnn_kt_sru_12layer.sh

