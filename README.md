## Overview

This repo contains code for a [CSET](https://cset.georgetown.edu/) project that uses `SciBERT` to identify the universe of research publications relating to the application and development of artificial intelligence.
`SciBERT` is a project of [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org) described in [SciBERT: Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676).

## Python

Create a new Python 3.7 environment, e.g. via Conda

```
conda create -n scibert python=3.7
conda activate scibert
```

Clone the repo *with its submodules*:

```
git clone --recurse-submodules -j2 https://github.com/georgetown-cset/ai-relevant-papers.git
```

Set up the environment. 

```
cd ai-relevant-papers
pip install -r replication-requirements.txt
```

SciBERT [requires a fork of allennlp](https://github.com/allenai/scibert/issues/65). Be sure to use it:

```
pip install -e git://github.com/ibeltagy/allennlp@fp16_and_others#egg=allennlp
```

If training models on GPUs, install NVIDIA's [apex](https://github.com/NVIDIA/apex).

```
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

It's possible to install apex on a system without a GPU: `pip install -v --no-cache-dir ./`.
You probably don't want to do this.
In prediction, availability of apex will cause allennlp or scibert to try to load CUDA extensions, which will fail with `ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`.

## Model training

We're using these driver and Cuda versions:

```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Fri_Feb__8_19:08:17_PST_2019
Cuda compilation tools, release 10.1, V10.1.105

$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  418.67  Sat Apr  6 03:07:24 CDT 2019
GCC version:  gcc version 6.3.0 20170516 (Debian 6.3.0-18+deb9u1)
```

Refer to the arXiv directory for code and documentation of preprocessing.
An example of training data is included in `data/train/arxiv-20200213-ai-binary-10pct-sample`.
To use the example data, decompress it.

```
cd data/train/arxiv-20200213-ai-binary-10pct-sample && gzip -dk *.gz
```

Download the pretrained SciBERT model.

```
./scripts/download-pretrained-scibert.sh
```

To train a model, pass the name of the directory under `data/train` containing the training data, like

```
scripts/train-baseline.sh arxiv-20200213-ai-binary-10pct-sample
```

If training a model halts unexpectedly, you can resume from the most recent checkpoint like so:

```
export MODEL_DIR="example_dir"
python -m allennlp.run train $MODEL_DIR/config.json -s $MODEL_DIR --include-package scibert --recover 
```

The remainder of this readme is from the SciBERT repo at the time of our fork.
See the [SciBERT repo](https://github.com/allenai/scibert) for the latest. 

# <p align=center>`SciBERT`</p>
`SciBERT` is a `BERT` model trained on scientific text.

* `SciBERT` is trained on papers from the corpus of [semanticscholar.org](https://semanticscholar.org). Corpus size is 1.14M papers, 3.1B tokens. We use the full text of the papers in training, not just abstracts.

* `SciBERT` has its own vocabulary (`scivocab`) that's built to best match the training corpus. We trained cased and uncased versions. We also include models trained on the original BERT vocabulary (`basevocab`) for comparison.

* It results in state-of-the-art performance on a wide range of scientific domain nlp tasks. The details of the evaluation are in the [paper](https://arxiv.org/abs/1903.10676). Evaluation code and data are included in this repo. 

### Downloading Trained Models
We release the tensorflow and the pytorch version of the trained models. The tensorflow version is compatible with code that works with the model from [Google Research](https://github.com/google-research/bert). The pytorch version is created using the [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT) library, and this repo shows how to use it in AllenNLP.  All combinations of `scivocab` and `basevocab`, `cased` and `uncased` models are available below. Our evaluation shows that `scivocab-uncased` usually gives the best results.

#### Tensorflow Models
* __[`scibert-scivocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz) (Recommended)__
* [`scibert-scivocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_cased.tar.gz)
* [`scibert-basevocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_basevocab_uncased.tar.gz)
* [`scibert-basevocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_basevocab_cased.tar.gz)

#### PyTorch AllenNLP Models
* __[`scibert-scivocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar) (Recommended)__
* [`scibert-scivocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar)
* [`scibert-basevocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_uncased.tar)
* [`scibert-basevocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_cased.tar)

#### PyTorch HuggingFace Models
* __[`scibert-scivocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar) (Recommended)__
* [`scibert-scivocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_cased.tar)
* [`scibert-basevocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_basevocab_uncased.tar)
* [`scibert-basevocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_basevocab_cased.tar)

### Using SciBERT in your own model

SciBERT models include all necessary files to be plugged in your own model and are in same format as BERT.
If you are using Tensorflow, refer to Google's [BERT repo](https://github.com/google-research/bert) and if you use PyTorch, refer to [Hugging Face's repo](https://github.com/huggingface/pytorch-pretrained-BERT) where detailed instructions on using BERT models are provided. 

### Training new models using AllenNLP

To run experiments on different tasks and reproduce our results in the [paper](https://arxiv.org/abs/1903.10676), you need to first setup the Python 3.6 environment:

```pip install -r requirements.txt```

which will install dependencies like [AllenNLP](https://github.com/allenai/allennlp/).

Use the `scibert/scripts/train_allennlp_local.sh` script as an example of how to run an experiment (you'll need to modify paths and variable names like `TASK` and `DATASET`).

We include a broad set of scientific nlp datasets under the `data/` directory across the following tasks. Each task has a sub-directory of available datasets.
```
├── ner
│   ├── JNLPBA
│   ├── NCBI-disease
│   ├── bc5cdr
│   └── sciie
├── parsing
│   └── genia
├── pico
│   └── ebmnlp
└── text_classification
    ├── chemprot
    ├── citation_intent
    ├── mag
    ├── rct-20k
    ├── sci-cite
    └── sciie-relation-extraction
```

For example to run the model on the Named Entity Recognition (`NER`) task and on the `BC5CDR` dataset (BioCreative V CDR), modify the `scibert/train_allennlp_local.sh` script according to:
```
DATASET='bc5cdr'
TASK='ner'
...
```

Decompress the PyTorch model that you downloaded using  
`tar -xvf scibert_scivocab_uncased.tar`  
The results will be in the `scibert_scivocab_uncased` directory containing two files:
A vocabulary file (`vocab.txt`) and a weights file (`weights.tar.gz`).
Copy the files to your desired location and then set correct paths for `BERT_WEIGHTS` and `BERT_VOCAB` in the script:
```
export BERT_VOCAB=path-to/scibert_scivocab_uncased.vocab
export BERT_WEIGHTS=path-to/scibert_scivocab_uncased.tar.gz
```

Finally run the script:

```
./scibert/scripts/train_allennlp_local.sh [serialization-directory]
```

Where `[serialization-directory]` is the path to an output directory where the model files will be stored. 

### Citing

If you use `SciBERT` in your research, please cite [SciBERT: Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676).
```
@inproceedings{Beltagy2019SciBERT,
  title={SciBERT: Pretrained Language Model for Scientific Text},
  author={Iz Beltagy and Kyle Lo and Arman Cohan},
  year={2019},
  booktitle={EMNLP},
  Eprint={arXiv:1903.10676}
}
```

`SciBERT` is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.




