#!/usr/bin/env bash

set -euo pipefail

DATA_PATH='data/train/micro-sample'
export TRAIN_PATH=$DATA_PATH/train.jsonl
export DEV_PATH=$DATA_PATH/dev.jsonl
export TEST_PATH=$DATA_PATH/test.jsonl
export DATASET_SIZE=`wc -l < $DATA_PATH/train.jsonl`
OUTPUT_DIR="models/demo"

rm -rf $OUTPUT_DIR

# Appear to require absolute paths
SCIBERT_PATH=`realpath .`
export BERT_VOCAB=$SCIBERT_PATH/models/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=$SCIBERT_PATH/models/scibert_scivocab_uncased/weights.tar.gz

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

# Corresponds with dataset_reader.token_indexers.bert.do_lowercase
# -> Use 'false' if tokens already lowercased
export IS_LOWERCASE=false

export CUDA_DEVICE=0
export GRAD_ACCUM_BATCH_SIZE=32
export NUM_EPOCHS=75
export LEARNING_RATE=0.00002
export TRUNCATE=1

CONFIG_FILE=allennlp_config/text_classification_local.json

python -m allennlp.run train $CONFIG_FILE  --include-package scibert -s "$OUTPUT_DIR"
