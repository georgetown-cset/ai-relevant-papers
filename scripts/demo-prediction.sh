#!/usr/bin/env bash

set -euo pipefail

export BERT_VOCAB=`realpath models/scibert_scivocab_uncased/vocab.txt`
export BERT_WEIGHTS=`realpath models/scibert_scivocab_uncased/weights.tar.gz`
export IS_LOWERCASE=true
export GRAD_ACCUM_BATCH_SIZE=32
export NUM_EPOCHS=75
export LEARNING_RATE=0.001

allennlp predict \
  models/demo/model.tar.gz \
  data/train/micro-sample/dev.jsonl \
  --output demo_predictions.jsonl \
  --include-package scibert \
  --use-dataset-reader \
  --predictor scibert_predictor \
  --cuda-device -1 \
  --overrides "`cat allennlp_config/text_classification_prediction.json`"
