#!/usr/bin/env bash
# Download pretrained base model

set -euo pipefail

mkdir -p models
cd models
wget -nc https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar
tar -xvf scibert_scivocab_uncased.tar
cd scibert_scivocab_uncased
tar -xvf weights.tar.gz

