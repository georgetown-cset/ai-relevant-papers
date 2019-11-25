#!/usr/bin/env bash

gsutil -m cp -nr $GCS_DATA_PREFIX/arxiv-202002\* data/train

