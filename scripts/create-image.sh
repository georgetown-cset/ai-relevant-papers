#!/usr/bin/env bash

set -euo

SOURCEDISK=$1
DATE=`date +"%Y%m%d"`

gcloud compute images create \
  scibert-$DATE \
  --project=$GCP_PROJECT \
  --family=scibert \
  --source-disk=$SOURCEDISK \
  --source-disk-zone=us-east1-c \
  --storage-location=us

