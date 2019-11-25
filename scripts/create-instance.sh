#!/usr/bin/env bash

set -euo pipefail

name=$1
gcloud beta compute \
  --project=$GCP_PROJECT \
  instances create $name \
  --zone=us-east1-c \
  --machine-type=custom-4-65536-ext \
  --subnet=default \
  --network-tier=PREMIUM \
  --maintenance-policy=TERMINATE \
  --service-account=$SERVICE_ACCOUNT \
  --scopes=https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write \
  --accelerator=type=nvidia-tesla-p100,count=1 \
  --image=$GCE_IMAGE \
  --image-project=$GCP_PROJECT \
  --boot-disk-size=1024GB \
  --boot-disk-type=pd-standard \
  --boot-disk-device-name=$name \
  --reservation-affinity=any


