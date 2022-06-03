#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate dpi-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --save_dir $DATA/artifact"

PYARGS="$PYARGS --data $DATA/datasets/marson_prepped_no-ood.h5ad"
PYARGS="$PYARGS --perturbation_key perturbation"
PYARGS="$PYARGS --covariate_keys celltype donor stim"

PYARGS="$PYARGS --max_epochs 1000"
PYARGS="$PYARGS --loss_ae gauss"

python cpa/train.py $PYARGS
