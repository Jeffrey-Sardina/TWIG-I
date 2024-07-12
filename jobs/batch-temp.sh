#!/bin/bash

datasets="CoDExSmall"
epochs=10
npp=100
lr=5e-3
hyp_validation_mode=1
batch_size=64
batch_size_test=64
loss_fn="margin-ranking(0.1)"
sampler="simple"
version="base"
normalisation="zscore"
optimiser="adam"

## do a baseline run
fts_blacklist="None"
tag="standard-run"
./train-eval.sh \
    $datasets \
    $epochs \
    $npp \
    $lr \
    $hyp_validation_mode \
    $batch_size \
    $batch_size_test \
    $loss_fn \
    "$fts_blacklist" \
    $sampler \
    $version \
    $normalisation \
    $optimiser \
    $tag
