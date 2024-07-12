#!/bin/bash

# constants for TWIG-I from-scratch baselines
hyp_validation_mode=1
batch_size_test=64
fts_blacklist="None"
loss_fn="margin-ranking(0.1)"

for epochs in 10
do
    npp=100
    lr=5e-3
    batch_size=64
    dataset="CoDExSmall"
    ./TWIG-I_pipeline.sh \
        $dataset \
        $epochs \
        $npp \
        $lr \
        $hyp_validation_mode \
        $batch_size \
        $batch_size_test \
        $loss_fn \
        $fts_blacklist \
        "$dataset-e${epochs}"
done
