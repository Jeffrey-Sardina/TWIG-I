#!/bin/bash

cd ../src/

# gather data
datasets=$1 #delimit by _; i.e. "CoDExSmall_UMLS"
epochs=$2
npp=$3
lr=$4
hyp_validation_mode=$5
batch_size=$6
batch_size_test=$7
loss_fn=$8
fts_blacklist=$9 # delimit by spaces; i.e. "A B C"
sampler=${10} #simple, vector
version=${11} #base, linear
normalisation=${12} #zscore, minmax, none
optimiser=${13} #adam, adagrad, etc
tag=${14}

# literature-standard settings
use_train_filter='0'
use_valid_and_test_filters='1'

# config env
out_file="rec_v${version}_${datasets}_e${epochs}_tag-${tag}.log"

# run training
start=`date +%s`
python -u run_exp.py \
    $version \
    $datasets \
    $epochs \
    $lr \
    $optimiser \
    $normalisation \
    $batch_size \
    $batch_size_test \
    $npp \
    $use_train_filter \
    $use_valid_and_test_filters \
    $sampler \
    $loss_fn \
    "$fts_blacklist" \
    $hyp_validation_mode \
    &> $out_file

# output final stats
end=`date +%s`
runtime=$((end-start))
echo "Experiments took $runtime seconds" &>> $out_file
