#!/bin/bash

datasets=$1 #delimit by _
epochs=$2
npp=$3
lr=$4 # 5e-4 works for UMLS; 5e-5 is good for Kinships; 
hyp_validation_mode=$5
batch_size=$6
batch_size_test=$7
loss_fn=$8
fts_blacklist=$9 # delimit by spaces; i.e. "A B C"
tag=${10}

sampler="simple"
version="base"
normalisation="zscore"
use_train_filter='0'
use_valid_and_test_filters='1'

export TWIG_CHANNEL=3

cd TWIG-I/
out_file="rec_v${version}_${datasets}_norm-${normalisation}_e${epochs}-lr${lr}_bs${batch_size}_npp${npp}_loss-${loss_fn}_${sampler}-sampler_filter-code${use_train_filter}${use_valid_and_test_filters}_tag-${tag}.log"

start=`date +%s`
python -u run_exp.py \
    $version \
    $datasets \
    $epochs \
    $lr \
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

end=`date +%s`
runtime=$((end-start))
echo "Experiments took $runtime seconds on " &>> $out_file
