#!/bin/bash

cd ../

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
tag="base-run"
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


## do a second baseline (only 2-hop fts)
fts_blacklist="s_deg o_deg p_freq s_p_cofreq o_p_cofreq s_o_cofreq"
tag="2-hop-only"
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


## try removing by elems (the 1-hop fts)
for fts_blacklist in "s_deg" \
    "o_deg" \
    "p_freq" \
    "s_p_cofreq" \
    "o_p_cofreq" \
    "s_o_cofreq"
do
    tag="bl-${fts_blacklist}"
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
done

## try removing by pairs (the 2-hop fts)
for fts in "o_min_deg_neighbour s_min_deg_neighbour" \
    "o_max_deg_neighbour s_max_deg_neighbour" \
    "o_mean_deg_neighbour s_mean_deg_neighbour" \
    "o_num_neighbours s_num_neighbours" \
    "o_min_freq_rel s_min_freq_rel" \
    "o_max_freq_rel s_max_freq_rel" \
    "o_mean_freq_rel s_mean_freq_rel" \
    "o_num_rels s_num_rels"
do
    tag="bl-${fts_blacklist}"
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
done
