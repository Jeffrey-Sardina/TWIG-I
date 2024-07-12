#!/bin/bash

cd ../

# TWIG-I hyp search
epochs=20
sampler='simple'
version="base"
normalisation="zscore"
optimiser="adam"
hyp_validation_mode=1
batch_size_test=64
tag="example-hypsearch"
fts_blacklist="None"

for datasets in CoDExSmall
do
    for margin in 0.01 0.1 0.5
    do
        for batch_size in 64 128 256
        do
            for lr in 5e-3 5e-4 5e-5
            do
                for npp in 30 100 500
                do
                    loss_fn="margin-ranking($margin)"
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
            done
        done
    done
done
