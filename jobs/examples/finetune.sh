#!/bin/bash

cd ../

final_epoch_state=100
for epochs in 20
do
    # Note: you need to supply your own checkpoint ID and override json file
    # See the README. These are only examples, and will not run
    wn_chkpt_id="chkpt-ID_2980578518512252"
    optimal_config="override-CoDExSmall-e20-bs128-lr5e-3-npp500.json"
    ./from-checkpoint.sh \
        $wn_chkpt_id \
        $final_epoch_state \
        TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
        wn-to-codex-e${epochs}-opt_hyp
done
