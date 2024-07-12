#!/bin/bash

final_epoch_state=100
for epochs in 20
do
    # WN -- we use the model version as it was after 100 epochs
    wn_chkpt_id="chkpt-ID_2980578518512252"
    optimal_config="override-CoDExSmall-e20-bs128-lr5e-3-npp500.json"
    ./TWIG-I_from_checkpoint.sh \
        $wn_chkpt_id \
        $final_epoch_state \
        TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
        wn-to-codex-e${epochs}-opt_hyp

    optimal_config="override-DBpedia50-e20-bs64-lr5e-4-npp30.json"
    ./TWIG-I_from_checkpoint.sh \
        $wn_chkpt_id \
        $final_epoch_state \
        TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
        wn-to-dbpedia-e${epochs}-opt_hyp

    optimal_config="override-FB15k237-e20-bs128-lr5e-4-npp100.json"
    ./TWIG-I_from_checkpoint.sh \
        $wn_chkpt_id \
        $final_epoch_state \
        TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
        wn-to-fb-e${epochs}-opt_hyp

    fb_chkpt_id="chkpt-ID_8315447818826931"
    optimal_config="override-CoDExSmall-e20-bs64-lr5e-3-npp100.json"
    ./TWIG-I_from_checkpoint.sh \
        $fb_chkpt_id \
        $final_epoch_state \
        TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
        fb-to-codex-e${epochs}-opt_hyp

    optimal_config="override-DBpedia50-e20-bs128-lr5e-4-npp30.json"
    ./TWIG-I_from_checkpoint.sh \
        $fb_chkpt_id \
        $final_epoch_state \
        TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
        fb-to-dbpedia-e${epochs}-opt_hyp

    optimal_config="override-WN18RR-e20-bs128-lr5e-4-npp500.json"
    ./TWIG-I_from_checkpoint.sh \
        $fb_chkpt_id \
        $final_epoch_state \
        TWIG-I/overrides/finetune-opt-hyps/${optimal_config} \
        fb-to-wn-e${epochs}-opt_hyp
done
