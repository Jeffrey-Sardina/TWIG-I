#!/bin/bash

checkpoint_id=$1 # e.x. chkpt-ID_1726265127922990
epoch_state_to_load_from=$2 # e.x. 50
setting_overrides_file=$(realpath $3) # e.x. TWIG-I/override.json. We make it absolute to avoid issues with dir. chenges later in the code
tag=$4

checkpoints_dir="checkpoints"
torch_model_file="${checkpoints_dir}/${checkpoint_id}_e${epoch_state_to_load_from}.pt"
orignal_model_settings_file="${checkpoints_dir}/${checkpoint_id}.pkl"
out_file="rerun_${checkpoint_id}_e${epoch_state_to_load_from}_tag-${tag}.log"

cd TWIG-I/
start=`date +%s`
python -u run_from_checkpoint.py \
    $torch_model_file \
    $orignal_model_settings_file \
    $setting_overrides_file \
    &> $out_file

end=`date +%s`
runtime=$((end-start))
echo "Experiments took $runtime seconds on " &>> $out_file
