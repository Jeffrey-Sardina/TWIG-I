#!/bin/bash

# rec_v0_FB15k237_norm-zscore_e20-lr5e-4_bs128_npp100_simple-sampler_filter-code01_tag-negfix

## do a baseline run
fts="None"
./TWIG-I_pipeline.sh FB15k237 20 100 5e-4 1 128 128 1 $fts base-run


## do a second baseline (only 1 hop fts)
fts="None"
./TWIG-I_pipeline.sh FB15k237 20 100 5e-4 1 128 128 0 $fts 1-hop-only


## do a third baseline (only 2-hop fts, none of the 1-hop ones)
fts="s_deg o_deg p_freq s_p_cofreq o_p_cofreq s_o_cofreq"
./TWIG-I_pipeline.sh FB15k237 20 100 5e-4 1 128 128 1 "$fts" 2-hop-only


## try removing by elems (the 1-hop fts)
for fts in "s_deg" \
    "o_deg" \
    "p_freq" \
    "s_p_cofreq" \
    "o_p_cofreq" \
    "s_o_cofreq"
do
    ./TWIG-I_pipeline.sh FB15k237 20 100 5e-4 1 128 128 1 "$fts" bl-${fts}
done


## try removing by pairs
for fts in "o_min_deg_neighbour s_min_deg_neighbour" \
    "o_max_deg_neighbour s_max_deg_neighbour" \
    "o_mean_deg_neighbour s_mean_deg_neighbour" \
    "o_num_neighbours s_num_neighbours" \
    "o_min_freq_rel s_min_freq_rel" \
    "o_max_freq_rel s_max_freq_rel" \
    "o_mean_freq_rel s_mean_freq_rel" \
    "o_num_rels s_num_rels"
do
    ./TWIG-I_pipeline.sh FB15k237 20 100 5e-4 1 128 128 1 "$fts" bl-${fts}
done
