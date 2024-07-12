#!/bin/bash

## do a baseline run
fts="None"
./TWIG-I_pipeline.sh CoDExSmall 20 100 5e-3 1 64 64 1 $fts base-run


## do a second baseline (only 1 hop fts)
fts="None"
./TWIG-I_pipeline.sh CoDExSmall 20 100 5e-3 1 64 64 0 "$fts" 1-hop-only


## do a third baseline (only 2-hop fts, none of the 1-hop ones)
fts="s_deg o_deg p_freq s_p_cofreq o_p_cofreq s_o_cofreq"
./TWIG-I_pipeline.sh CoDExSmall 20 100 5e-3 1 64 64 1 "$fts" 2-hop-only


## try removing by elems (the 1-hop fts)
for fts in "s_deg" \
    "o_deg" \
    "p_freq" \
    "s_p_cofreq" \
    "o_p_cofreq" \
    "s_o_cofreq"
do
    ./TWIG-I_pipeline.sh CoDExSmall 20 100 5e-3 1 64 64 1 "$fts" bl-${fts}
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
    ./TWIG-I_pipeline.sh CoDExSmall 20 100 5e-3 1 64 64 1 "$fts" bl-${fts}
done
