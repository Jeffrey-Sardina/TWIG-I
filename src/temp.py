from twigi import *


dataset = 'CoDExSmall'
npp = 100
lr = 5e-3
batch_size = 64
do_job(
    "Wn18RR",
    model='base',
    negative_sampler="simple",
    loss_function="margin-ranking(0.1)",
    early_stopper=None,
    optimizer='Adam',
    optimizer_args = {
        "lr": lr
    },
    data_args = {
        "normalisation": "zscore",
        "batch_size": batch_size,
        "batch_size_test": 64,
        "fts_blacklist": set(['s_o_cofreq']),
    },
    training_args={
        "epochs": 20,
        "npp": npp,
        "hyp_validation_mode": False,
        "valid_every_n": -1
    },
    tag=f"{dataset}-pretrain"
)

dataset = 'DBpedai50'
npp = 30
lr = 5e-3
batch_size = 128
do_job(
    "Wn18RR",
    model='base',
    negative_sampler="simple",
    loss_function="margin-ranking(0.1)",
    early_stopper=None,
    optimizer='Adam',
    optimizer_args = {
        "lr": lr
    },
    data_args = {
        "normalisation": "zscore",
        "batch_size": batch_size,
        "batch_size_test": 64,
        "fts_blacklist": set(['s_o_cofreq']),
    },
    training_args={
        "epochs": 20,
        "npp": npp,
        "hyp_validation_mode": False,
        "valid_every_n": -1
    },
    tag=f"{dataset}-pretrain"
)

dataset = 'WN18RR'
npp = 500
lr = 5e-3
batch_size = 128
do_job(
    "Wn18RR",
    model='base',
    negative_sampler="simple",
    loss_function="margin-ranking(0.1)",
    early_stopper=None,
    optimizer='Adam',
    optimizer_args = {
        "lr": lr
    },
    data_args = {
        "normalisation": "zscore",
        "batch_size": batch_size,
        "batch_size_test": 64,
        "fts_blacklist": set(['s_o_cofreq']),
    },
    training_args={
        "epochs": 20,
        "npp": npp,
        "hyp_validation_mode": False,
        "valid_every_n": -1
    },
    tag=f"{dataset}-pretrain"
)

dataset = 'FB15k237'
npp = 100
lr = 5e-4
batch_size = 128
do_job(
    "Wn18RR",
    model='base',
    negative_sampler="simple",
    loss_function="margin-ranking(0.1)",
    early_stopper=None,
    optimizer='Adam',
    optimizer_args = {
        "lr": lr
    },
    data_args = {
        "normalisation": "zscore",
        "batch_size": batch_size,
        "batch_size_test": 64,
        "fts_blacklist": set(['s_o_cofreq']),
    },
    training_args={
        "epochs": 20,
        "npp": npp,
        "hyp_validation_mode": False,
        "valid_every_n": -1
    },
    tag=f"{dataset}-pretrain"
)
