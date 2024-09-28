from twigi import *

# pretrain
# dataset = 'CoDExSmall'
# npp = 100
# lr = 5e-3
# batch_size = 64
# do_job(
#     dataset,
#     model='base',
#     negative_sampler="simple",
#     loss_function="margin-ranking(0.1)",
#     early_stopper=None,
#     optimizer='Adam',
#     optimizer_args = {
#         "lr": lr
#     },
#     data_args = {
#         "normalisation": "zscore",
#         "batch_size": batch_size,
#         "batch_size_test": 64,
#         "fts_blacklist": set(['s_o_cofreq']),
#     },
#     training_args={
#         "epochs": 20,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"{dataset}-pretrain"
# )

# dataset = 'DBpedia50'
# npp = 30
# lr = 5e-3
# batch_size = 128
# do_job(
#     dataset,
#     model='base',
#     negative_sampler="simple",
#     loss_function="margin-ranking(0.1)",
#     early_stopper=None,
#     optimizer='Adam',
#     optimizer_args = {
#         "lr": lr
#     },
#     data_args = {
#         "normalisation": "zscore",
#         "batch_size": batch_size,
#         "batch_size_test": 64,
#         "fts_blacklist": set(['s_o_cofreq']),
#     },
#     training_args={
#         "epochs": 20,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"{dataset}-pretrain"
# )

# dataset = 'WN18RR'
# npp = 500
# lr = 5e-3
# batch_size = 128
# do_job(
#     dataset,
#     model='base',
#     negative_sampler="simple",
#     loss_function="margin-ranking(0.1)",
#     early_stopper=None,
#     optimizer='Adam',
#     optimizer_args = {
#         "lr": lr
#     },
#     data_args = {
#         "normalisation": "zscore",
#         "batch_size": batch_size,
#         "batch_size_test": 64,
#         "fts_blacklist": set(['s_o_cofreq']),
#     },
#     training_args={
#         "epochs": 20,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"{dataset}-pretrain"
# )

# dataset = 'FB15k237'
# npp = 100
# lr = 5e-4
# batch_size = 128
# do_job(
#     dataset,
#     model='base',
#     negative_sampler="simple",
#     loss_function="margin-ranking(0.1)",
#     early_stopper=None,
#     optimizer='Adam',
#     optimizer_args = {
#         "lr": lr
#     },
#     data_args = {
#         "normalisation": "zscore",
#         "batch_size": batch_size,
#         "batch_size_test": 64,
#         "fts_blacklist": set(['s_o_cofreq']),
#     },
#     training_args={
#         "epochs": 20,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"{dataset}-pretrain"
# )

# # finetune 10 -- from FB15k237
# checkpoint_id = "chkpt-ID_6385341367018417"

# dataset = 'CoDExSmall'
# npp = 100
# lr = 5e-3
# batch_size = 64
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"FB15k237-to-{dataset}"
# )

# dataset = 'DBpedia50'
# npp = 30
# lr = 5e-3
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"FB15k237-to-{dataset}"
# )

# dataset = 'WN18RR'
# npp = 500
# lr = 5e-3
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"FB15k237-to-{dataset}"
# )


# # finetune 10 -- from WN18RR
# checkpoint_id = "chkpt-ID_5360871134295512"

# dataset = 'CoDExSmall'
# npp = 100
# lr = 5e-3
# batch_size = 64
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"WN18RR-to-{dataset}"
# )

# dataset = 'DBpedia50'
# npp = 30
# lr = 5e-3
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"WN18RR-to-{dataset}"
# )

# dataset = 'FB15k237'
# npp = 100
# lr = 5e-4
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"WN18RR-to-{dataset}"
# )


# # finetune 10 -- from CoDExSmall
# checkpoint_id = "chkpt-ID_8182621431323017"

# dataset = 'DBpedia50'
# npp = 30
# lr = 5e-3
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"CoDExSmall-to-{dataset}"
# )

# dataset = 'FB15k237'
# npp = 100
# lr = 5e-4
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"CoDExSmall-to-{dataset}"
# )

# dataset = 'WN18RR'
# npp = 500
# lr = 5e-3
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"CoDExSmall-to-{dataset}"
# )


# # finetune 10 -- from DBpedia50
# checkpoint_id = "chkpt-ID_2429873403687784"

# dataset = 'CoDExSmall'
# npp = 100
# lr = 5e-3
# batch_size = 64
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"DBpedia50-to-{dataset}"
# )

# dataset = 'FB15k237'
# npp = 100
# lr = 5e-4
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"DBpedia50-to-{dataset}"
# )

# dataset = 'WN18RR'
# npp = 500
# lr = 5e-3
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 10,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"DBpedia50-to-{dataset}"
# )


# # eval after 10 -- from FB15k237
# checkpoint_id = "chkpt-ID_6385341367018417"
# dataset = 'FB15k237'
# npp = 100
# lr = 5e-4
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 0,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"{dataset}-e10-eval"
# )

# # eval after 10 -- from WN18RR
# checkpoint_id = "chkpt-ID_5360871134295512"
# dataset = 'WN18RR'
# npp = 500
# lr = 5e-3
# batch_size = 128
# finetune_job(
#     dataset,
#     checkpoint_id,
#     epoch_state=10,
#     negative_sampler=None,
#     loss_function=None,
#     early_stopper=None,
#     optimizer=None,
#     optimizer_args={
#         "lr": lr
#     },
#     data_args={
#         "batch_size": batch_size
#     },
#     training_args={
#         "epochs": 0,
#         "npp": npp,
#         "hyp_validation_mode": False,
#         "valid_every_n": -1
#     },
#     tag=f"{dataset}-e10-eval"
# )

# eval after 10 -- from DBpedia50
checkpoint_id = "chkpt-ID_2429873403687784"
dataset = 'DBpedia50'
npp = 30
lr = 5e-3
batch_size = 128
finetune_job(
    dataset,
    checkpoint_id,
    epoch_state=10,
    negative_sampler=None,
    loss_function=None,
    early_stopper=None,
    optimizer=None,
    optimizer_args={
        "lr": lr
    },
    data_args={
        "batch_size": batch_size
    },
    training_args={
        "epochs": 0,
        "npp": npp,
        "hyp_validation_mode": False,
        "valid_every_n": -1
    },
    tag=f"{dataset}-e10-eval"
)

# eval after 10 -- from CoDExSmall
checkpoint_id = "chkpt-ID_8182621431323017"
dataset = 'CoDExSmall'
npp = 100
lr = 5e-3
batch_size = 64
finetune_job(
    dataset,
    checkpoint_id,
    epoch_state=10,
    negative_sampler=None,
    loss_function=None,
    early_stopper=None,
    optimizer=None,
    optimizer_args={
        "lr": lr
    },
    data_args={
        "batch_size": batch_size
    },
    training_args={
        "epochs": 0,
        "npp": npp,
        "hyp_validation_mode": False,
        "valid_every_n": -1
    },
    tag=f"{dataset}-e10-eval"
)