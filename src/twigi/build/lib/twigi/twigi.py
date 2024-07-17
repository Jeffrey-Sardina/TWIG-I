# imports from twigi
from run_exp import main, load_dataset, load_filters, load_negative_samplers, train_and_eval, load_nn, load_loss_function, load_optimizer
from run_from_checkpoint import load_model_config
from twig_nn import *
from early_stopper import Early_Stopper
from negative_sampler import *

# external imports
import inspect
import os
from glob import glob
import copy

'''
=========
Constants
=========
'''
checkpoint_dir = 'checkpoints/'

def manage_job_inputs(
        dataset_names,
        model,
        negative_sampler,
        loss_function,
        early_stopper,
        optimizer,
        optimizer_args,
        data_args,
        training_args
    ):
    # assign defaults if needed
    if not "lr" in optimizer_args:
        optimizer_args["lr"] = 5e-3
    if not "normalisation" in data_args:
        data_args["normalisation"] = "zscore"
    if not "batch_size" in data_args:
        data_args["batch_size"] = 64
    if not "batch_size_test" in data_args:
        data_args["batch_size_test"] = 64
    if not "fts_blacklist" in data_args:
        data_args["fts_blacklist"] = []
    if not "epochs" in training_args:
        training_args["epochs"] = 20
    if not "npp" in training_args:
        training_args["npp"] = 30
    if not "hyp_validation_mode" in training_args:
        training_args["hyp_validation_mode"] = False

    # correct input types if needed
    if type(dataset_names) is str:
        dataset_names = [dataset_names]
    
    # we first need to load the data from the given dataset names
    dataloaders, norm_funcs, X_pos, n_local = load_dataset(
        dataset_names,
        normalisation=data_args["normalisation"],
        batch_size=data_args["batch_size"],
        batch_size_test=data_args["batch_size_test"],
        fts_blacklist=data_args["fts_blacklist"],
    )
    if training_args["hyp_validation_mode"]:
        valid_every_n = -1
        data_to_test_on = dataloaders['valid']
    else:
        valid_every_n = 5
        data_to_test_on = dataloaders['test']
    filters = load_filters(
        dataset_names=dataset_names,
        use_train_filter=False,
        use_valid_and_test_filters=True
    )

    # we now load the user-requested model
    if type(model) is str:
        if model.lower() == "TWIGI_Base".lower():
            model = "base"
        if model.lower() == "TWIGI_Linear".lower():
            model = "linear"
        model = load_nn(model, n_local=n_local)
    elif inspect.isclass(negative_sampler):
        if model == TWIGI_Base:
            model = "base"
        elif model == TWIGI_Linear:
            model = "linear"
        else:
            assert False, f"Invalid TWIG-I NN class given: {negative_sampler}"
    elif type(model) is torch.nn.Module:
        pass #user-defined model (already instantiated)
    else:
        assert False, f"Unsupported input for model: {model}"

    # load default optimizer if needed
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_args)
    elif type(optimizer) == str:
        optimizer = load_optimizer(optimizer, model, optimizer_args['lr'])
    elif inspect.isclass(optimizer):
        optimizer = optimizer(model.parameters(), **optimizer_args)
    elif isinstance(optimizer, torch.Optimizer):
        pass #user-defined negative sampler (already instantiated)
    else:
        assert False, f"Unsupported input for optimizer: {optimizer}"

    # now we need to set up the negative sampler
    if type(negative_sampler) == str:
        if negative_sampler.lower() == "Optimised_Negative_Sampler".lower():
            negative_sampler = "simple"
        if negative_sampler.lower() == "Vector_Negative_Sampler".lower():
            negative_sampler = "vector"
        negative_samplers = load_negative_samplers(
            dataset_names=dataset_names,
            filters=filters,
            norm_funcs=norm_funcs,
            sampler_type=negative_sampler,
            fts_blacklist=data_args["fts_blacklist"],
            X_pos=X_pos
        )
    elif inspect.isclass(negative_sampler):
        if negative_sampler == Optimised_Negative_Sampler:
            negative_samplers = load_negative_samplers(
                dataset_names=dataset_names,
                filters=filters,
                norm_funcs=norm_funcs,
                sampler_type="simple",
                fts_blacklist=data_args["fts_blacklist"],
                X_pos=X_pos
            )
        elif negative_sampler == Vector_Negative_Sampler:
            negative_samplers = load_negative_samplers(
                dataset_names=dataset_names,
                filters=filters,
                norm_funcs=norm_funcs,
                sampler_type="vector",
                fts_blacklist=data_args["fts_blacklist"],
                X_pos=X_pos
            )
        else:
            assert False, f"Invalid negative sampler class given: {negative_sampler}"
    elif isinstance(negative_sampler, Negative_Sampler):
        pass #user-defined negative sampler (already instantiated)
    else:
        assert False, f"Unsupported input for negative sampler: {negative_sampler}"

    # we now load the loss function
    if type(loss_function) is str:
        loss_function = load_loss_function(loss_fn_name=loss_function)
    elif callable(loss_function):
        pass #user-defined loss, already instantiated

    # now that this is all in a format TWIG-I can read, return it
    return model, dataloaders, data_to_test_on, training_args, optimizer, negative_samplers, loss_function, valid_every_n, early_stopper

def do_job(
        dataset_names,
        model='base',
        negative_sampler="simple",
        loss_function="margin-ranking(0.1)",
        early_stopper=Early_Stopper(
            start_epoch=5,
            patience=10,
            mode="on-falter",
            precision=3
        ),
        optimizer='Adam',
        optimizer_args = {
            "lr": 5e-3
        },
        data_args = {
            "normalisation": "zscore",
            "batch_size": 64,
            "batch_size_test": 64,
            "fts_blacklist": set(),
        },
        training_args={
            "epochs": 20,
            "npp": 30,
            "hyp_validation_mode": False
        },
        tag="super-cool-model"
):
    print('Running job! The arguemnts recieved are:')
    print(f"\t dataset_names: {dataset_names}")
    print(f"\t model: {model}")
    print(f"\t negative_sampler: {negative_sampler}")
    print(f"\t loss_function: {loss_function}")
    print(f"\t early_stopper: {early_stopper}")
    print(f"\t optimizer: {optimizer}")
    print(f"\t optimizer_args: {optimizer_args}")
    print(f"\t data_args: {data_args}")
    print(f"\t training_args: {training_args}")
    print(f"\t tag: {tag}")
    print()
    
    (
        model,
        dataloaders,
        data_to_test_on,
        training_args,
        optimizer,
        negative_samplers,
        loss_function,
        valid_every_n,
        early_stopper
    ) = manage_job_inputs(
        dataset_names=dataset_names,
        model=model,
        negative_sampler=negative_sampler,
        loss_function=loss_function,
        early_stopper=early_stopper,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        data_args=data_args,
        training_args=training_args
    )

    # set up internal variables
    model_name_prefix = tag + 'chkpt-ID_' + str(int(random.random() * 10**16))

    results = train_and_eval(
        model=model,
        training_dataloaders=dataloaders['train'],
        testing_dataloaders=data_to_test_on,
        valid_dataloaders=dataloaders['valid'],
        epochs=training_args["epochs"],
        optimizer=optimizer,
        npp=training_args["npp"],
        negative_samplers=negative_samplers,
        loss_function=loss_function,
        verbose=True,
        model_name_prefix=model_name_prefix,
        checkpoint_every_n=5,
        valid_every_n=valid_every_n,
        early_stopper=early_stopper
    )
    return results

def ablation_job(
        dataset_names,
        model=['base'],
        negative_sampler=['simple'],
        loss_function=['margin-ranking(0.01), margin-ranking(0.1), margin-ranking(0.5), pairwise-logistic'],
        early_stopper=[None],
        optimizer=['Adam'],
        optimizer_args = {
            "lr": [5e-3, 5e-4]
        },
        data_args = {
            "normalisation": ["zscore", "minmax"],
            "batch_size": [64, 128, 256],
            "batch_size_test": [64],
            "fts_blacklist": [
                set()
            ],
        },
        training_args={
            "epochs": [10],
            "npp": [30, 100, 250],
            "hyp_validation_mode": [True]
        },
        tag="Ablation Job",
        ablation_metric='mrr',
        train_and_eval_after = False,
        train_and_eval_args = {
            "epochs": 100,
            "early_stopper": Early_Stopper(
                start_epoch=5,
                patience=10,
                mode="on-falter",
                precision=3
            ),
            "batch_size_test": 64,
            "hyp_validation_mode": False
        }
    ):
    # assign defaults if needed
    if not "lr" in optimizer_args:
        optimizer_args["lr"] =  [5e-3, 5e-4]
    if not "normalisation" in data_args:
        data_args["normalisation"] = ["zscore", "minmax"]
    if not "batch_size" in data_args:
        data_args["batch_size"] = [64, 128, 256]
    if not "batch_size_test" in data_args:
        data_args["batch_size_test"] = [64]
    if not "fts_blacklist" in data_args:
        data_args["fts_blacklist"] = [set()]
    if not "epochs" in training_args:
        training_args["epochs"] = [10]
    if not "npp" in training_args:
        training_args["npp"] = [30, 100, 250]
    if not "hyp_validation_mode" in training_args:
        training_args["hyp_validation_mode"] = [True]
    if train_and_eval_after:
        if not "epochs" in train_and_eval_args:
            train_and_eval_args["epochs"] = 100
        if not "early_stopper" in train_and_eval_args:
            train_and_eval_args["early_stopper"] = Early_Stopper(
                start_epoch=5,
                patience=10,
                mode="on-falter",
                precision=3
            ),
        if not "batch_size_test" in train_and_eval_args:
            train_and_eval_args["batch_size_test"] = 64
        if not "hyp_validation_mode" in train_and_eval_args:
            train_and_eval_args["hyp_validation_mode"] = False

    ablation_results = {}
    best_metric = 0.0
    best_results = None
    best_settings = None
    for mod in model:
        for neg_samp in negative_sampler:
            for loss_fn in loss_function:
                for es in early_stopper:
                    for norm in data_args["normalisation"]:
                        for opt in optimizer:
                            for lr in optimizer_args["lr"]:
                                for bs in data_args["batch_sizes"]:
                                    for test_bs in data_args["batch_size_test"]:
                                        for ft_blacklist in data_args["fts_blacklist"]:
                                            for n_epochs in training_args["epochs"]:
                                                for npp_val in training_args["npp"]:
                                                    for hyp_val_mode in training_args["hyp_validation_mode"]:
                                                        settings = (
                                                            mod,
                                                            neg_samp,
                                                            loss_fn,
                                                            es,
                                                            opt,
                                                            lr,
                                                            norm,
                                                            bs,
                                                            test_bs,
                                                            ft_blacklist,
                                                            n_epochs,
                                                            npp_val,
                                                            hyp_val_mode
                                                        )
                                                        # need this so if a model type is ever re-used, we train it from its initial point, not from the last ablation!
                                                        # TODO: should verify manually that this is the case as well!
                                                        mod_copy = copy.deepcopy(mod)
                                                        results = do_job(
                                                            dataset_names=dataset_names,
                                                            model=mod_copy,
                                                            optimizer=opt,
                                                            negative_sampler=neg_samp,
                                                            loss_function=loss_fn,
                                                            early_stopper=es,
                                                            data_args = {
                                                                "normalisation": norm,
                                                                "batch_size": bs,
                                                                "batch_size_test": test_bs,
                                                                "fts_blacklist": ft_blacklist,
                                                            },
                                                            training_args={
                                                                "epochs": n_epochs,
                                                                "npp": npp_val,
                                                                "hyp_validation_mode": hyp_val_mode
                                                            },
                                                            tag=tag
                                                        )
                                                        ablation_results[settings] = results
                                                        metrics = []
                                                        for dataset_name in dataset_names:
                                                            metrics.append(results[dataset_name][ablation_metric])
                                                        metric_avg = sum(metrics) / len(metrics)
                                                        if metric_avg > best_metric:
                                                            best_metric = metric_avg
                                                            best_results = results
                                                            best_settings = settings
    print('Ablation done!')
    print('The best settings found were:')
    (
        mod,
        neg_samp,
        loss_fn,
        es,
        opt,
        lr,
        norm,
        bs,
        test_bs,
        ft_blacklist,
        n_epochs,
        npp_val,
        hyp_val_mode
    ) = best_settings
    print(f"\t model: {mod}")
    print(f"\t negative sampler: {neg_samp}")
    print(f"\t loss function: {loss_fn}")
    print(f"\t early stopper: {es}")
    print(f"\t optimizer: {opt}")
    print(f"\t learning rate: {lr}")
    print(f"\t normalisation: {norm}")
    print(f"\t batch_size: {bs}")
    print(f"\t test batch size: {test_bs}")
    print(f"\t feature blacklist: {ft_blacklist}")
    print(f"\t epochs: {n_epochs}")
    print(f"\t npp: {npp_val}")
    print(f"\t hyperparameter validation mode: {hyp_val_mode}")
    print()

    print('The best results achaived (on average) were:')
    print(f"Average {ablation_metric}: {best_metric}")
    for dataset_name in dataset_names:
        print(f"\t On {dataset_name}: {best_results[dataset_name]}")
    print()

    if train_and_eval_after:
        print('Now training your final model!')
        do_job(
            dataset_names=dataset_names,
            model=mod,
            negative_sampler=neg_samp,
            loss_function=loss_fn,
            early_stopper=train_and_eval_args["early_stopper"],
            optimizer=opt,
            optimizer_args = {
                "lr": lr
            },
            data_args = {
                "normalisation": norm,
                "batch_size": bs,
                "batch_size_test": train_and_eval_args["batch_size_test"],
                "fts_blacklist": ft_blacklist,
            },
            training_args={
                "epochs": train_and_eval_args["epochs"],
                "npp": npp_val,
                "hyp_validation_mode": train_and_eval_args["hyp_validation_mode"]
            },
            tag=f"{tag}_train-and-eval-on-best"
        )

def get_epoch_state(checkpoint_id):
    all_epoch_states = glob(os.path.join(checkpoint_dir, f"{checkpoint_id}*.pt"))
    all_epoch_states = []
    for filename in all_epoch_states:
        file_epoch_state = os.path.basename(filename).split('_')[-1].split('.')[0].replace('e', '')
        all_epoch_states.append(file_epoch_state)
    if epoch_state < 0:
        epoch_state = max(all_epoch_states)
    assert epoch_state in all_epoch_states, f"Provided epoch state {epoch_state} does not have a corresponding checkpont. Please choose an epoch state from {all_epoch_states}"
    return epoch_state

def apply_user_override(
        model_config,
        negative_sampler,
        loss_function,
        early_stopper,
        optimizer,
        optimizer_args,
        data_args,
        training_args
    ):
    # apply override from what the user provided
    if negative_sampler:
        model_config['sampler_type'] = negative_sampler
    if loss_function:
        model_config['loss_function'] = loss_function
    if early_stopper:
        model_config['early_stopper'] = early_stopper
    if optimizer:
        model_config['optimizer_name'] = optimizer
    if optimizer_args:
        model_config['optimizer_args'] = optimizer_args
    if data_args:
        if 'normalisation' in data_args:
            model_config['normalisation'] = data_args['normalisation']
        if 'batch_size' in data_args:
            model_config['batch_size'] = data_args['batch_size']
        if 'batch_size_test' in data_args:
            model_config['batch_size_test'] = data_args['batch_size_test']
        if 'fts_blacklist' in data_args:
            model_config['fts_blacklist'] = data_args['fts_blacklist']
    if 'epochs' in training_args:
        model_config['epochs'] = training_args['epochs']
    else:
        assert False, "A number of epochs to finetune for must be provided!"
    if 'npp' in training_args:
        model_config['npp'] = training_args['npp']
    if 'hyp_validation_mode' in training_args:
        model_config['hyp_validation_mode'] = training_args['hyp_validation_mode']
    return model_config

def finetune_job(
        dataset_names,
        checkpoint_id,
        epoch_state=-1,
        negative_sampler=None,
        loss_function=None,
        early_stopper=Early_Stopper(
            start_epoch=5,
            patience=10,
            mode="on-falter",
            precision=3
        ),
        optimizer=None,
        optimizer_args=None,
        data_args=None,
        training_args=None,
        tag="Finetune-Job"
    ):
    if data_args:
        assert not "fts_blacklist" in data_args, "Feature blacklist cannot be changed during fintuning"
    tag += "-from-" + checkpoint_id

    # load the desired epoch state
    epoch_state = get_epoch_state(checkpoint_id)

    # load the model and config
    torch_checkpont_path = os.path.join(checkpoint_dir, f"{checkpoint_id}_e{epoch_state}.pt")
    model_config_path = os.path.join(checkpoint_dir, f"{checkpoint_id}.pkl")
    pretrained_model = torch.load(torch_checkpont_path)
    model_config = apply_user_override(
        model_config=load_model_config(model_config_path),
        negative_sampler=negative_sampler,
        loss_function=loss_function,
        early_stopper=early_stopper,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        data_args=data_args,
        training_args=training_args
    )
    
    do_job(
        dataset_names=dataset_names,
        model=pretrained_model,
        negative_sampler=model_config['sampler_type'],
        loss_function=model_config['loss_function'],
        early_stopper=model_config['early_stopper'],
        optimizer_args=model_config['optimizer_args'],
        optimizer=model_config['optimizer'],
        data_args = {
            "normalisation": model_config['normalisation'],
            "batch_size": model_config['batch_size'],
            "batch_size_test": model_config['batch_size_test'],
            "fts_blacklist": model_config['fts_blacklist'],
        },
        training_args={
            "epochs": model_config['epochs'],
            "npp": model_config['nmodelpp'],
            "hyp_validation_mode": model_config['hyp_validation_mode']
        },
        tag=tag
    )

def finetune_ablation_job(
        dataset_names,
        checkpoint_id,
        epoch_state=-1,
        negative_sampler=['simple'],
        loss_function=['margin-ranking(0.01), margin-ranking(0.1), margin-ranking(0.5), pairwise-logistic'],
        early_stopper=[None],
        optimizer=['Adam'],
        optimizer_args = {
            "lr": [5e-3, 5e-4]
        },
        data_args = {
            "normalisation": ["zscore", "minmax"],
            "batch_size": [64, 128, 256],
            "batch_size_test": [64],
            "fts_blacklist": [
                set()
            ],
        },
        training_args={
            "epochs": [10],
            "npp": [30, 100, 250],
            "hyp_validation_mode": [True]
        },
        tag="Ablation Job",
        ablation_metric='mrr',
        train_and_eval_after = False,
        train_and_eval_args = {
            "epochs": 100,
            "early_stopper": Early_Stopper(
                start_epoch=5,
                patience=10,
                mode="on-falter",
                precision=3
            ),
            "batch_size_test": 64,
            "hyp_validation_mode": False
        }
    ):
    if data_args:
        assert not "fts_blacklist" in data_args, "Feature blacklist cannot be changed during fintuning"
    tag += "-from-" + checkpoint_id

    # load the desired epoch state
    epoch_state = get_epoch_state(checkpoint_id)

    # load the model and config
    torch_checkpont_path = os.path.join(checkpoint_dir, f"{checkpoint_id}_e{epoch_state}.pt")
    model_config_path = os.path.join(checkpoint_dir, f"{checkpoint_id}.pkl")
    pretrained_model = torch.load(torch_checkpont_path)
    model_config = apply_user_override(
        model_config=load_model_config(model_config_path),
        negative_sampler=negative_sampler,
        loss_function=loss_function,
        early_stopper=early_stopper,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        data_args=data_args,
        training_args=training_args
    )

    # run the ablation
    results = ablation_job(
        dataset_names=dataset_names,
        model=pretrained_model,
        negative_sampler=model_config['sampler_type'],
        loss_function=model_config['loss_function'],
        early_stopper=model_config['early_stopper'],
        optimizer=model_config['optimizer'],
        optimizer_args=model_config['optimizer_args'],
        data_args = {
            "normalisation": model_config['normalisation'],
            "batch_size": model_config['batch_size'],
            "batch_size_test": model_config['batch_size_test'],
            "fts_blacklist": model_config['fts_blacklist'],
        },
        training_args={
            "epochs": model_config['epochs'],
            "npp": model_config['nmodelpp'],
            "hyp_validation_mode": model_config['hyp_validation_mode']
        },
        tag=tag,
        ablation_metric=ablation_metric,
        train_and_eval_after=train_and_eval_after,
        train_and_eval_args=train_and_eval_args
    )
    return results
