# imports from twigi
from load_data import do_load, get_adj_data
from twig_nn import *
from trainer import run_training
from negative_sampler import *
from loss import *
from early_stopper import *
from utils import get_triples, calc_graph_stats, get_triples_by_idx, load_custom_dataset

# external imports
import sys
import os
import torch
from pykeen import datasets as pykeendatasets
import random
import pickle

'''
===============
Reproducibility
===============
'''
torch.manual_seed(17)

'''
=========
Constants
=========
'''
checkpoint_dir = 'checkpoints/'

'''
================
Module Functions
================
'''
def load_nn(version, n_local):
    '''
    load_nn() loads a PyTorch model from twig_nn.py. 

    The arguments it accepts are:
        - version (str): the version of the NN to load. Currently, "base" and "single".
        - n_local (int) the number of input features to the network. Used to  default to 22 (or 6 in 1-hop mode)

    The values it returns are:
        - model (torch.nn.Module): the PyTorch NN Model object containing TWIG's neural architecture.
    '''
    print('loading NN')
    valid_versions = ('base', 'linear')
    if version == "base":
        model = TWIGI_Base(
            n_local=n_local
        )
    elif version == "linear":
        model = TWIGI_Linear(
            n_local=n_local
        )
    else:
        assert False, f"Invald NN version given: {version}. Valid options are {valid_versions}."
    print("done loading NN")
    return model

def load_optimizer(optimizer_name, model, lr):
    '''
    load_optimizer() loads a PyTorch optimizer for use during learning. 

    The arguments it accepts are:
        - optimizer_name (str): the name of the optimizer that should be used.
        - model (torch.nn.Module): the PyTorch NN Model object containing TWIG's neural architecture.
        - lr: the learning rate to be usedd by TWIG-I.

    The values it returns are:
        - optimizer (torch.optim.Optimizer): the optimizer object to be used
    '''
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        assert False, f"Unrecognised optimizer: {optimizer_name}"
    return optimizer

def load_dataset(
        dataset_names,
        normalisation,
        batch_size,
        batch_size_test,
        fts_blacklist
    ):
    ''''
    load_dataset() loads all training, testing, and validation data
    assocaited with a single dataset. The dataset is given by name and
    must be one of the datasets defined self, start_epoch, patience, mode, precisionin PyKEEN (https://github.com/pykeen/pykeen#datasets)

    The dataset load uses function implementation in load_data.py and also
    performs all needed preprocessing, normalisation, etc such that the returned
    data can be directly used in learning.

    The arguments it accepts are:
        - dataset_names (list of str): A list of the dataset names expressed as strings
        - normalisation (str): A string representing the method that should be used to normalise all data. Currently, "zscore" and "minmax" are implemented; "none" can be given to not use normalisation.
        - batch size (int): the batch size to use during training. It is needed here to construct the dataloader objects that are returned, as PyTorch bundles batch size into the dataloader object.
        - batch size_test (int): the batch size to use during testing and validation. It is needed here to construct the dataloader objects that are returned, as PyTorch bundles batch size into the dataloader object.
        - fts_blacklist (list of str): A  list of feature names that should NOT be calculated for negatives.

    The values it returns are:
        - dataloaders (dict str -> str -> torch.utils.data.DataLoader): a dict that maps a training split ("train", "test", or "valid") and a dataset (with a name as in dataset_names) to a DataLoader that can be used to load batches for that dataset on that training split. An example of accessing its data could be dataloaders["train"]["UMLS"], which would return the training dataloader for the UMLS dataset.
        - norm_funcs (dict of str -> Callable): A dictionary that maps dataset names (as str) to the normalisation function (Callable) to be used when loading data for that dataset (such as feature vectors for its generated triple).
        - n_local (int): the total number of features calculated in the feature vector of all triples / negatives. The "local" in "n_local" comes from the fact that all of these are local-structure features (however, global feartures are not used and therefore calling this "local" is technically redundant)
    '''

    print('loading dataset')

    dataloaders, norm_funcs, X_pos, n_local = do_load(
        dataset_names,
        normalisation=normalisation,
        batch_size=batch_size,
        batch_size_test=batch_size_test,
        fts_blacklist=fts_blacklist
    )
    print(f'Using a total of {n_local} features')
    print("done loading dataset")
    return dataloaders, norm_funcs, X_pos, n_local

def load_filters(
        dataset_names,
        use_train_filter,
        use_valid_and_test_filters
    ):
    '''
    load_filters() loads the filters that should be used for a given dataset. Filters are split by their purpose (what phase they are ussed at; i.e. train, test, or valid). In the current implmenetation,
    - the train filters consist of all training triples (if used)
    - the valid filters consist of all training and validation triples (if used)
    - the tetst filters consist of all training, validation, and testing triples (i.e. all triples, if used).

    This is done to ensure that the filters themsevles do not allow for test or validation leakage during model training or creation.

    The arguments it accepts are:
        - dataset_names (list of str): A list of the dataset names expressed as strings
        - use_train_filter (bool): True if negative samples generated during training should be filtered, False if they should not be filtered
        - use_valid_and_test_filters (bool): True if negative samples generated during validation and testing should be filtered, False if they should not be filtered

    The values it returns are:
        - filters (dict str -> str -> dict): a dict that maps a dataset name (str) and a training split name (i.e. "train", "test", or "valid") to a dictionary describing the triples to use i filtering. To be exact, this second triples_dict has the structure (dict str -> int -> tuple<int,int,int>). It maps first the training split name to the triple index, and that trtrriple index maps to a single triple expressed as (s, p, o) with integral representations of each triple element.
    '''
    print('loading filters')
    filters = {dataset_name:{} for dataset_name in dataset_names}
    for dataset_name in dataset_names:
        try:
            pykeen_dataset = pykeendatasets.get_dataset(dataset=dataset_name)
        except:
            pykeen_dataset = load_custom_dataset(dataset_name)
        triples_dicts = get_triples(pykeen_dataset)
        filters[dataset_name] = {
            'train': set(triples_dicts['train']) if use_train_filter else set(),
            'valid': set(triples_dicts['train'] + triples_dicts['valid']) if use_valid_and_test_filters else set(),
            'test': set(triples_dicts['all']) if use_valid_and_test_filters else set()
        }
    print('done loading filters')
    return filters

def load_negative_samplers(
        dataset_names,
        filters,
        norm_funcs,
        sampler_type,
        fts_blacklist,
        X_pos
    ):
    '''
    load_negative_samplers() loads all negative samplers. Note that negative samplers are dataset-specific, so when multiple datasets are being learned on, multiple negative samplers must be created. This si why all dataset names must be given to this function.

    The arguments it accepts are:
        - dataset_names (list of str): A list of the dataset names expressed as strings
        - filters (dict str -> str -> dict): a dict that maps a dataset name (str) and a training split name (i.e. "train", "test", or "valid") to a dictionary describing the triples to use i filtering. To be exact, this second triples_dict has the structure (dict str -> int -> tuple<int,int,int>). It maps first the training split name to the triple index, and that trtrriple index maps to a single triple expressed as (s, p, o) with integral representations of each triple element.
        - norm_funcs (dict of str -> Callable): A dictionary that maps dataset names (as str) to the normalisation function (Callable) to be used when loading data for that dataset (such as feature vectors for its generated triple).
        - sampler_type (str) a string defining what type of negative sampler is desired. Currently only "simple" is the only option (this means that the sampling procedure is random with no biases, pseudo-typing, etc). 
        - fts_blacklist (list of str): A  list of feature names that should NOT be calculated for negatives.
        - X_pos (torch.Tensor): A tensor containing all positive triples' feature vectors as rows.

    The values it returns are:
        - negative_samplers (dict str -> Negative_Sampler): a dict that maps a dataset name to the negative sampler associated with that dataset. 
    '''
    print('loading negative samplers')

    negative_samplers = {}
    for dataset_name in dataset_names:
        try:
            pykeen_dataset = pykeendatasets.get_dataset(dataset=dataset_name)
        except:
            pykeen_dataset = load_custom_dataset(dataset_name)  
        triples_dicts = get_triples(pykeen_dataset)
        graph_stats = calc_graph_stats(triples_dicts, do_print=False)
        triples_map = get_triples_by_idx(triples_dicts, 'all')
        ents_to_triples = get_adj_data(triples_map)
        simple_sampler = Optimised_Negative_Sampler(
            filters=filters[dataset_name],
            graph_stats=graph_stats,
            triples_map=triples_map,
            ents_to_triples=ents_to_triples,
            norm_func=norm_funcs[dataset_name],
            dataset_name=dataset_name,
            fts_blacklist=fts_blacklist
        )
        if sampler_type == 'simple':
            negative_samplers[dataset_name] = simple_sampler
        elif sampler_type == 'vector':
            negative_samplers[dataset_name] = Vector_Negative_Sampler(
                X_pos=X_pos,
                n_bins=30,
                mode="near-negative",
                dataset_name=dataset_name,
                simple_negative_sampler=simple_sampler
            )
        else:
            assert False, f'Unknown negative sampler type requested: {sampler_type}. Only "simple" and "vecgtor" are supported currently.'
    print('done loading negative samplers')
    return negative_samplers

def load_loss_function(loss_fn_name):
    '''
    load_loss_function() loads the loss function that TWIG-I should use during training.

    The arguments it accepts are:
        - loss_fn_name (str): the name of the loss function to use. Valid options are "margin-ranking" and "pairwise-logistic". If margin rankking is used, the margin must be given in parentheses; i.e. "margin-ranking(0.1)".

    The values it returns are:
        - loss_function: (func): the loss function object created with the specified hyperparameters
    '''
    reduction = 'mean'
    if "margin-ranking" in loss_fn_name:
        margin = float(loss_fn_name.strip().replace("margin-ranking", "").replace("(", "").replace(")", ""))
        loss_function = get_margin_ranking_loss(
            reduction=reduction,
            margin=margin
        )
    elif "pairwise-logistic" in loss_fn_name:
        loss_function = get_pairwise_logistic_loss(
            reduction=reduction,
        )
    else:
        assert False, f"Invalid loss function name given: {loss_fn_name}. Expected one of: 'margin-ranking(<margin-val>)', 'pairwise-logistic'"
    return loss_function

def load_early_stopper(start_epoch, patience, mode):
    '''
    load_early_stopper() created an early stopper object for TWIG-I to track if it should tirgger early stopping or not

    The arguments it accepts are:
        - start_epoch (int): the first epoch on which the early stopping could be applied (this is often called a `burn-in` time)
        - patience (int): the number of epochs to wait after seeing what would otherwise be a stop-condition before the early stopping is actually applied.
        - mode (str): the mode of early stopping to use. Options are as follows:
            - "on-falter" -- trigger early stopping the first time a validation result does not get better than a previous result
            - "never" -- never do early stopping
    The values it returns are:
        - early_stopper (Early_Stopper): an Early_Stopper object that TWIG-I should use to determine if it should stop training early, or None if not early stopping is wanted.
    '''
    precision = 3
    early_stopper = Early_Stopper(
        start_epoch=start_epoch,
        patience=patience,
        mode=mode,
        precision=precision
    )
    return early_stopper

def train_and_eval(
        model,
        training_dataloaders,
        testing_dataloaders,
        valid_dataloaders,
        epochs,
        optimizer,
        npp,
        negative_samplers,
        loss_function,
        verbose,
        model_name_prefix,
        checkpoint_every_n,
        valid_every_n,
        early_stopper
    ):
    ''''
    train_and_eval() runs the training and evaluation loops on the data, and prints all results.

    The arguments it accepts are:
        - model (torch.nn.Module): the PyTorch NN Model object containing TWIG-I's neural architecture.
        - training_dataloaders (dict str -> torch.utils.data.DataLoader): a dict that maps a dataset name string to a DataLoader that can be used to load batches for that dataset on that training split. An example of accessing its data could be training_dataloaders["UMLS"], which would return the training dataloader for the UMLS dataset.
        - testing_dataloaders (dict str -> torch.utils.data.DataLoader): a dict that maps a dataset name string to a DataLoader that can be used to load batches for that dataset on that testing split. An example of accessing its data could be testing_dataloaders["UMLS"], which would return the testing dataloader for the UMLS dataset.
        - valid_dataloaders (dict str -> torch.utils.data.DataLoader): a dict that maps a dataset name string to a DataLoader that can be used to load batches for that dataset on that validation split. An example of accessing its data could be valid_dataloaders["UMLS"], which would return the validation dataloader for the UMLS dataset.
        - epochs (int): the number of epochs to train for
        - optimizer (torch.optim.Optimizer): the optimizer object to be used
        - npp (int): the number of negative samples to generate per positive triple
          during training.
        - negative_samplers (dict str -> Negative_Sampler): a dict that maps a dataset name to the negative sampler associated with that dataset.
        - loss_function (func): the loss function object that should be used during training.
        - verbose (bool): whether or not all information should be output. If True, TWIG will be run in verbose mode, whhich means more information will be printed.
        - model_name_prefix (str): the prefix to prepend to the model name when saving checkpoints (currently unused, as checkpoints are not saved)
        - checkpoint_every_n (int): the interval of epochs after which a checkpoint should be saved during training.
        - valid_every_n (int): the interval of epochs after which TWIG-I should be evaluated on its validation dataset.
        - early_stopper (Early_Stopper): an Early_Stopper object that TWIG-I should use to determine if it should stop training early, or None if not early stopping is wanted.

    The values it returns are:
        - results (dict str -> str -> float): the results output from train_and_eval(). The first key is the dataset name, and the second key is the name of the metric. The value contained is a float value of the specified metric on the specified dataset. An example of accessing its data could thus be results['UMLS']['mrr'], which will return the MRR value TWIG acheived on the UMLS dataset.
    '''
    print("running training and eval")
    results = run_training(
        model=model,
        training_dataloaders=training_dataloaders,
        testing_dataloaders=testing_dataloaders,
        valid_dataloaders=valid_dataloaders,
        epochs=epochs,
        optimizer=optimizer,
        npp=npp,
        negative_samplers=negative_samplers,
        loss_fn=loss_function,
        verbose=verbose,
        model_name_prefix=model_name_prefix,
        checkpoint_every_n=checkpoint_every_n,
        valid_every_n=valid_every_n,
        early_stopper=early_stopper
    )
    print("done with training and eval")
    return results

def main(
        version,
        dataset_names,
        epochs,
        optimizer_args,
        optimizer,
        normalisation,
        batch_size,
        batch_size_test,
        npp,
        use_train_filter,
        use_valid_and_test_filters,
        sampler_type,
        loss_function,
        fts_blacklist,
        hyp_validation_mode,
        preexisting_model=None,
        early_stopper=None
    ):
    '''
    main() coordinates all of the major modules of TWIG, loads and prepares data, the NN, filters, and model hyparapmeters, (etc). It then calls the training loop, and finally conducts an evaluation of TWIG-I's performance. Since all of these functionalities are implemented in various modules, main() is more or less just a list of function calls that coordinate them all.

    For a list of arguments and their meaning, please see the documentation below `if __name__ == '__main__'`. Only arguments are notsent there are listed here.
    - preexisting_model: (torch.nn.Module) a previously saved TWIG-I model (such as a checkpoint).
    
    The values it returns are:
        - results (dict str -> str -> float): the results output from train_and_eval(). The first key is the dataset name, and the second key is the name of the metric. The value contained is a float value of the specified metric on the specified dataset. An example of accessing its data could thus be results['UMLS']['mrr'], which will return the MRR value TWIG acheived on the UMLS dataset.
    '''
    # load datasets (as PyTorch Dataloaders) and normalisation functions (they
    # are needed again in the training loopo for negative generation)
    dataloaders, norm_funcs, X_pos, n_local = load_dataset(
        dataset_names,
        normalisation=normalisation,
        batch_size=batch_size,
        batch_size_test=batch_size_test,
        fts_blacklist=fts_blacklist
    )

    # load user-specified fulters for the datasets for the training and the
    # test / validation phases
    filters = load_filters(
        dataset_names=dataset_names,
        use_train_filter=use_train_filter,
        use_valid_and_test_filters=use_valid_and_test_filters
    )

    # load all negative samplers (each dataset needs its own, since each one
    # has different nodes and edges used for negative generation). Filters are
    # provided so that they can be used in the train / test / valid phases as 
    # needed.
    negative_samplers = load_negative_samplers(
        dataset_names=dataset_names,
        filters=filters,
        norm_funcs=norm_funcs,
        sampler_type=sampler_type,
        fts_blacklist=fts_blacklist,
        X_pos=X_pos
    )

    # load (or create from scratch) the model we will be training
    if preexisting_model is not None:
        print('Using provided pre-existing model')
        model = preexisting_model
    else:
        print('Creating a new model from scratch')
        model = load_nn(version=version, n_local=n_local)

    # load an optimizer for the TWIG-I model
    optimizer = load_optimizer(
        optimizer_name=optimizer,
        model=model,
        **optimizer_args
    )

    # load loss function
    loss_function = load_loss_function(loss_function)

    # load the early stopper and configute testing / validation data
    if hyp_validation_mode:
        print('Running in hyperparameter evaluation mode')
        print('TWIG will be evaulaited on the validation set')
        print('and will not be tested each epoch on the validation set')
        valid_every_n = -1
        early_stopper_mode = "never"
        data_to_test_on = dataloaders['valid']
    else:
        print('Running in standard evaluation mode')
        print('TWIG will be evaulaited on the test set')
        print('and will not be tested each epoch on the validation set')
        valid_every_n = 5
        early_stopper_mode = "on-falter"
        data_to_test_on = dataloaders['test']
    if not early_stopper:
        early_stopper = load_early_stopper(
            start_epoch=5,
            patience=10,
            mode=early_stopper_mode
        )

    # save hyperparameter settings
    model_name_prefix = 'chkpt-ID_' + str(int(random.random() * 10**16))
    checkpoint_config_name = os.path.join(checkpoint_dir, f'{model_name_prefix}.pkl')
    with open(checkpoint_config_name, 'wb') as cache:
        to_save = {
            "version": version,
            "dataset_names": dataset_names,
            "epochs": epochs,
            "optimizer": optimizer_name,
            "optimizer_args": optimizer_args,
            "lr": lr,
            "normalisation": normalisation,
            "batch_size": batch_size,
            "batch_size_test": batch_size_test,
            "npp": npp,
            "use_train_filter": use_train_filter,
            "use_valid_and_test_filters": use_valid_and_test_filters,
            "sampler_type": sampler_type,
            "loss_function": loss_function,
            "fts_blacklist": fts_blacklist,
            "hyp_validation_mode": hyp_validation_mode,
            "early_stopper": early_stopper
        }
        pickle.dump(to_save, cache)

    # finally, we call the training and evaluation loops
    results = train_and_eval(
        model=model,
        training_dataloaders=dataloaders['train'],
        testing_dataloaders=data_to_test_on,
        valid_dataloaders=dataloaders['valid'],
        epochs=epochs,
        optimizer=optimizer,
        npp=npp,
        negative_samplers=negative_samplers,
        loss_function=loss_function,
        verbose=True,
        model_name_prefix=model_name_prefix,
        checkpoint_every_n=5,
        valid_every_n=valid_every_n,
        early_stopper=early_stopper
    )
    return results

if __name__ == '__main__':
    '''
    This section just exists to collect command line arguments and pass them to the main() function. Command line arguments are converted to the correct type (such as int or boolean) before being passed on.

    The command-line arguments accepted, and their meaning, are described below:
        - version: the version of the Neural Network to run. Default: "base"
        - dataset_names: the name (or names) of datasets to train and test on. This should either be a dataset natively present in PyKEEN (https://github.com/pykeen/pykeen#datasets), or a dataset (pre-split into train, test, valid splits) present in the local `./custom_datasets/` folder as [dataset_name].<train/test/valid>. If multiple are given, their training triples will be concatenated into a single tensor that is trained on, and testing will be done individually on each testing set of each dataset. When multiple datasets are given, they should be delimited by "_", as in "UMLS_DBpedia50_Kinships"
        - epochs: the number of epochs to train for
        - lr: the learning rate to use during training. We do not use an LR scheduler, so the learning rate will not update during training. In all, cases, the default implementation of the Adam optimizer is used
        - optimizer_name: the optimizer to use to update model weights during learning.
        - normalisation: the normalisation method to be used when loading data (and when created vectorised forms of negatively sampled triples). "zscore", "minmax", and "none" are currently supported.
        - batch_size: the batch size to use while training
        - npp: the number of negative samples to generate per positive triple during training
        - use_train_filter: "1" if negative samples generated during training should be filtered, "0" if they should not be filtered. Note tht "0" is the literature default and matches the method used in the TWIG-I paper
        - use_valid_and_test_filters: "1" if negative samples generated during validation and testing should be filtered, "0" if   they should not be filtered. Note that "1" is the literature default and matches the method used in the TWIG-I paper
        - sampler_type: the type of negative sampler to use. Valid options are "simple" (standard link prediction negative sampling) and "vector" (experimental, seems to have low performance and longer runtimes)
        - loss_function: the loss function to be used. Valid options are "margin-ranking" and "pairwise-logistic". If margin rankking is used, the margin must be given in parentheses; i.e. "margin-ranking(0.1)".
        - fts_blacklist: A space (" ")-delimited list of feature names that should NOT be used during training. This is provided to allow for feature ablation studies. If you want to use all feartures (the default), enter "None" (or any other invalid feature name). All currently possible options, based on the implemented features, are: 
            s_deg,
            o_deg,
            p_freq,
            s_p_cofreq,
            o_p_cofreq,
            s_o_cofreq
            s_min_deg_neighbour,
            s_max_deg_neighbour,
            s_mean_deg_neighbour,
            s_num_neighbours,
            s_min_freq_rel,
            s_max_freq_rel,
            s_mean_freq_rel,
            s_num_rels,
            o_min_deg_neighbour,
            o_max_deg_neighbour,
            o_mean_deg_neighbour,
            o_num_neighbours,
            o_min_freq_rel,
            o_max_freq_rel,
            o_mean_freq_rel,
            o_num_rels
        - hyp_validation_mode: "1" if TWIG-I should be evaluated on the *validation* set (as done when doing hyperparameter testing); "0" if it should be evaluated on the test dataset (as done in the evaluation setting to get final performance statistics)
          
    Once all data is collected and converted to its correct data type, main() is called with it as arguments.
    '''
    print('Arguments received:', sys.argv)
    version = sys.argv[1]
    dataset_names = sys.argv[2].split('_')
    epochs = int(sys.argv[3])
    lr = float(sys.argv[4])
    optimizer_name = sys.argv[5]
    normalisation = sys.argv[6]
    batch_size = int(sys.argv[7])
    batch_size_test = int(sys.argv[8])
    npp = int(sys.argv[9])
    use_train_filter = sys.argv[10] == '1'
    use_valid_and_test_filters = sys.argv[11] == '1'
    sampler_type = sys.argv[12]
    loss_function = sys.argv[13]
    fts_blacklist = set(sys.argv[14].split(' '))
    hyp_validation_mode = sys.argv[15] == '1'

    main(
        version=version,
        dataset_names=dataset_names,
        epochs=epochs,
        optimizer_args={"lr": lr},
        optimizer_name=optimizer_name,
        normalisation=normalisation,
        batch_size=batch_size,
        batch_size_test=batch_size_test,
        npp=npp,
        use_train_filter=use_train_filter,
        use_valid_and_test_filters=use_valid_and_test_filters,
        sampler_type=sampler_type,
        loss_function=loss_function,
        fts_blacklist=fts_blacklist,
        hyp_validation_mode=hyp_validation_mode
    )
