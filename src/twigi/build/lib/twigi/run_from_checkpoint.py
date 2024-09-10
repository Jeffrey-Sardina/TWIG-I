# imports from twigi
from run_exp import main

# external imports
import pickle
import torch
import sys
import json

def load_model_config(model_config_path):
    '''
    load_model_config() loads a saved model config from disk.

    The arguments it accepts are:
        - model_config_path (str): the path to the model configuration file on disk

    The values it returns are;
        - model_config (dict of str -> any): a dictionary mapping each element of the model config, by name, to its value
    '''
    with open(model_config_path, 'rb') as cache:
        print('loading model settings from cache:', model_config_path)
        model_config = pickle.load(cache)
    return model_config

def load_override_config(model_config_override):
    '''
    load_override_config() loads settings to override (replacing the original model config).

    The arguments it accepts are:
        - model_config_path (str or dict): the path to the override configuration file on disk (if a str) or the override dict itself (if a dict)

    The values it returns are;
        - model_config_override (dict of str -> any): a dictionary mapping each element of the model config that should be overwritten, by name, to its value
    '''
    if type(model_config_override) is str:
        with open(model_config_override) as inp:
            model_config_override = json.load(inp)
    if not 'epochs' in model_config_override:
        assert False, "A new number of epoch, at least, must be given in the override config"
    return model_config_override

def apply_override(model_config, model_config_override):
    '''
    apply_override() applied a given override to the base model config, and returns the final result

    The arguments it accepts are:
        - model_config (dict of str -> any): a dictionary mapping each element of the model config, by name, to its value
        - model_config_override (dict of str -> any): a dictionary mapping each element of the model config that should be overwritten, by name, to its value

    The values it returns are:
        - model_config (dict of str -> any): a dictionary containing the model config with all overrides applied
    '''
    for key in model_config_override:
        print(f'overriding original values for {key}. Was {model_config[key]}, now is {model_config_override[key]}')
        model_config[key] = model_config_override[key]
    return model_config

def load_chkpt(
        torch_checkpont_path,
        model_config_path,
        model_config_override
    ):
    '''
    load_chkpt() loads a checkpoint, TWIG-I hyperparameters / settings, and overrides to those settings

    The arguments it accepts are:
        - torch_checkpont_path (str): the path to the checkpoint file written by torch that should be loaded. The default (every 5 epochs) checkpoints written by TWIG-I are located at `./checkpoints//[checkpoint_id]_e[num_epochs].pt`
        - model_config_path (str): the path to the saved faile containing a serialisation of all command-line arguments given to the original model for training (this means that when you load a chekcpoint, you can use the same hyperparameters, datasets, settings etc as specifiied in this fuile without any further effort). By default, TWIG-I will write this to `./checkpoints/[checkpoint_id].pkl`
        - model_config_override_path (str or dict): the path to a custom-user made override file to specify new hyperparameters, datasets, etc to be used with the loaded model or a pre-laoded dict containing that information. For example, you can use a saved TWIG-I checkpoint as a pretrained model to then fine-tune on a new dataset, or specify more epochs to run to continue training. NOTE: This file MUST be created as a .json file and MUST contain, at a minimum, the line `"eppochs": X` to specify hw many more epochs to run. TWIG-I does not currently know how many epochs a model was run for, so if you want to finish training after a crash, for example, you need to manually tell if how many more epochs it needs to do.

    The values it returns are:
        - model (torch.nn.Module): the loaded TWIG-I model to use
        - model_config (dict of str -> any): the training config to use to train the model
    '''
    # load the checkpointed model
    print('loadng TWIG-I model from disk at:', torch_checkpont_path)
    model = torch.load(torch_checkpont_path)

    # load config with override
    model_config = apply_override(
        model_config = load_model_config(model_config_path),
        model_config_override = load_override_config(model_config_override)
    )

    return model, model_config

def run_from_chkpt(model, model_config):
    '''
    run_from_chkpt() uses the given model and model configuration to run more training for TWIG-I (i.e. finetuning, or continuation) from where training last stopped. All training is done by handing this off to `run_exp.py` to avoid code duplication -- this function just gets the data in the fight order and format.

    The arguments it accepts are:
        - model (torch.nn.Module): the loaded TWIG-I model to use
        - model_config (dict of str -> any): the training config to use to train the model

    The values it returns are:
        - results (dict str -> str -> float): the results output from train_and_eval(). The first key is the dataset name, and the second key is the name of the metric. The value contained is a float value of the specified metric on the specified dataset. An example of accessing its data could thus be results['UMLS']['mrr'], which will return the MRR value TWIG acheived on the UMLS dataset.
    '''
    print(f'I loaded a model that will be trained for {model_config["epochs"]} more epochs now.')
    print('If you are running a model that was itself loaded from checkpoint, the number of epochs listed as already completed above will be incorrect')
    print('until I get around to fxing this, you will have to recursively go through logs to find the total number of epochs run')
    print(f'the full config being used is: {model_config}')
    results = main(
        version=model_config['version'],
        dataset_names=model_config['dataset_names'],
        epochs=model_config['epochs'],
        optimizer_args=model_config['optimizer_args'],
        optimizer_name=model_config['optimizer'],
        normalisation=model_config['normalisation'],
        batch_size=model_config['batch_size'],
        batch_size_test=model_config['batch_size_test'],
        npp=model_config['npp'],
        use_train_filter=model_config['use_train_filter'],
        use_valid_and_test_filters=model_config['use_valid_and_test_filters'],
        sampler_type=model_config['sampler_type'],
        loss_function=model_config['loss_function'],
        fts_blacklist=model_config['fts_blacklist'],
        hyp_validation_mode=model_config['hyp_validation_mode'],
        early_stopper=model_config['early_stopper'],
        preexisting_model=model
    )
    return results


if __name__ == '__main__':
    '''
    This section gathers all data needed from the command line to run the load_and_run_from_chkpt() function, which is in essence this file's main function.

    The command-line arguments accepted, and their meaning, are described below:
        - torch_checkpont_path (str): the path to the checkpoint file written by torch that should be loaded. The default (every 5 epochs) checkpoints written by TWIG-I are located at `./checkpoints//[checkpoint_id]_e[num_epochs].pt`
        - model_config_path (str): the path to the saved faile containing a serialisation of all command-line arguments given to the original model for training (this means that when you load a chekcpoint, you can use the same hyperparameters, datasets, settings etc as specifiied in this fuile without any further effort). By default, TWIG-I will write this to `./checkpoints/[checkpoint_id].pkl`
        - model_config_override_path (str or dict): the path to a custom-user made override file to specify new hyperparameters, datasets, etc to be used with the loaded model or a pre-laoded dict containing that information. For example, you can use a saved TWIG-I checkpoint as a pretrained model to then fine-tune on a new dataset, or specify more epochs to run to continue training. NOTE: This file MUST be created as a .json file and MUST contain, at a minimum, the line `"eppochs": X` to specify hw many more epochs to run. TWIG-I does not currently know how many epochs a model was run for, so if you want to finish training after a crash, for example, you need to manually tell if how many more epochs it needs to do.

    Please note that checkpoint_id takes the form of something like `chkpt-ID_1726265127922990`, and will be printed in the log file that TWIG-I created as it was running your experiment. (Hint: find it using CRTL-F for "chkpt-ID_" :)) )

    Once all data is collected and converted to its correct data type, load_and_run_from_chkpt() is called with it as arguments.
    '''
    torch_checkpont_path = sys.argv[1]
    model_config_path = sys.argv[2]
    model_config_override = sys.argv[3]
    model, model_config = load_chkpt(
        torch_checkpont_path=torch_checkpont_path,
        model_config_path=model_config_path,
        model_config_override=model_config_override
    )
    run_from_chkpt(model, model_config)
