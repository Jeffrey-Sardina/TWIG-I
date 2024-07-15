# imports from twigi
from run_exp import main

# external imports
import pickle
import torch
import sys
import json

def load_and_run_from_chkpt(
        torch_checkpont_path,
        model_config_path,
        model_config_override
    ):
    '''
    load_and_run_from_chkpt() loads a checkpoint, TWIG-I hyperparameters / settings, and overrides to thsoe settings, and uses them to run more training for TWIG-I as they specify. This function really only does the actuial laod -- all training is done by handing this off to `run_exp.py` to avoid code duplication.

    For a list of all arguments accepted and what they mena, please see the documentation below `if __name__ == '__main__'`.

    The values it returns are:
        - results (dict str -> str -> float): the results output from train_and_eval(). The first key is the dataset name, and the second key is the name of the metric. The value contained is a float value of the specified metric on the specified dataset. An example of accessing its data could thus be results['UMLS']['mrr'], which will return the MRR value TWIG acheived on the UMLS dataset.
    '''
    # load original config
    with open(model_config_path, 'rb') as cache:
        print('loading model settings from cache:', model_config_path)
        model_config = pickle.load(cache)

    # NOTE: you may want to override datasets to test on new datasets in the
    # few- or zero- shot setting
    if type(model_config_override) is str:
        with open(model_config_override_path) as inp:
            model_config_override = json.load(inp)
    for key in model_config_override:
        print(f'overriding original values for {key}. Was {model_config[key]}, now is {model_config_override[key]}')
        model_config[key] = model_config_override[key]
    if not 'epochs' in model_config_override:
        assert False, "A new number of epoch, at least, must be given in the override config"
    print(f'It will be trained for {model_config["epochs"]} more epochs now.')
    print('If you are loading a modle that was itself loaded from checkpoint, the number of epochs listed as already completed above will be incorrect')
    print('until I get around to fxing this, you will have to recursively go through logs to find the total number of epochs run')

    # load the checkpointed model
    print('loadng TWIG-I model from disk at:', torch_checkpont_path)
    model = torch.load(torch_checkpont_path)

    # run checkpointed model with new config
    print(f'the full config being used is: {model_config}')
    results = main(
        version=model_config['version'],
        dataset_names=model_config['dataset_names'],
        epochs=model_config['epochs'],
        lr=model_config['lr'],
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
        preexisting_model=model
    )
    return results

if __name__ == '__main__':
    '''
    This section gathers all data needed from the command line to run the load_and_run_from_chkpt() function, which is in essence this file's main function.

    The command-line arguments accepted, and their meaning, are described below:
        - torch_checkpont_path: the path to the checkpoint file written by torch that should be loaded. The default (every 5 epochs) checkpoints written by TWIG-I are located at `./checkpoints//[checkpoint_id]_e[num_epochs].pt`
        - model_config_path: the path to the saved faile containing a serialisation of all command-line arguments given to the original model for training (this means that when you load a chekcpoint, you can use the same hyperparameters, datasets, settings etc as specifiied in this fuile without any further effort). By default, TWIG-I will write this to `./checkpoints/[checkpoint_id].pkl`
        - model_config_override_path: the path to a custom-user made override file to specify new hyperparameters, datasets, etc to be used with the loaded model. For example, you can use a saved TWIG-I checkpoint as a pretrained model to then fine-tune on a new dataset, or specify more epochs to run to continue training. NOTE: This file MUST be created as a .json file and MUST contain, at a minimum, the line `"eppochs": X` to specify hw many more epochs to run. TWIG-I does not currently know how many epochs a model was run for, so if you want to finish training after a crash, for example, you need to manually tell if how many more epochs it needs to do.

    Please note that checkpoint_id takes the form of something like `chkpt-ID_1726265127922990`, and will be printed in the log file that TWIG-I created as it was running your experiment. (Hint: find it using CRTL-F for "chkpt-ID_" :)) )

    Once all data is collected and converted to its correct data type, load_and_run_from_chkpt() is called with it as arguments.
    '''
    torch_checkpont_path = sys.argv[1]
    model_config_path = sys.argv[2]
    model_config_override_path = sys.argv[3]
    load_and_run_from_chkpt(
        torch_checkpont_path,
        model_config_path,
        model_config_override_path
    )
    