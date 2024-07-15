# imports from twigi
from run_exp import main, load_dataset, load_filters, load_negative_samplers, train_and_eval, load_nn, load_loss_function
from twig_nn import *
from early_stopper import Early_Stopper
from negative_sampler import *

# external imports
import inspect

def do_job(
        model,
        dataset_names,
        optimizer=None,
        negative_sampler=None,
        loss_function=None,
        early_stopper=None,
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
        tag=None
):
    # assign defaults if needed
    if not negative_sampler:
        negative_sampler = "simple"
    if not loss_function:
        loss_function = "margin-ranking(0.1)"
    if not early_stopper:
        early_stopper = Early_Stopper(
            start_epoch=0,
            patience=0,
            mode="never",
            precision=3
        )
    if not "normalisation" in data_args:
        data_args["normalisation"] = "zscore"
    if not "batch_size" in data_args:
        data_args["batch_size"] = 64
    if not "batch_size_test" in data_args:
        data_args["batch_size_test"] = 128
    if not "fts_blacklist" in data_args:
        data_args["fts_blacklist"] = []
    if not "epochs" in training_args:
        training_args["epochs"] = 20
    if not "npp" in training_args:
        training_args["npp"] = 30
    if not "hyp_validation_mode" in training_args:
        training_args["hyp_validation_mode"] = False
    if not tag:
        tag = "super-cool-model"

    # correct input types if needed
    if type(dataset_names) is str:
        dataset_names = [dataset_names]
    
    # we first need to laod the data from the given dataset names
    dataloaders, norm_funcs, X_pos, n_local = load_dataset(
        dataset_names,
        normalisation=data_args["normalisation"],
        batch_size=data_args["batch_size"],
        batch_size_test=data_args["batch_size_test"],
        fts_blacklist=data_args["fts_blacklist"],
    )
    if training_args["hyp_validation_mode"]:
        valid_every_n = -1
        early_stopper_mode = "never"
        data_to_test_on = dataloaders['valid']
    else:
        valid_every_n = 5
        early_stopper_mode = "on-falter"
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
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

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
    elif type(negative_sampler) is Negative_Sampler:
        pass #user-defined negative sampler (already instantiated)
    else:
        assert False, f"Unsupported input for negative sampler: {negative_sampler}"

    # we now load the loss function
    if type(loss_function) is str:
        loss_function = load_loss_function(loss_fn_name=loss_function)
    elif callable(loss_function):
        pass #user-defined loss, already instantiated

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

def ablation_job():
    pass

def finetune_job():
    pass
