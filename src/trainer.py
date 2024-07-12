import torch
import os
import time
from negative_sampler import *

'''
===============
Reproducibility
===============
'''
torch.manual_seed(17)

'''
====================
Constant Definitions
====================
'''
device = "cuda"

'''
================
Helper Functions
================
'''
def do_batch(
        model,
        loss_fn,
        X_pos,
        batch,
        npp,
        negative_samplers,
        dateset_name,
        num_total_batches,
        phase, #train, valid, or test
        verbose=False
    ):
    '''
    do_batch() runs a single batch on TWIG. Overall, what this involves is:
        - generative negative triples for all positives in the batch
        - scoring the positive and negative triples
        - calculating a loss functions based on the scores of poitive and negative triples
    
    The arguments it accepts are:
        - model (torch.nn.Module): the PyTorch NN Model object containing TWIG's neural architecture.
        - loss_fn (func): a function from loss.py that calculates loss from tensors of positive and negative triples scores.
        - X_pos (Tensor): a 2D matrix where rows are triple vectors and columns represent different features.
        - batch (int): the number of the batch in the current epoch
        - npp (int): the number of negative samples to generate per positive triple
        - negative_samplers (dict str -> Simple_Negative_Sampler): a dict that maps a dataset name to the negative sampler associated with that dataset.
        - dataset_name (str): The name of the dataset from which the triples in the batch are drawn. Note that all data for a given abtch will only ever be drawn from one dataset.
        - num_total_batches (int): the total number of batches that are run per epoch. Only used when outputting data on epoch progress.`
        - phase (str): "train", "valid", or "test" -- the phase of training / evaluation in which this batch is being run.
        - verbose (bool): whether to output loss values occaisonally at certain batch numbers.
    
    The values it returns are:
        - loss_val (Tensor): the loss value calculated for this batch, reduced to a single scalar value.
        - score_pos (Tensor): a Tensor containing all calculated scores for  positive triples.
        - score_neg (Tensor): a Tenor containing all calculated scores for the negative triples.
        - npps (Tensor): a tensor containing the number of negatives per per positive that were used in the negative generation. All values in npps will be equal to the input npp unless upsampling is disabled in the negative sampler. 
    '''
    #first column (column 0) is the triple indicies; remove it and pass that to the negative sampler
    triple_idxs = X_pos[:, 0]
    X_pos = X_pos[:, 1:]

    assert phase in ["train", "valid", "test"], f"Invalid phase: {phase}"

    # get predicted data
    if phase != 'train':
        npp_flag = -1 #i.e. get all in test and valid
        sides = ('s', 'o')
    else:
        if type(negative_samplers[dateset_name]) is Optimised_Negative_Sampler:
            npp_flag = npp // 2
            sides = ('s', 'o') # ('both')
        else:
            npp_flag = npp
            sides = ('both',)

    results = {}
    for side in sides:
        X_neg, npps = negative_samplers[dateset_name].get_batch_negatives(
            purpose=phase,
            triple_idxs=triple_idxs,
            npp=npp_flag,
            side=side
        )
        assert torch.sum(npps) == X_neg.shape[0], f'{torch.sum(npps)} =/= {X_neg.shape}[0]'

        score_pos = model(X_pos)
        score_neg = model(X_neg)
        score_pos_expandeed = score_pos.repeat_interleave(npps, 0)
        loss_val = loss_fn(score_pos_expandeed, score_neg)
        results[side] = {
            'score_pos': score_pos,
            'score_neg': score_neg,
            'loss_val': loss_val,
            'npps': npps
        }

    if 'both' in sides:
        aggr_loss = results['both']['loss_val']
    elif 's' in sides and 'o' in sides:
        aggr_loss = (results['s']['loss_val'] + results['o']['loss_val']) / 2
    else:
        assert False, f"sides var set improperly: is {sides}"

    if batch % 500 == 0 and verbose:
        print(f"batch {batch} / {num_total_batches} loss: {aggr_loss.item()}")

    return aggr_loss, results

def prep_dataloaders(dataloaders):
    '''
    prep_dataloaders() prepares dataloaders to be used in training. Since each epoch is run with data from potentially multiple datasets, which all have different dataloaders, this function turns a name -> dataloader map into a list of iterators over dataloaders. This allows each dataloader to be used for one iteration for each batch, and for the iterators to track their progress such that the same initial batch is not used over and over.

    It also returns some data on batch size and the number of batches.

    The arguments it accepts are:
        - dataloaders (dict str -> torch.utils.data.DataLoader): a dict that maps the name of a dataset to the training dataloader created for that dataset.
    
    The values it returns are:
        - dl_num_batches (dict str -> int): a map of a dataset name to the number of batches used by the dataloader for that dataset.
        - num_total_batches (int): the total number of batches that will be run, across all dataloaders, in one epoch.
        - dataloaders_pos (list of tuple <iter over torch.utils.data.DataLoader, str>): a list of tuples whose fitst element is an iterator over a dataloader and whose second element is the name of the dataset who data the dataloader contains.
    '''
    dataloaders_pos = []
    for name in dataloaders:
        dataloaders_pos.append((iter(dataloaders[name]), name))
    dl_num_batches = {
        name: len(dataloaders[name]) for name in dataloaders
    }

    num_total_batches = 0
    for name in dl_num_batches:
        num_total_batches += dl_num_batches[name]
    return dl_num_batches, num_total_batches, dataloaders_pos

def train_epoch(
        dataloaders,
        model,
        loss_fn,
        npp,
        negative_samplers,
        optimizer,
        verbose=False
    ):
    '''
    train_epoch() runs a single epoch on input training data. As it does backpropogation, it should not be called during testing or validation. Overall, it iterates over the number of epochs requested, and then uses do_batch() to run all data batches in those epochs.

    The arguments it accepts are:
        - dataloaders (dict str -> torch.utils.data.DataLoader): a dict that maps the name of a dataset to the training dataloader created for that dataset.
        - model (torch.nn.Module): the PyTorch NN Model object containing WIG's neural architecture.
        - loss_fn (func): a function from loss.py that calculates loss from tensors of positive and negative triples scores.
        - npp (int): the number of negative samples to generate per positive triple
        - negative_samplers (dict str -> Simple_Negative_Sampler): a dict that maps a dataset name to the negative sampler associated with that dataset.
        - optimizer (torch.optim.Optimizer): the optimiser to use to backpropogate error during training. 
        - verbose (bool): whether to output loss values occaisonally at certain batch numbers.

    The values it returns are:
        - No values returned
    '''

    dl_num_batches, \
        num_total_batches, \
        dataloaders_pos = prep_dataloaders(dataloaders=dataloaders)

    batch = -1
    for dl, dateset_name in dataloaders_pos:
        for _ in range(dl_num_batches[dateset_name]):
            batch += 1
            X_pos = next(dl)
            assert len(X_pos) == 1
            X_pos = X_pos[0]

            loss_val, _,= do_batch(
                model=model,
                loss_fn=loss_fn,
                X_pos=X_pos,
                batch=batch,
                npp=npp,
                negative_samplers=negative_samplers,
                dateset_name=dateset_name,
                num_total_batches=num_total_batches,
                verbose=verbose,
                phase='train'
            )
            
            # Backpropagation
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()

def calc_ranks(score_pos_rows, score_negs, npps, mode):
    '''
    calc_ranks() calculated the rank of positive tripels among their negative corruptions. This function is typucally used when all possible negatives, except those given in filters (taken into account before this function call in the negative sampler).

    This function assumes that **the higher a score of a triple, the more likely that triple is considered to be true**. All loss functions must follow this rule, or else evaluation will suggest very poor results.

    The arguments it accepts are:
        - score_pos_rows (Tensor): The scores of all positive triples.
        - score_negs (Tensor): The score of all negatives generated. It has is ordered in the same order as score_pos_rows, which means that the first (idx 0) triple in score_pos_rows will have its negatives listed in the first npps[0]-many rows of score_negs
        - npps (Tensor): a tensor containing the number of negatives per per positive that were used in the negative generation. As filtering and upsampling (or not) can make this vary, it contains the number of negatives used for each positive in score_pos_rows, with the same size and in the same order as that Tensor.
        - mode (str): 'mean', 'optimisitic' or 'pessimistic'. How to deal with the rank of a positive triples that has the exact same score as one or more negative triples. 'mean' means it will have the average rank of all triples with the same score; 'optimisitic' means that it will give the best (lowest) rank to the true triple; 'pessimistic' means that it will give the worst (highest) rank to the true triple. Default 'mean'.

    The values it returns are:
        - ranks (Tensor): a tensor containing all ranks calculated for positive triples among their negative corruptions.
    '''
    ranks = []
    for i, score_pos in enumerate(score_pos_rows):
        num_corrs = npps[i]
        local_corr_scores = score_negs[i*num_corrs : (i+1)*num_corrs, :]
        if mode == 'optimisitic':
            ranks.append(
                torch.sum(local_corr_scores > score_pos) + 1 #min rank is 1
            )
        elif mode == 'pessimistic':
            ranks.append(
                torch.sum(local_corr_scores >= score_pos) + 1 #min rank is 1
            )
        elif mode == 'mean':
            ranks.append(
                (
                    (torch.sum(local_corr_scores > score_pos) + 1) \
                    + (torch.sum(local_corr_scores >= score_pos) + 1)
                ) / 2
            )

    ranks = torch.tensor(ranks)
    return ranks

def test(
        dataloaders,
        model,
        loss_fn,
        npp,
        negative_samplers,
        verbose=False,
        purpose='test'
    ):
    '''
    test() runs evaluation on the model using the data in the given dataloaders. This can be used either as a validation phase during training, or for evaluation afterwards. 

    **CRITICAL NOTE**: This function puts the model into eval() mode, but does not put it into train() mode after. As such, if called during training, you MUST run model.train() again to put it back in training mode and allow learning to continue correctly.

    The arguments it accepts are:
        - dataloaders (dict str -> torch.utils.data.DataLoader): a dict that maps the name of a dataset to the training dataloader created for that dataset.
        - model (torch.nn.Module): the PyTorch NN Model object containing TWIG's neural architecture.
        - loss_fn (func): a function from loss.py that calculates loss from tensors of positive and negative triples scores.
        - npp (int): the number of negative samples to generate per positive triple
        - negative_samplers (dict str -> Simple_Negative_Sampler): a dict that maps a dataset name to the negative sampler associated with that dataset.
        - verbose (bool): whether to output loss values occaisonally at certain batch numbers. Default False.
        - purpose (str): "train", "test", or "valid", the split (such as the testing or validation split) from which the dataloaders come. Default "test". This is just a text tag for printing, so you can add metadata to it if you'd like.

    The values it returns are:
        - results (dict str -> str -> float): the results output from train_and_eval(). The first key is the dataset name, and the second key is the name of the metric. The value contained is a float value of the specified metric on the specified dataset. An example of accessing its data could thus be results['UMLS']['mrr'], which will return the MRR value TWIG acheived on the UMLS dataset.
    '''
    model.eval()
    results = dict()

    print()
    print('=' * 34)
    with torch.no_grad():
        for dataloader_name in dataloaders:
            batch = -1
            test_loss = 0
            ranks_s = None
            ranks_o = None
            dataloader_pos = iter(dataloaders[dataloader_name])

            print(f'{purpose}: dataloader for dataset {dataloader_name}')
            for _ in range(len(dataloader_pos)):
                X_pos = next(dataloader_pos)
                assert len(X_pos) == 1
                X_pos = X_pos[0]
                batch += 1
                if batch % 1 == 0 and verbose:
                    print(f'{purpose}: batch {batch} / {len(dataloader_pos)}')

                loss_val, results = do_batch(
                    model=model,
                    loss_fn=loss_fn,
                    X_pos=X_pos,
                    batch=batch,
                    npp=npp,
                    negative_samplers=negative_samplers,
                    dateset_name=dataloader_name,
                    num_total_batches=len(dataloader_pos),
                    verbose=2, #was False
                    phase='test'
                )
                
                # compute metrics
                test_loss += loss_val.item()

                # get subject ranked list
                ranks_si = calc_ranks(
                    score_pos_rows=results['s']['score_pos'],
                    score_negs=results['s']['score_neg'],
                    npps=results['s']['npps'],
                    mode='mean'
                )
                if ranks_s is None:
                    ranks_s = ranks_si
                else:
                    ranks_s = torch.concat(
                        [ranks_s, ranks_si],
                        dim=0
                    )

                # get object ranked list
                ranks_oi = calc_ranks(
                    score_pos_rows=results['o']['score_pos'],
                    score_negs=results['o']['score_neg'],
                    npps=results['o']['npps'],
                    mode='mean'
                )
                if ranks_o is None:
                    ranks_o = ranks_oi
                else:
                    ranks_o = torch.concat(
                        [ranks_o, ranks_oi],
                        dim=0
                    )

                # get full ranked list
                ranks = torch.concat(
                    [ranks_s, ranks_o],
                    dim=0
                )

            mr = torch.mean(ranks)
            mrr = torch.mean(1 / ranks)
            h1 = torch.sum(ranks <= 1) / ranks.shape[0]
            h3 = torch.sum(ranks <= 3) / ranks.shape[0]
            h5 = torch.sum(ranks <= 5) / ranks.shape[0]
            h10 = torch.sum(ranks <= 10) / ranks.shape[0]

            results[dataloader_name] = {
                'test_loss': test_loss,
                'mr': mr,
                'mrr': mrr,
                'h1': h1,
                'h3': h3,
                'h5': h5,
                'h10': h10,
            }

            print("total number of ranks,", ranks.shape)
            if verbose:
                print('====== Ranks ======')
                print(f'ranks size: {ranks.shape}')
                # for r in ranks:
                #   print(float(r))
                # print()
            print(f'test_loss: {test_loss}')
            print(f'mr: {mr}')
            print(f'mrr: {mrr}')
            print(f'h1: {h1}')
            print(f'h3: {h3}')
            print(f'h5: {h5}')
            print(f'h10: {h10}')

    print('=' * 34)
    print()

    return results

def run_training(
        model,
        training_dataloaders,
        testing_dataloaders,
        valid_dataloaders,
        epochs,
        optimizer,
        npp,
        negative_samplers,
        loss_fn,
        verbose,
        model_name_prefix,
        checkpoint_dir,
        checkpoint_every_n,
        valid_every_n
    ):
    '''
    run_training() is the main interface through which the trainer is acessed. It first trains a model, and then runs evaluation on it. It is also responsible for making sure all data is in the right format for learning, and for setting certain hyperparameters to be used during learning.

    The arguments it accepts are:
        - model (torch.nn.Module): the PyTorch NN Model object containing TWIG's neural architecture.
        - training_dataloaders (dict str -> torch.utils.data.DataLoader): a dict that maps the name of a dataset to the training dataloader created for that dataset.
        - testing_dataloaders (dict str -> torch.utils.data.DataLoader): a dict that maps the name of a dataset to the testing dataloader created for that dataset.
        - valid_dataloaders (dict str -> torch.utils.data.DataLoader): a dict that maps the name of a dataset to the validation dataloader created for that dataset.
        - epochs (int): the number of epochs to train for
        - lr (float): the learning rate to use during training
        - npp (int): the number of negative samples to generate per positive triple during training
        - negative_samplers (dict str -> Simple_Negative_Sampler): a dict that maps a dataset name to the negative sampler associated with that dataset.
        - verbose (bool): whether to output loss values occaisonally at certain batch numbers.
        - model_name_prefix (str): prefix to the name of saved checkpoints.
        - checkpoint_dir (str): directiory in which to store checkpoints.
        - checkpoint_every_n (int): not used.
        - valid_every_n (int): the number of epochs after which validation should be run on the validation dataset to track training. Currently also reports performance on the training dataset to give a measure of how everfir the model may be.

    The values it returns are:
        - No values are returned.
    '''
    model.to(device)
    print(model)

    # Training
    model.train()
    print(f'REC: Training with epochs = {epochs}')

    for t in range(epochs):
        print(f"Epoch {t+1} -- ", end='')
        train_epoch(
            dataloaders=training_dataloaders,
            model=model,
            loss_fn=loss_fn,
            npp=npp,
            negative_samplers=negative_samplers,
            optimizer=optimizer, 
            verbose=verbose
        )
        if (t+1) % checkpoint_every_n == 0:
            print(f'Saving checkpoint at epoch {t+1}; prefix = {model_name_prefix}')
            state_data = f'e{t+1}'
            torch.save(
                model,
                    os.path.join(
                        checkpoint_dir,
                        f'{model_name_prefix}_{state_data}.pt'
                    )
                )
        if valid_every_n > 0 and (t+1) % valid_every_n == 0 and t+1 != epochs:
            test(
                dataloaders=valid_dataloaders,
                model=model,
                loss_fn=loss_fn,
                npp=npp,
                negative_samplers=negative_samplers,
                verbose=True,
                purpose='Validation'
            )
            model.train() # put back in training mode (test puts it in eval mode)
    print("Done Training!")

    # Testing
    # we do it for each DL since we want to do each dataset testing separately for now
    test(
        dataloaders=testing_dataloaders,
        model=model,
        loss_fn=loss_fn,
        npp=npp,
        negative_samplers=negative_samplers,
        verbose=True,
        purpose='Testing (cite this)'
    )
    print("Done Testing!")
