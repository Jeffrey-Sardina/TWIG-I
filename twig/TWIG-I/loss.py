import torch

def get_margin_ranking_loss(reduction, margin):
    '''
    get_margin_ranking_loss() creates a margin ranking loss function using the given parameters for use by TWIG-I. That function is returned and can be used as a standard loss function.
        
    The arguments it accepts are:
        - reduction (str): 'mean' or 'sum', the method why which loss values are reduced from tensors into single scalars.
        - margin (float), the margin value to use when computing margin ranking loss. This represents the desired margine the model should enforce between the values of the scores of positive and negative triples.

    The values it returns are:
        - local_margin_ranking_loss (func): a callable function that computes margin ranking loss for TWIG-I using the given reduction method and margin value.
    '''
    # make sure the given reduction is valid
    if reduction == 'mean':
            reduce = torch.mean
    elif reduction == 'sum':
        reduce = torch.sum
    else:
        assert False, f"invalid reduction: {reduction}"
    
    # make sure the given hyperparameters are valid
    assert type(margin) == float and margin > 0, f"Margin must be a float >= 0 but is {margin}"

    def local_margin_ranking_loss(score_pos, score_neg):
        '''
        margin_ranking_loss() is an implementation of Margin Ranking Loss for TWIG. It is calculated as max((score_neg - score_pos + margin), 0).

        Special arguments:
            - "margin" -> float, the margin value to use when computing margin ranking loss. This represents the desired margine the model should enforce between the values of the scores of positive and negative triples.

        The arguments it accepts are:
            - score_pos (Tensor): the scores of all positive triples. Note when npp > 1, score_pos MUST have its rows expanded such that each negative score in score_neg at index i has, at index i in score_pos, the score of the positive triple from which it comes. This means that score_pos will have repeated elements.
            - score_neg (Tensor) the scores of all negative triples, with indicies to, those in score_pos.
            - reduction (str): 'mean' or 'sum', the method why which loss values are reduced from tensors into single scalars. Default 'mean'.

        The values it returns are:
            - loss (Tensor): the calculated loss value as a single float scalar.
        '''
        loss = reduce(
            (score_neg - score_pos + margin).relu()
        )
        return loss
    return local_margin_ranking_loss

def get_pairwise_logistic_loss(reduction):
    '''
    get_pairwise_logistic_loss() creates a pairwise logistic loss function using the given parameters for use by TWIG-I. That function is returned and can be used as a standard loss function.
        
    The arguments it accepts are:
        - reduction (str): 'mean' or 'sum', the method why which loss values are reduced from tensors into single scalars.

    The values it returns are:
        - local_pairwise_logistic_loss (func): a callable function that computes pairwise logistic loss for TWIG-I using the given reduction method.
    '''
    # make sure the given reduction is valid
    if reduction == 'mean':
            reduce = torch.mean
    elif reduction == 'sum':
        reduce = torch.sum
    else:
        assert False, f"invalid reduction: {reduction}"

    def local_pairwise_logistic_loss(score_pos, score_neg):
        '''
        pairwise_logistic_loss() is an implementation of Pairwise Logistic Loss for TWIG. It is calculated as log(1 + exp((score_neg - score_pos))

        The arguments it accepts are:
            - score_pos (Tensor): the scores of all positive triples. Note when npp > 1, score_pos MUST have its rows expanded such that each negative score in score_neg at index i has, at index i in score_pos, the score of the positive triple from which it comes. This means that score_pos will have repeated elements.
            - score_neg (Tensor) the scores of all negative triples, with indicies to, those in score_pos.
            - reduction (str): 'mean' or 'sum', the method why which loss values are reduced from tensors into single scalars. Default 'mean'.

        The values it returns are:
            - loss (Tensor): the calculated loss value as a single float scalar.
        '''
        loss = reduce(
            torch.log(1 + torch.exp((score_neg - score_pos)))
        )
        return loss
    return local_pairwise_logistic_loss
