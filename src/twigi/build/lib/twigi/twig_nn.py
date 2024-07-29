# external imports
import torch
from torch import nn

'''
==========================
Neural Network Definitions
==========================
'''
class TWIGI_Base(nn.Module):
    # this is the "standard" TWIG-I version
    def __init__(self, n_local):
        '''
        init() init creates the neural network object and initialises its parameters. It also defines all layers used in learned. For an overview of the neural architecture, please see comments on the forward() function.

        The arguments it accepts are:
            - n_local (int)the number of graph structural features that are present in the input features vectors.

        The values it returns are:
            - None
        '''
        super().__init__()
        self.n_local = n_local #22

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_struct_1 = nn.Linear(
            in_features=n_local,
            out_features=10
        )
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=1e-2)
        
        self.linear_struct_2 = nn.Linear(
            in_features=10,
            out_features=10
        )
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=1e-2)

        self.linear_final = nn.Linear(
            in_features=10,
            out_features=1
        )
        self.sigmoid_final = nn.Sigmoid()

    def forward(self, X):
        '''
        forward() defines the forward pass of the NN and the neural architecture. For TWIG-I Base, this achitecture is approximately as follows.
            - all features are passed through three dense layers separated by ReLU activation.

        The arguments it accepts are:
            - X (Tensor): a tensor with feature vectors as rows, and as many rows as the batch size that is in use.

        The values it returns are:
            - X (Tensor): tensor with the same number of rows as the input, but only one value in each row, wich represents the score of the triple that was described by that row's feature vector.
        '''
        X = self.linear_struct_1(X)
        X = self.relu_1(X)
        X = self.dropout_1(X)

        X = self.linear_struct_2(X)
        X = self.relu_2(X)
        X = self.dropout_2(X)

        X = self.linear_final(X)
        X = self.sigmoid_final(X) #maybe use softmax instead,but I like this for now theoretically
        return X


class TWIGI_Linear(nn.Module):
    def __init__(self, n_local):
        '''
        init() init creates the neural network object and initialises its parameters. It also defines all layers used in learned. For an overview of the neural architecture, please see comments on the forward() function.

        The arguments it accepts are:
            - n_local (int): the number of graph structural features that are present in the input features vectors.

        The values it returns are:
            - None
        '''
        super().__init__()
        self.n_local = n_local #22

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_final = nn.Linear(
            in_features=n_local,
            out_features=1
        )
        self.sigmoid_final = nn.Sigmoid()

    def forward(self, X):
        '''
        forward() defines the forward pass of the NN and the neural architecture. For TWIG-I Linear, this is done as a basic linear regression task, without multiple hidden layers.

        The arguments it accepts are:
            - X (Tensor): a tensor with feature vectors as rows, and as many rows as the batch size that is in use.

        The values it returns are:
            - X (Tensor): tensor with the same number of rows as the input, but only one value in each row, wich represents the score of the triple that was described by that row's feature vector.
        '''
        X = self.linear_final(X)
        X = self.sigmoid_final(X)
        return X
