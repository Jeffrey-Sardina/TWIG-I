# imports from twigi
from utils import get_triples, calc_graph_stats, get_triples_by_idx, load_custom_dataset

# external imports
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pykeen import datasets as pykeendatasets
from frozendict import frozendict
from array import array

'''
====================
Constant Definitions
====================
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Feature index maps
TOTAL_NUM_FTS = 22
S_DEG_IDX = 0
O_DEG_IDX = 1
P_FREQ_IDX = 2
S_P_COFREQ_IDX = 3
O_P_COFREQ_IDX = 4
S_O_COFREQ_IDX = 5
S_MIN_DEG_NEIGHBOUR_IDX = 6
S_MAX_DEG_NEIGHBOUR_IDX = 7
S_MEAN_DEG_NEIGHBOUR_IDX = 8
S_NUM_NEIGHBOURS_IDX = 9
S_MIN_FREQ_REL_IDX = 10
S_MAX_FREQ_REL_IDX = 11
S_MEAN_FREQ_REL_IDX = 12
S_NUM_RELS_IDX = 13
O_MIN_DEG_NEIGHBOUR_IDX = 14
O_MAX_DEG_NEIGHBOUR_IDX = 15
O_MEAN_DEG_NEIGHBOUR_IDX = 16
O_NUM_NEIGHBOURS_IDX = 17
O_MIN_FREQ_REL_IDX = 18
O_MAX_FREQ_REL_IDX = 19
O_MEAN_FREQ_REL_IDX = 20
O_NUM_RELS_IDX = 21

# note that this dict will underpeform the map above and so should be used minimally
ft_to_idx = {
    "s_deg": S_DEG_IDX,
    "o_deg": O_DEG_IDX,
    "p_freq": P_FREQ_IDX,

    "s_p_cofreq": S_P_COFREQ_IDX,
    "o_p_cofreq": O_P_COFREQ_IDX,
    "s_o_cofreq": S_O_COFREQ_IDX,

    "s_min_deg_neighbour": S_MIN_DEG_NEIGHBOUR_IDX,
    "s_max_deg_neighbour": S_MAX_DEG_NEIGHBOUR_IDX,
    "s_mean_deg_neighbour": S_MEAN_DEG_NEIGHBOUR_IDX,
    "s_num_neighbours": S_NUM_NEIGHBOURS_IDX,

    "s_min_freq_rel": S_MIN_FREQ_REL_IDX,
    "s_max_freq_rel": S_MAX_FREQ_REL_IDX,
    "s_mean_freq_rel": S_MEAN_FREQ_REL_IDX,
    "s_num_rels": S_NUM_RELS_IDX,

    "o_min_deg_neighbour": O_MIN_DEG_NEIGHBOUR_IDX,
    "o_max_deg_neighbour": O_MAX_DEG_NEIGHBOUR_IDX,
    "o_mean_deg_neighbour": O_MEAN_DEG_NEIGHBOUR_IDX,
    "o_num_neighbours": O_NUM_NEIGHBOURS_IDX,

    "o_min_freq_rel": O_MIN_FREQ_REL_IDX,
    "o_max_freq_rel:": O_MAX_FREQ_REL_IDX,
    "o_mean_freq_rel": O_MEAN_FREQ_REL_IDX,
    "o_num_rels": O_NUM_RELS_IDX,
}

# Cache index maps
TOTAL_CACHE_SIZE = 4
MIN_VALUE_IDX = 0
MAX_VALUE_IDX = 1
MEAN_VALUE_IDX = 2
NUM_VALUE_IDX = 3


'''
=======
Classes
=======
'''
class Structure_Loader():
    def __init__(
            self,
            triples_map,
            metadata,
            ents_to_triples,
            fts_blacklist
        ):
        '''
        __init__() initisalises the Structure_Loader object. This class really just serves to provide a common interfact to query structural characteristics, such that all structural features are calculated the same way.

        The arguments it accepts are:
            - triples_map (dict int -> tuple (int, int int)): a dict that maps a triple index to (s,p,o) integer IDs for nodes and edges. 
            - metadata (dict of str -> any): A dict containing graph stats for triples in the train set (only!). It is identical to graph_stats['train'] -- i.e. the calculated graph stats for only the training set. It has the format:
                - 'degrees': dict int (node ID) -> float (degree)
                - 'pred_freqs': dict int (edge ID) -> float (frequency count)
                - 'subj / obj / total _relationship_degrees': dict tuple <int, int> (pair of <subj/obj, predicate> IDs)-> float (co-occurrence count) 
                - 'percentiles': dict int (percentile) -> float (degree at that percentile)
                - 'subj / obj / total _rel_degree_percentiles': dict int (percentile) -> float (percentile of relationship_degrees as above)
            - ents_to_triples (dict int -> int): a dict that maps an entity ID to a list of all triples IDs (expressed as containing that original entity.
            - fts_blacklist (list of str): A  list of feature names that should NOT be calculated for negatives.

        The values it returns are:
            - None
        '''
        self.triples_map = triples_map
        self.metadata = metadata
        self.ents_to_triples = ents_to_triples
        self.fts_blacklist = fts_blacklist
        if not self.fts_blacklist or "None" in self.fts_blacklist or len(self.fts_blacklist) == 0:
            self.fts_blacklist = False
        self.neighbour_nodes, self.neighbour_preds = self.build_neighbour_cache()

    def build_neighbour_cache(self):
        '''
        build_neighbour_cache()  builds up a cache of the structural features describing the neighbourhood around each node in the gives triples. This is used to very quickly later on extract the 2-hop / corase grained features for a triple, as it can then just be read from this cache when needed.
        
        The arguments it accepts are:
            - None

        The values it returns are:
            - neighbour_nodes_stats (dict of int -> str -> float): a dict that maps an entitiy ID and the name of a metric to the value of that metric of its neibourhood degree distribtuion. For example, neighbour_nodes[node_id]['mean'] would return a float encoding the mean degree of all nodes around the node with ID node_id. 
            - neighbour_preds_stats (dict of int -> str -> float): a dict that maps an entitiy ID and the name of a metric to the value of that metric of its predicate frequency distribtuion. For example, neighbour_nodes[node_id]['mean'] would return a float encoding the mean frequency of all predicates around the node with ID node_id.
        '''
        neighbour_nodes = {}
        neighbour_preds = {}
        for target in self.ents_to_triples:
            neighbour_nodes[target] = {}
            neighbour_preds[target] = {}
            for t_idx in self.ents_to_triples[target]:
                t_s, t_p, t_o = self.triples_map[t_idx]
                ent = t_s if target != t_s else t_o
                if not ent in neighbour_nodes[target]:
                    neighbour_nodes[target][ent] = self.metadata['degrees'][ent]
                if not t_p in neighbour_preds[target]:
                    neighbour_preds[target][t_p] = self.metadata['pred_freqs'][t_p]

        neighbour_nodes_stats = {}
        for target in self.ents_to_triples:
            neighbour_nodes_stats[target] = array("f", [0 for _ in range(TOTAL_CACHE_SIZE)])
            neighbour_nodes_stats[target][MEAN_VALUE_IDX] = np.mean(
              list(neighbour_nodes[target].values())
            )
            neighbour_nodes_stats[target][MIN_VALUE_IDX] = np.min(
                list(neighbour_nodes[target].values())
            )
            neighbour_nodes_stats[target][MAX_VALUE_IDX] = np.max(
                list(neighbour_nodes[target].values())
            ) 
            neighbour_nodes_stats[target][NUM_VALUE_IDX] = len(neighbour_nodes[target])

        neighbour_preds_stats = {}
        for target in self.ents_to_triples:
            neighbour_preds_stats[target] = array("f", [0 for _ in range(TOTAL_CACHE_SIZE)])
            neighbour_preds_stats[target][MEAN_VALUE_IDX] = np.mean(
                list(neighbour_preds[target].values())
            )
            neighbour_preds_stats[target][MIN_VALUE_IDX] = np.min(
                list(neighbour_preds[target].values())
            )
            neighbour_preds_stats[target][MAX_VALUE_IDX] = np.max(
                list(neighbour_preds[target].values())
            )
            neighbour_preds_stats[target][NUM_VALUE_IDX] = len(neighbour_preds[target])

        neighbour_nodes_stats = frozendict(neighbour_nodes_stats)
        neighbour_preds_stats = frozendict(neighbour_preds_stats)
        return neighbour_nodes_stats, neighbour_preds_stats
    
    def create_negative_prefab_ocorr(self, s, p):
        prefab = [0 for _ in range(TOTAL_NUM_FTS)]

        prefab[S_DEG_IDX] = self.metadata['degrees'][s]
        prefab[P_FREQ_IDX] = self.metadata['pred_freqs'][p]

        prefab[S_P_COFREQ_IDX] = self.metadata['subj_relationship_degrees'][(s,p)] if (s,p) in self.metadata['subj_relationship_degrees'] else 0

        prefab[S_MIN_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[s][MIN_VALUE_IDX]
        prefab[S_MAX_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[s][MAX_VALUE_IDX]
        prefab[S_MEAN_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[s][MEAN_VALUE_IDX]
        prefab[S_NUM_NEIGHBOURS_IDX] = self.neighbour_nodes[s][NUM_VALUE_IDX]

        prefab[S_MIN_FREQ_REL_IDX] = self.neighbour_preds[s][MIN_VALUE_IDX]
        prefab[S_MAX_FREQ_REL_IDX] = self.neighbour_preds[s][MAX_VALUE_IDX]
        prefab[S_MEAN_FREQ_REL_IDX] = self.neighbour_preds[s][MEAN_VALUE_IDX]
        prefab[S_NUM_RELS_IDX] = self.neighbour_preds[s][NUM_VALUE_IDX]

        return prefab
    
    def vec_from_prefab_ocorr(self, prefab, s, p, o):
        prefab[O_DEG_IDX] = self.metadata['degrees'][o]
        prefab[O_P_COFREQ_IDX] = self.metadata['obj_relationship_degrees'][(o,p)] if (o,p) in self.metadata['obj_relationship_degrees'] else 0
        prefab[S_O_COFREQ_IDX] = self.metadata['subj_obj_cofreqs'][(s,o)] if (s,o) in self.metadata['subj_obj_cofreqs'] else 0

        prefab[O_MIN_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[o][MIN_VALUE_IDX]
        prefab[O_MAX_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[o][MAX_VALUE_IDX]
        prefab[O_MEAN_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[o][MEAN_VALUE_IDX]
        prefab[O_NUM_NEIGHBOURS_IDX] = self.neighbour_nodes[o][NUM_VALUE_IDX]

        prefab[O_MIN_FREQ_REL_IDX] = self.neighbour_preds[o][MIN_VALUE_IDX]
        prefab[O_MAX_FREQ_REL_IDX] = self.neighbour_preds[o][MAX_VALUE_IDX]
        prefab[O_MEAN_FREQ_REL_IDX] = self.neighbour_preds[o][MEAN_VALUE_IDX]
        prefab[O_NUM_RELS_IDX] = self.neighbour_preds[o][NUM_VALUE_IDX]

        if self.fts_blacklist:
            idxs_to_delete = [ft_to_idx[ft] for ft in self.fts_blacklist]
            for idx in reversed(sorted(idxs_to_delete)):
                del prefab[idx]

        return prefab
    
    def create_negative_prefab_scorr(self, p, o):
        prefab = [0 for _ in range(TOTAL_NUM_FTS)]

        prefab[O_DEG_IDX] = self.metadata['degrees'][o]
        prefab[P_FREQ_IDX] = self.metadata['pred_freqs'][p]

        prefab[O_P_COFREQ_IDX] = self.metadata['obj_relationship_degrees'][(o,p)] if (o,p) in self.metadata['obj_relationship_degrees'] else 0

        prefab[O_MIN_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[o][MIN_VALUE_IDX]
        prefab[O_MAX_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[o][MAX_VALUE_IDX]
        prefab[O_MEAN_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[o][MEAN_VALUE_IDX]
        prefab[O_NUM_NEIGHBOURS_IDX] = self.neighbour_nodes[o][NUM_VALUE_IDX]

        prefab[O_MIN_FREQ_REL_IDX] = self.neighbour_preds[o][MIN_VALUE_IDX]
        prefab[O_MAX_FREQ_REL_IDX] = self.neighbour_preds[o][MAX_VALUE_IDX]
        prefab[O_MEAN_FREQ_REL_IDX] = self.neighbour_preds[o][MEAN_VALUE_IDX]
        prefab[O_NUM_RELS_IDX] = self.neighbour_preds[o][NUM_VALUE_IDX]

        return prefab
    
    def vec_from_prefab_scorr(self, prefab, s, p, o):
        prefab[S_DEG_IDX] = self.metadata['degrees'][s]
        prefab[S_P_COFREQ_IDX] = self.metadata['subj_relationship_degrees'][(s,p)] if (s,p) in self.metadata['subj_relationship_degrees'] else 0
        prefab[S_O_COFREQ_IDX] = self.metadata['subj_obj_cofreqs'][(s,o)] if (s,o) in self.metadata['subj_obj_cofreqs'] else 0

        prefab[S_MIN_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[s][MIN_VALUE_IDX]
        prefab[S_MAX_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[s][MAX_VALUE_IDX]
        prefab[S_MEAN_DEG_NEIGHBOUR_IDX] = self.neighbour_nodes[s][MEAN_VALUE_IDX]
        prefab[S_NUM_NEIGHBOURS_IDX] = self.neighbour_nodes[s][NUM_VALUE_IDX]

        prefab[S_MIN_FREQ_REL_IDX] = self.neighbour_preds[s][MIN_VALUE_IDX]
        prefab[S_MAX_FREQ_REL_IDX] = self.neighbour_preds[s][MAX_VALUE_IDX]
        prefab[S_MEAN_FREQ_REL_IDX] = self.neighbour_preds[s][MEAN_VALUE_IDX]
        prefab[S_NUM_RELS_IDX] = self.neighbour_preds[s][NUM_VALUE_IDX]

        if self.fts_blacklist:
            print(ft_to_idx.keys())
            print(len(ft_to_idx.keys()))
            exit()
            idxs_to_delete = [ft_to_idx[ft] for ft in self.fts_blacklist]
            for idx in reversed(sorted(idxs_to_delete)):
                del prefab[idx]

        return prefab

    def __call__(self, s, p, o):
        '''
        __call__() returns the full structural feature vector for the given triple.

        The arguments it accepts are:
            - s (int): the ID of the subject node in the triple
            - p (int): the ID of the predicate in the triple
            - o (int): the ID of the object node in the triple

        The values it returns are:
            - data (list of float): the feature vector calculated for the input triple, as a list
        '''
        data = []
        s_deg = self.metadata['degrees'][s]
        o_deg = self.metadata['degrees'][o]
        p_freq = self.metadata['pred_freqs'][p]

        s_p_cofreq = self.metadata['subj_relationship_degrees'][(s,p)] if (s,p) in self.metadata['subj_relationship_degrees'] else 0
        o_p_cofreq = self.metadata['obj_relationship_degrees'][(o,p)] if (o,p) in self.metadata['obj_relationship_degrees'] else 0
        s_o_cofreq = self.metadata['subj_obj_cofreqs'][(s,o)] if (s,o) in self.metadata['subj_obj_cofreqs'] else 0
        
        fts1_all = {
            "s_deg": s_deg,
            "o_deg": o_deg,
            "p_freq": p_freq,
            "s_p_cofreq": s_p_cofreq,
            "o_p_cofreq": o_p_cofreq,
            "s_o_cofreq": s_o_cofreq,
        }

        if self.fts_blacklist:
            fts1_allowed = {key:fts1_all[key] for key in fts1_all if not key in self.fts_blacklist}
            data.extend(fts1_allowed.values())
        else:
            data.extend(fts1_all.values())

        target_dict = {'s': s, 'o': o}
        for target_name in target_dict:
            target = target_dict[target_name]

            min_deg_neighbour = self.neighbour_nodes[target][MIN_VALUE_IDX]
            max_deg_neighbour = self.neighbour_nodes[target][MAX_VALUE_IDX]
            mean_deg_neighbour = self.neighbour_nodes[target][MEAN_VALUE_IDX]
            num_neighbours = self.neighbour_nodes[target][NUM_VALUE_IDX]

            min_freq_rel = self.neighbour_preds[target][MIN_VALUE_IDX]
            max_freq_rel = self.neighbour_preds[target][MAX_VALUE_IDX]
            mean_freq_rel = self.neighbour_preds[target][MEAN_VALUE_IDX]
            num_rels = self.neighbour_preds[target][NUM_VALUE_IDX]

            fts2_all = {
                f"{target_name}_min_deg_neighbour": min_deg_neighbour,
                f"{target_name}_max_deg_neighbour": max_deg_neighbour,
                f"{target_name}_mean_deg_neighbour": mean_deg_neighbour,
                f"{target_name}_num_neighbours": num_neighbours,

                f"{target_name}_min_freq_rel": min_freq_rel,
                f"{target_name}_max_freq_rel": max_freq_rel,
                f"{target_name}_mean_freq_rel": mean_freq_rel,
                f"{target_name}_num_rels": num_rels,
            }

            if self.fts_blacklist:
                fts2_allowed = {key:fts2_all[key] for key in fts2_all if not key in self.fts_blacklist}
                data.extend(fts2_allowed.values())
            else:
                data.extend(fts2_all.values())

        return data

'''
=========
Functions
=========
'''
def get_adj_data(triples_map):
    '''
    get_adj_data() generates a mapping from every entity to all triples that contain it as a subject or object.

    The arguments it accepts are:
        - triples_map (dict int to tuple<int,int,int>): a dict mapping from a triple ID to the IDs of the three elements (subject, predicate, and object) that make up that triple. 

    The values it returns are:
        - ents_to_triples (dict int -> list of tuple<int,int,int>): a dict that maps an entity ID to a list of all triples (expressed as the IDs for the subj, pred, and obj) containing that original entity.
    '''
    ents_to_triples = {} # entity to all relevent data
    for t_idx in triples_map:
        s, _, o = triples_map[t_idx]
        if not s in ents_to_triples:
            ents_to_triples[s] = set()
        if not o in ents_to_triples:
            ents_to_triples[o] = set()
        ents_to_triples[s].add(t_idx)
        ents_to_triples[o].add(t_idx)
    return ents_to_triples

def get_twm_data_augment(
        triples_map,
        graph_stats,
        ents_to_triples,
        fts_blacklist=None
    ):
    '''
    get_twm_data_augment() generates all feature vectors for the given input data (triple IDs) and KG metadata.

    The arguments it accepts are:
        - triples_map (dict int to tuple<int,int,int>): a dict mapping from a triple ID to the IDs of the three elements (subject, predicate, and object) that make up that triple. 
        - graph_stats (dict of a lot of things): dict with the format:
              all / train / test / valid : 
              {
                  'degrees': dict int (node ID) -> float (degree)
                  'pred_freqs': dict int (edge ID) -> float (frequency count)
                  'subj / obj / total _relationship_degrees': dict tuple <int, int> (pair of <subj/obj, predicate> IDs)-> float (co-occurrence count) 
                  'percentiles': dict int (percentile) -> float (degree at that percentile)
                  'subj / obj / total _rel_degree_percentiles': dict int (percentile) -> float (percentile of relationship_degrees as above)
              } 
        - ents_to_triples (dict int -> list of tuple <int,int,int>): a dict that maps an entity ID to a list of all triples (expressed as the IDs for the subj, pred, and obj) containing that original entity.

    The values it returns are:
        - X_p (Tensor): A matrix of feature vectors for all triples described in triples_map. It also includes, in column 0, the ID of the triple for which the vector was generated. This is removed before the vector is passed through TWIG-I's NN (it must be to avoid overfitting) but is needed so the negative sampler knows that triple to generate negatives for.
    '''
    metadata = graph_stats['train'] #always use data from training set to annotate to avoid test leakage
    struct_loader  = Structure_Loader(
        triples_map=triples_map,
        metadata=metadata,
        ents_to_triples=ents_to_triples,
        fts_blacklist=fts_blacklist
    )
    
    all_data_pos = []
    for triple_idx in triples_map:
        s, p, o = triples_map[triple_idx]
        data_pos = struct_loader(s, p, o)
        all_data_pos.append([triple_idx] + data_pos)
    X_p = torch.tensor(all_data_pos, dtype=torch.float32)
    return X_p

def get_norm_func(
        X,
        normalisation,
    ):
    '''
    get_norm_func() returns a function that can be called on any feature vector to normalise it. This is particularly useful for the negative sampler, as this function can be given to it to use for all negatives so they are normalised just like the positives.

    The arguments it accepts are:
        - X (torch.Tensor): the Tesnor containing all feature vectors for all triples **in the train set only** -- this tensor is used as a bsis for normalisation (for example, in minmax normalisation, its min and its max are used for the normalisation procesdure).
        - normalisation (str): The normaliation methods that should be encoded in the returned normalisation function. Must be one of  ('minmax', 'zscore', 'none'). 

    The values it returns are:
        - norm_func (func): a function that accepts a Tensor (base_data) and a bool (col_0_removed) as input and that outputs a row-by-row normalised version of the input tensor. col_0_removed is used internally to make sure that colunn 0 (if present) is not modified as it represents the triple index and at various points is present or removed.
    '''
    assert normalisation in ('minmax', 'zscore', 'none')
    
    if normalisation == 'none':
        def norm_func(base_data):
            return base_data
        return norm_func
    
    elif normalisation == 'minmax':
        running_min = torch.min(X, dim=0).values
        running_max = torch.max(X, dim=0).values

        def norm_func(base_data, col_0_removed):
            return minmax_norm_func(
                X=base_data,
                train_min=running_min,
                train_max=running_max,
                col_0_already_removed=col_0_removed
            )
        
        return norm_func

    elif normalisation == 'zscore':
        # running average has been verified to be coreect
        num_samples = 0.
        num_samples += X.shape[0]
        running_avg = torch.sum(X, dim=0) / num_samples

        # running std has been verified to be coreect
        running_std = torch.sum(
            (X - running_avg) ** 2,
            dim=0
        )
        running_std = torch.sqrt(
            (1 / (num_samples - 1)) * running_std
        )

        def norm_func(base_data, col_0_removed):
            return zscore_norm_func(
                X=base_data,
                train_mean=running_avg,
                train_std=running_std,
                col_0_already_removed=col_0_removed
            )
        
        return norm_func

def minmax_norm_func(X, train_min, train_max, col_0_already_removed):
    '''
    minmax_norm_func() performs min-max normalisation on the given data

    The arguments it accepts are:
        - X (torch.Tensor): A tensor containing the data to normalise
        - train_min (torch.Tensor): A single-row tensor of the minimum values observed for each feature
        - train_max (torch.Tensor): A single-row tensor of the maximum values observed for each feature
        - col_0_already_removed (bool): whether column 0 has been removed from X before passing it to this function or not

    The values it returns are:
        - X_norm (torch.Tensor): A tensor containing the normalised data

    NOTE:  due to how this function is created, train_min and train_max will contain "column 0" (having triple indicies) and thus are not used in normalisation
    '''
    if not col_0_already_removed:
        X_graph, X_other = X[:, :1], X[:, 1:] # ignore col 0; that is the triple index, and we needs its original value!
        X_other = (X_other - train_min[1:]) / (train_max[1:] - train_min[1:])
        X_norm = torch.concat(
            [X_graph, X_other],
            dim=1
        )
    else:
        X_norm = (X - train_min[1:]) / (train_max[1:] - train_min[1:])

    # if we had nans (i.e. min = max) set them all to 0.5
    X_norm = torch.nan_to_num(X_norm, nan=0.5, posinf=0.5, neginf=0.5) 
    return X_norm

def zscore_norm_func(X, train_mean, train_std, col_0_already_removed):
    '''
    zscore_norm_func() performs z-score normalisation on the given data

    The arguments it accepts are:
        - X (torch.Tensor): A tensor containing the data to normalise
        - train_mean (torch.Tensor): A single-row tensor of the mean values observed for each feature
        - train_std (torch.Tensor): A single-row tensor of the standard eviation of the values observed for each feature
        - col_0_already_removed (bool): whether column 0 has been removed from X before passing it to this function or not

    The values it returns are:
        - X_norm (torch.Tensor): A tensor containing the normalised data

    NOTE:  due to how this function is created, train_mean and train_std will contain "column 0" (having triple indicies) and thus are not used in normalisation
    '''
    if not col_0_already_removed:
        X_graph, X_other = X[:, :1], X[:, 1:] # ignore col 0; that is the triple index, and we needs its original value!
        X_other = (X_other - train_mean[1:]) / train_std[1:]
        X_norm = torch.concat(
            [X_graph, X_other],
            dim=1
        )
    else:
        X_norm = (X - train_mean[1:]) / train_std[1:]

    # if we had nans (i.e. std = 0) set them all to 0
    X_norm = torch.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0) 
    return X_norm 

def calc_split_data(
        triples_dicts,
        graph_stats,
        purpose,
        batch_size,
        normalisation,
        norm_func,
        fts_blacklist
    ):
    '''
    calc_split_data() creates a full dataloader for a given data split (i.e. train, test, valid).
    
    The arguments it accepts are:
        - triples_dicts (dict of str -> list<tuple<int, int int>>): a dict that maps the triple split (train, test, valid, or all) to all triples contained in that split. The "all" split contains all triples in the KG. Note that all nodes and edges are represented by their numeric (integer) IDs, not their labels, in the returned lists. In each row of this list, the integers represent, in order, the IDs of the subject, predicate, and object of a triple.
        - graph_stats (dict of a lot of things): dict with the format:
              all / train / test / valid : 
              {
                  'degrees': dict int (node ID) -> float (degree)
                  'pred_freqs': dict int (edge ID) -> float (frequency count)
                  'subj / obj / total _relationship_degrees': dict tuple <int, int> (pair of <subj/obj, predicate> IDs)-> float (co-occurrence count) 
                  'percentiles': dict int (percentile) -> float (degree at that percentile)
                  'subj / obj / total _rel_degree_percentiles': dict int (percentile) -> float (percentile of relationship_degrees as above)
              } 
        - purpose (str): "train", "valid", or "test" -- the phase of training / evaluation for which data is being collected.
        - batch_size (int): the batch size to use for this dataloader.
        - normalisation (str): A string representing the methos that should be used to normalise all data. Currently, "zscore" and "minmax" are implemented; "none" can be given to not use normalisation. Will only be used if norm_func is None.
        - norm_func (func): a function that accepts a Tensor (base_data) and a bool (col_0_removed) as input and that outputs a row-by-row normalised version of the input tensor. If given, it should only ever be the norm func derrived from the train set.
        - fts_blacklist (list of str): A  list of feature names that should NOT be calculated for negatives.

    The values it returns are:
        - dataloader_pos (torch.utils.data.DataLoader): A dataloader configured to load the input data with the requested batch size.
        - norm_func (func): a function that accepts a Tensor (base_data) and a bool (col_0_removed) as input and that outputs a row-by-row normalised version of the input tensor. The orm func returned was created using only the training set to avoid data leakage.
        - X_p (torch.Tensor): the tensor containing all data in the dataloader as a single object. This is needed for use with the VectorNegativeSampler to build a triple ID -> triple feature vector map. If thta negative sampler is not being used, it can be (and will be) ignored.
        - n_local (int): the total number of features calculated in the feature vector of all triples / negatives. The "local" in "n_local" comes from the fact that all of these are local-structure features (however, global feartures are not used and therefore calling this "local" is technically redundant)
    '''
    triples_map = get_triples_by_idx(
        triples_dicts=triples_dicts,
        triples_set=purpose
    )
    ents_to_triples = get_adj_data(triples_map=triples_map)
    X_p = get_twm_data_augment(
        triples_map=triples_map,
        graph_stats=graph_stats,
        ents_to_triples=ents_to_triples,
        fts_blacklist=fts_blacklist
    )
    X_p = X_p.to(device)
    n_local = int(X_p.shape[1]) - 1 # subtract 1 since col 0 (with triple indivies) is still included at this point

    print('X_p:', X_p.shape)
    print('n_local:', n_local)

    if norm_func is None:
        assert purpose == 'train', f'norm_func can only be non for training data, but the given purpse is {purpose}'
        norm_func = get_norm_func(
            X=X_p,
            normalisation=normalisation,
        )
    X_p = norm_func(X_p, col_0_removed=False)

    torch_dataset_pos = TensorDataset(X_p)
    dataloader_pos = DataLoader(
        torch_dataset_pos,
        batch_size=batch_size
    )

    return dataloader_pos, norm_func, X_p, n_local

def do_load(
        all_datasets,
        normalisation,
        batch_size,
        batch_size_test,
        fts_blacklist=None
    ):
    '''
    do_load() is the port-of-call function for loading data. It orchestrates all helpers to load data from disk (or from a saved intermediary file with all the pre-processing already done) and returns ready-to-use Dataloaders.
    
    The arguments it accepts are:
        - all_datasets (list of str): a list of the names of the datasets that should be loaded.
        - normalisation (str): the normalisation method to be used when loading data (and when created vectorised forms of negatively sampled triples). "zscore", "minmax", and "none" are currently supported.
        - batch_size (int): the batch size to use while training.
        - batch_size_test (int): the batch size to use during testing and validation.
        - fts_blacklist (list of str): A  list of feature names that should NOT be calculated for negatives.

    The values it returns are:
        - dataloaders (dict str -> str -> torch.utils.data.DataLoader): a dict that maps a training split ("train", "test", or "valid") and a dataset (with a name as in dataset_names) to a DataLoader that can be used to load batches for that dataset on that training split. An example of accessing its data could be dataloaders["train"]["UMLS"], which would return the training dataloader for the UMLS dataset.
        - norm_funcs (dict of str -> func): Maps the sname of each dataset to the normalisation functions created from the train set of the dataset
        - n_local (int): the total number of features calculated in the feature vector of all triples / negatives. The "local" in "n_local" comes from the fact that all of these are local-structure features (however, global feartures are not used and therefore calling this "local" is technically redundant). n_local will always be the same for all datasets the way the code is written now, so only one value is returned.
    '''
    dataloaders = {
        'train': dict(),
        'test': dict(),
        'valid': dict()
    }

    norm_funcs = {}
    n_local = 0
    for dataset_name in all_datasets:
        print(dataset_name)
        try:
            pykeen_dataset = pykeendatasets.get_dataset(dataset=dataset_name)
        except:
            pykeen_dataset = load_custom_dataset(dataset_name)  
        
        triples_dicts = get_triples(pykeen_dataset)
        graph_stats = calc_graph_stats(triples_dicts, do_print=False)

        train_X_pos, norm_func, X_pos, n_local = calc_split_data(
            triples_dicts=triples_dicts,
            graph_stats=graph_stats,
            purpose='train',
            batch_size=batch_size,
            normalisation=normalisation,
            norm_func=None,
            fts_blacklist=fts_blacklist
        )
        test_X_pos, _, _, _ = calc_split_data(
            triples_dicts=triples_dicts,
            graph_stats=graph_stats,
            purpose='test',
            batch_size=batch_size_test,
            normalisation=normalisation,
            norm_func=norm_func,
            fts_blacklist=fts_blacklist
        )
        valid_X_pos, _, _, _ = calc_split_data(
            triples_dicts=triples_dicts,
            graph_stats=graph_stats,
            purpose='valid',
            batch_size=batch_size_test,
            normalisation=normalisation,
            norm_func=norm_func,
            fts_blacklist=fts_blacklist
        )

        dataloaders['train'][dataset_name] = train_X_pos
        dataloaders['test'][dataset_name] = test_X_pos
        dataloaders['valid'][dataset_name] = valid_X_pos
        norm_funcs[dataset_name] = norm_func

    return dataloaders, norm_funcs, X_pos, n_local
