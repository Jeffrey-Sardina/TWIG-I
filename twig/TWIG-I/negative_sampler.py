from load_data import Structure_Loader
import random
import torch
import time
from frozendict import frozendict

'''
====================
Constant Definitions
====================
'''
device = "cuda"

class Negative_Sampler:   
    def get_negatives(self, purpose, triple_index, npp, **kwargs):
        raise NotImplementedError
    
    def get_batch_negatives(self, purpose, triple_idxs, npp, side):
        raise NotImplementedError


class Vector_Negative_Sampler(Negative_Sampler):
    '''
    NOTE: Not tested completely, not fully implemented
    If you really do want to use this, please be aware that it may contain arbitrary errors until further testing of it is completed.
    '''
    def __init__(
            self,
            X_pos,
            n_bins,
            mode,
            dataset_name,
            simple_negative_sampler
        ):
        self.mode = mode # near-negative, invert-positive, near-positive
        self.dataset_name = dataset_name
        self.simple_negative_sampler = simple_negative_sampler #for use in testing, as this cannot be used there

        # check that a valid mode was given
        assert mode in ("near-negative", "invert-positive", "near-positive", "unif-random"), f'Invalid mode given: {self.mode}'

        # generate map of triple index to fts (positive triples)
        triple_idx_to_ft_vec = {}
        for row in X_pos:
            triple_idx = int(row[0])
            fts = row[1:]
            triple_idx_to_ft_vec[triple_idx] = fts
        all_fts = X_pos[:, 1:]
        self.num_fts = all_fts.shape[1]

        # get global fts details
        self.global_max = torch.max(all_fts)
        self.global_min = torch.min(all_fts)
        self.global_range = self.global_max - self.global_min

        # generate map of features to their distributions
        num_cols = all_fts.shape[1]
        ft_idx_to_hist = [0 for _ in range(num_cols)]
        for col in range(num_cols):
            col_vals = all_fts[:, col]
            min_val = torch.min(col_vals)
            max_val = torch.max(col_vals)
            ft_hist = torch.histc(
                col_vals,
                bins=n_bins,
                min=min_val,
                max=max_val
            ).to(device)
            # the below is verified
            bin_edges = torch.linspace(min_val, max_val, steps=n_bins+1).to(device)
            ft_idx_to_hist[col] = ft_hist, bin_edges
        self.ft_idx_to_hist = ft_idx_to_hist
        self.triple_idx_to_ft_vec = frozendict(triple_idx_to_ft_vec)

        # generate a histogram of negative ft dists, if needed
        if self.mode == "near-negative":
            sample_npp = 60 #for 30 per side total (Central Limit Theorem go brrr)
            triple_idxs = triple_idx_to_ft_vec.keys()
            all_negs, _ = self.simple_negative_sampler.get_batch_negatives(
                purpose='train',
                triple_idxs=triple_idxs,
                npp=sample_npp,
                side='both'
            )
            num_cols = all_fts.shape[1]
            ft_idx_to_hist_neg = [0 for _ in range(num_cols)]
            for col in range(num_cols):
                col_vals = all_negs[:, col]
                min_val = torch.min(col_vals)
                max_val = torch.max(col_vals)
                ft_hist = torch.histc(
                    col_vals,
                    bins=n_bins,
                    min=min_val,
                    max=max_val
                ).to(device)
                # the below is verified
                bin_edges = torch.linspace(min_val, max_val, steps=n_bins+1).to(device)
                ft_idx_to_hist_neg[col] = ft_hist, bin_edges
            self.ft_idx_to_hist_neg = ft_idx_to_hist_neg

    def get_batch_negatives(self, purpose, triple_idxs, npp, side):
        if purpose != 'train':
            return self.simple_negative_sampler.get_batch_negatives(purpose, triple_idxs, npp, side)
        
        # get true triple embeddings
        pos_vecs = []
        for triple_idx in triple_idxs:
            for _ in range(npp):
                pos_vecs.append(self.triple_idx_to_ft_vec[int(triple_idx)])
        pos_vecs = torch.stack(pos_vecs, dim=0)
        
        if self.mode != "unif-random":
            # generate base negatives
            base_negs = []
            for ft_idx in range(self.num_fts):
                ft_vals = self.sample_ft_vals(ft_idx, npp * triple_idxs.shape[0])
                base_negs.append(ft_vals)
            base_negs = torch.stack(base_negs).to(device).T
        else:
            base_negs = torch.rand(pos_vecs.shape, device=device) * self.global_range + self.global_min

        # get a mask to merge negs and the triple
        del_prob = 0.5
        pre_mask = torch.rand(pos_vecs.shape, device=device)
        pos_mask = pre_mask <= del_prob
        neg_mask = pre_mask > del_prob

        # apply mask and output
        all_negs = (pos_vecs * pos_mask) + (base_negs * neg_mask)
        npps = [npp] * len(triple_idxs)
        npps = torch.tensor(npps, device=device)
        return all_negs, npps

    def unif_rand_val_in_bins(self, sampled_bins, bin_edges, bin_width, npp):
        sample = torch.rand(npp, device=device) * bin_width + torch.index_select(bin_edges, dim=0, index=sampled_bins) # bin_min = bin_edges[sampled_bin]
        return sample

    def sample_ft_vals(self, ft_idx, npp):
        if self.mode == "invert-positive":
            # weights need not sum to 1, but must be in the right proportions
            histogram, bin_edges = self.ft_idx_to_hist[ft_idx]
            bin_width = bin_edges[1] - bin_edges[0]
        elif self.mode == "near-negative":
            histogram, bin_edges = self.ft_idx_to_hist_neg[ft_idx]
            bin_width = bin_edges[1] - bin_edges[0]
        elif self.mode == "near-positive":
            histogram, bin_edges = self.ft_idx_to_hist[ft_idx]
            bin_width = bin_edges[1] - bin_edges[0]
        else:
            assert False, f'invalid mode given: {self.mode}. Note that mode "unif-random" should not be used with sample_ft_vals() -- it is handled directly in get_batch_negatives()'

        sampled_bins = torch.multinomial(
            histogram,
            npp,
            replacement=True
        )
        samples = self.unif_rand_val_in_bins(
            sampled_bins=sampled_bins,
            bin_edges=bin_edges,
            bin_width=bin_width,
            npp=npp
        )
        return samples


class Optimised_Negative_Sampler(Negative_Sampler):
    def __init__(
            self,
            filters,
            graph_stats,
            triples_map,
            ents_to_triples,
            norm_func,
            dataset_name,
            fts_blacklist
        ):
        '''
        init() initialises the negative sampler with all data it will need to generate negatives -- including pre-calculation of all negative triple feature vectors so that they can be accessed rapidly during training.

        The arguments it accepts are:
            - filters (dict str -> str -> dict): a dict that maps a dataset name (str) and a training split name (i.e. "train", "test", or "valid") to a dictionary describing the triples to use i filtering. To be exact, this second triples_dict has the structure (dict str -> int -> tuple<int,int,int>). It maps first the training split name to the triple index, and that trtrriple index maps to a single triple expressed as (s, p, o) with integral representations
              of each triple element.
            - graph_stats (dict of a lot of things): dict with the format:
                all / train / test / valid : 
                {
                    'degrees': dict int (node ID) -> float (degree)
                    'pred_freqs': dict int (edge ID) -> float (frequency count)
                    'subj / obj / total _relationship_degrees': dict tuple <int, int> (pair of <subj/obj, predicate> IDs)-> float (co-occurrence count) 
                    'percentiles': dict int (percentile) -> float (degree at that percentile)
                    'subj / obj / total _rel_degree_percentiles': dict int (percentile) -> float (percentile of relationship_degrees as above)
                } 
            - triples_map (dict int -> tuple (int, int int)): a dict that maps a triple index to (s,p,o) integer IDs for nodes and edges. 
            - ents_to_triples (dict int -> int): a dict that maps an entity ID to a list of all triples IDs (expressed as containing that original entity.
            - dataset_name (str): the name of the dataset that should be used to save a cache of all precalculated features of avoid the need for redundant compuation each time TWIG is run.
            - fts_blacklist (list of str): A  list of feature names that should NOT be calculated for negatives.

        The values it returns are:
            - None
        '''
        self.filters = filters
        self.metadata = frozendict(graph_stats['train'])
        self.triples_map = frozendict(triples_map)
        self.norm_func = norm_func
        self.struct_loader = Structure_Loader(
            triples_map=self.triples_map,
            metadata=self.metadata,
            ents_to_triples=ents_to_triples,
            fts_blacklist=fts_blacklist
        )
        self.dataset_name = dataset_name
        self.all_ents = list(ents_to_triples.keys())
        self.fts_blacklist = frozenset(fts_blacklist)

        print('init negative sampler with args')
        print('\tfts_blacklist:', fts_blacklist)

        if len(self.filters['train']) > 0:
            assert False, "Filterng in training not currently supported"

    def get_batch_negatives(self, purpose, triple_idxs, npp, side):
        '''
        get_batch_negatives() generates a batch of negatives for a given set of triples.
        
        The arguments it accepts are:
            - purpose (str): "train", "valid", or "test" -- the phase of training / evaluation for which these negatives are being generated. This is used to determine what filters to use.
            - triple_idxs (Tensor of int): a tensor containing the triple indicies of all triples for which negatives are wanted.
            - npp (int): the number of negative samples to generate per positive triple during training. If the current purpose if not training, it MUST be -1 to generate all triples and avoid bias.
            - side (str): the side to corrupt. Options are 's', 'o', and 'both'

        The values it returns are:
            - all_negs (Tensor): a tensor containing all negatives that are generated, in blocks in the same order as the order of the triple indicies that are given. 
            - npps (Tensor): a tensor containing the number of negatives per per positive that were used in the negative generation. All values in npps will be equal to the input npp unless upsampling is disabled in the negative sampler. 
        '''
        assert side in ('s', 'o', 'both'), f"side must be one of {('s', 'o', 'both')}"

        all_negs = []
        npps = []
        for idx in triple_idxs:
            negs, npp_returned = self.get_negatives(
                purpose=purpose,
                triple_index=int(idx),
                npp=npp,
                side=side
            )
            npps.append(npp_returned)
            all_negs.extend(negs)

        all_negs = torch.tensor(all_negs, dtype=torch.float32, device=device)
        all_negs = self.norm_func(all_negs, col_0_removed=True)
        npps = torch.tensor(npps, device=device)

        return all_negs, npps

    def get_negatives(self, purpose, triple_index, npp, side):
        '''
        get_negatives() generates negates for a given triple. All sampling is done with replacement.

        The arguments it accepts are:
            - purpose (str): "train", "valid", or "test" -- the phase of training / evaluation for which these negatives are being generated. This is used to determine what filters to use.
            - triple_index (int): the triple index of the triple for which negatives are wanted.
            - npp (int): the number of negative samples to generate per positive triple during training. If the current purpose if not training, it MUST be -1 to generate all triples and avoid bias.
            - side (str): the side to corrupt. Options are 's', 'o', and 'both'

        The values it returns are:
            - negs (Tensor): a tensor containing all negatives that are generated for the given triple.
            - npp_returned (int): the number of negatives actually returned. This differs from npp only in the case that there are most negatives requested than can be generated (such as due to filters) and when upsampling is turned off. 
        '''
        # check input for validity
        assert side in ('s', 'o', 'both'), f"side must be one of {('s', 'o', 'both')}"
        if purpose == 'test' or purpose == 'valid':
            assert npp == -1, "npp = -1 should nbe used always in testing and validation"
            assert not side == 'both', 'In testing / validation you should corrupt both sides individually!'
        if purpose == 'test' or purpose == 'valid':
            assert npp == -1, "npp = -1 should nbe used always in testing and validation"
            assert not side == 'both', 'In testing / validation you should corrupt both sides individually!'
            gen_all_negs = True
        else:
            gen_all_negs = False

        # basic defs to start (note we will trim down s_corrs and o_corrs if we
        # are sampling and not generating all negatives)
        s, p, o = self.triples_map[triple_index]
        s_corrs = self.all_ents
        o_corrs = self.all_ents

        # trim so we only get npp negs (and have random ones)
        if not gen_all_negs:
            # generate random corruptions (this is the sampling step)
            if side == 'both':
                npp_s = npp // 2
                npp_o = npp // 2
                if npp % 2 != 0:
                    add_extra_to_s = random.random() > 0.5
                    if add_extra_to_s:
                        npp_s += 1
                    else:
                        npp_o += 1
                s_corrs = random.choices(
                    s_corrs,
                    k=npp_s
                )
                o_corrs = random.choices(
                    o_corrs,
                    k=npp_o
                )
            elif side == 's':
                npp_s = npp
                npp_o = 0
                s_corrs = random.choices(
                    s_corrs,
                    k=npp_s
                )
                o_corrs = []
            elif side == 'o':
                npp_s = 0
                npp_o = npp
                s_corrs = []
                o_corrs = random.choices(
                    o_corrs,
                    k=npp_o
                )

        # construct negative triples and filter if needed
        negs = []
        if side == 's' or side == 'both':
            prefab = self.struct_loader.create_negative_prefab_scorr(p, o)
            for s_corr in s_corrs:
                if not gen_all_negs:
                    # vec_old = self.struct_loader(s_corr, p, o)
                    # vec = self.struct_loader.vec_from_prefab_scorr(prefab, s_corr, p, o)
                    # assert vec == vec_old, (vec_old, vec)
                    negs.append(
                        self.struct_loader.vec_from_prefab_scorr(prefab, s_corr, p, o)
                    )
                elif not (s_corr, p, o) in self.filters[purpose]:
                    negs.append(
                        self.struct_loader.vec_from_prefab_scorr(prefab, s_corr, p, o)
                    )
        if side == 'o' or side == 'both':
            prefab = self.struct_loader.create_negative_prefab_ocorr(s, p)
            for o_corr in o_corrs:
                if not gen_all_negs:
                    # vec_old = self.struct_loader(s, p, o_corr)
                    # vec = self.struct_loader.vec_from_prefab_ocorr(prefab, s, p, o_corr)
                    # assert vec == vec_old, (vec_old, vec)
                    negs.append(
                        self.struct_loader.vec_from_prefab_ocorr(prefab, s, p, o_corr)
                    )
                elif not (s, p, o_corr) in self.filters[purpose]:
                    negs.append(
                        self.struct_loader(s, p, o_corr)
                    )

        # common-sense validation before we return everything!
        npp_returned = len(negs) #len(s_corrs) + len(o_corrs)
        if npp != -1:
            assert npp == npp_returned, f'{npp} =/= {npp_returned}'

        return negs, npp_returned
