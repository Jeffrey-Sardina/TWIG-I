# external imports
from pykeen import datasets
import numpy as np
from pykeen.triples import TriplesFactory

class Custom_Dataset():
    '''
    For loading a custom dataset with the same API as a PyKEEN dataset
    '''
    def __init__(self, factory_dict):
        self.factory_dict = factory_dict

'''
================
Helper Functions
================
'''
def get_triples(dataset):
    '''
    get_triples() gets all triples in the given dataset, both in total and by the train / test / validation split.

    The arguments it acepts are:
        - dataset (pykeen.datasets.Dataset or Custom_Dataset): the dataset whose triples are wanted

    The values returned are:
        - triples_dicts (dict of str -> list<tuple<int, int int>>): a dict that maps the triple split (train, test, valid, or all) to all triples contained in that split. The "all" split contains all triples in the KG. Note that all nodes and edges are represented by their numeric (integer) IDs, not their labels, in the returned lists. In each row of this list, the integers represent, in order, the IDs of the subject, predicate, and object of a triple.
    '''
    train_triples = []
    test_triples = []
    valid_triples = []

    for s, p, o in dataset.factory_dict['training'].mapped_triples:
        s, p, o = int(s), int(p), int(o)
        train_triples.append((s, p, o))
    for s, p, o in dataset.factory_dict['testing'].mapped_triples:
        s, p, o = int(s), int(p), int(o)
        test_triples.append((s, p, o))
    for s, p, o in dataset.factory_dict['validation'].mapped_triples:
        s, p, o = int(s), int(p), int(o)
        valid_triples.append((s, p, o))
    all_triples = train_triples + test_triples + valid_triples

    triples_dicts = {
        'all': all_triples,
        'train': train_triples,
        'test': test_triples,
        'valid': valid_triples
    }
    return triples_dicts

def get_count_data(triples):
    '''
    get_count_data() calculates the degree of all nodes and the frequency of all predicates in the given set of triples.

    The arguments it acepts are:
        - triples (list<tuple<int, int int>>): a list of all triples (using numeric IDs for their elements). In each row of this list, the integers represent, in order, the IDs of the subject, predicate, and object of a triple.

    The values returned are:
        - sorted_degrees (dict of int -> int): a dict, sorted by the numeric value of the key (node ID) mapping every node ID to the degree of that node among the given triples
        - sorted_freqs (dict of int -> int): a dict, sorted by the numeric value of the key (predicate ID) mapping every predicate ID to the frequency of that node among the given triples

    NOTE: dicts are sorted by key as a standardisation and reproducibility measure.
    '''
    degrees = {}
    pred_freqs = {}
    for s, p, o in triples:
        if not s in degrees:
            degrees[s] = 0
        if not o in degrees:
            degrees[o] = 0
        if not p in pred_freqs:
            pred_freqs[p] = 0
        degrees[s] += 1
        degrees[o] += 1
        pred_freqs[p] += 1
    sorted_degrees, sorted_freqs = dict(sorted(degrees.items())), dict(sorted(pred_freqs.items()))
    return sorted_degrees, sorted_freqs

def get_relationship_degrees(triples):
    '''
    get_relationship_degrees() calculates the co-frequencies of every node-predicate pair (i.e. subject-predicate and object-predicate), as well as the combination of them both (every node-predicate frequency, regardless of whether the node is a subject or obejct)

    The arguments it acepts are:
        - triples (list<tuple<int, int int>>): a list of all triples (using numeric IDs for their elements). In each row of this list, the integers represent, in order, the IDs of the subject, predicate, and object of a triple.

    The values returned are:
        - sorted_subj_relationship_degrees (dict of tuple<int,int> -> int): A dict that maps a subject ID and a predicate ID to the number of times that (s, p, *) is observed in the given triples.
        - sorted_obj_relationship_degrees (dict of tuple<int,int> -> int): A dict that maps an object ID and a predicate ID to the number of times that (*, p, o) is observed in the given triples.
        - sorted_total_relationship_degrees (dict of tuple<int,int> -> int): A dict that maps a node ID and a predicate ID to the number of times that (node, p, *) or (*, p, node) is observed in the given triples.

    NOTE: dicts are sorted by key as a standardisation and reproducibility measure.
    '''
    subj_relationship_degrees = {}
    obj_relationship_degrees = {}
    total_relationship_degrees = {}
    for s, p, o in triples:
        if not (s, p) in subj_relationship_degrees:
            subj_relationship_degrees[(s, p)] = 0
        if not (o, p) in obj_relationship_degrees:
            obj_relationship_degrees[(o, p)] = 0
        if not (s, p) in total_relationship_degrees:
            total_relationship_degrees[(s, p)] = 0
        if not (o, p) in total_relationship_degrees:
            total_relationship_degrees[(o, p)] = 0

        subj_relationship_degrees[(s, p)] += 1
        obj_relationship_degrees[(o, p)] += 1
        total_relationship_degrees[(s, p)] += 1
        total_relationship_degrees[(o, p)] += 1

    sorted_subj_relationship_degrees, \
        sorted_obj_relationship_degrees, \
        sorted_total_relationship_degrees = dict(sorted(subj_relationship_degrees.items())), \
            dict(sorted(obj_relationship_degrees.items())), \
            dict(sorted(total_relationship_degrees.items()))

    return sorted_subj_relationship_degrees, \
        sorted_obj_relationship_degrees, \
        sorted_total_relationship_degrees

def get_subj_obj_cofreqs(triples):
    '''
    get_subj_obj_cofreqs() calculates the co-frequencies of every node-node pair in the given triples, irrespective of the predicate that connects them.

    The arguments it acepts are:
        - triples (list<tuple<int, int int>>): a list of all triples (using numeric IDs for their elements). In each row of this list, the integers represent, in order, the IDs of the subject, predicate, and object of a triple.

    The values returned are:
        - sorted_subj_obj_cofreqs (dict of tuple<int,int> -> int): A dict that maps a subject ID and an object ID to the number of times that (s, *, o) is observed in the given triples.

    NOTE: dicts are sorted by key as a standardisation and reproducibility measure.
    '''
    subj_obj_cofreqs = {}
    for s, _, o in triples:
        if not (s, o) in subj_obj_cofreqs:
            subj_obj_cofreqs[(s, o)] = 0
        subj_obj_cofreqs[(s, o)] += 1
    sorted_subj_obj_cofreqs = dict(sorted(subj_obj_cofreqs.items()))
    return sorted_subj_obj_cofreqs

def get_percentiles(counts_data):
    '''
    get_percentiles() calculates the percentiles of the gives counts data

    The arguments it acepts are:
        - counts_data_list (dict of any -> int): a dict mapping a key (such as a node or predicate ID) to counts data for that ID (i.e. the node degree for predicate frequency)

    The values returned are:
        - percentiles (dict of int -> float): A dict that maps a a precentile level (int) to the percentile value calculated from among all the input counts data in `degrees`
    '''
    percentiles = {}
    counts_data_list = [counts_data[key] for key in counts_data]
    percentiles_wanted = [0, 1, 5, 10, 20, 25, 30, 33, 40, 50, 60, 67, 76, 75, 80, 90, 95, 99, 100]
    for percentile in percentiles_wanted:
        percentiles[percentile] = np.percentile(counts_data_list, percentile)
    return percentiles

def calc_triples_stats(triples):
    '''
    calc_triples_stats() calculates a large number of general stats for the given triples set

    The arguments it acepts are:
        - triples (list<tuple<int, int int>>): a list of all triples (using numeric IDs for their elements). In each row of this list, the integers represent, in order, the IDs of the subject, predicate, and object of a triple.

    The values returned are:
        - degrees: obtained from get_count_data() for these triples
        - pred_freqs: obtained from get_count_data() for these triples 
        - subj_relationship_degrees: obtained from get_relationship_degrees() for these triples
        - obj_relationship_degrees: obtained from get_relationship_degrees() for these triples
        - total_relationship_degrees: obtained from get_relationship_degrees() for these triples
        - subj_obj_cofreqs: obtained from get_subj_obj_cofreqs() for these triples
        - percentiles: percentiles obtained from get_percentiles() for all degrees
        - pred_percentiles: percentiles obtained from get_percentiles() for all pred_freqs
        - subj_rel_degree_percentiles: percentiles obtained from get_percentiles() for all subj_relationship_degrees
        - obj_rel_degree_percentiles: percentiles obtained from get_percentiles() for all obj_relationship_degrees
        - total_rel_degree_percentiles: percentiles obtained from get_percentiles() for all total_relationship_degrees
        - subj_obj_cofreqs_percentiles: percentiles obtained from get_percentiles() for all subj_obj_cofreqs
    '''
    degrees, pred_freqs = get_count_data(triples)
    subj_relationship_degrees, \
        obj_relationship_degrees, \
        total_relationship_degrees = get_relationship_degrees(triples)
    subj_obj_cofreqs = get_subj_obj_cofreqs(triples)
    percentiles = get_percentiles(degrees)
    pred_percentiles = get_percentiles(pred_freqs)
    subj_rel_degree_percentiles = get_percentiles(subj_relationship_degrees)
    obj_rel_degree_percentiles = get_percentiles(obj_relationship_degrees)
    total_rel_degree_percentiles = get_percentiles(total_relationship_degrees)
    subj_obj_cofreqs_percentiles = get_percentiles(subj_obj_cofreqs)
    return degrees, \
                pred_freqs, \
                subj_relationship_degrees, \
                obj_relationship_degrees, \
                total_relationship_degrees, \
                subj_obj_cofreqs, \
                percentiles, \
                pred_percentiles, \
                subj_rel_degree_percentiles, \
                obj_rel_degree_percentiles, \
                total_rel_degree_percentiles, \
                subj_obj_cofreqs_percentiles

def write_stats_data(triples_set_name, triples_data_dict):
    '''
    write_stats_data() is a really long function that is basically just a bunch of (kin-of pretty) print statements. IT just takes all the data given and prints it (to std.out)

    The arguments it acepts are:
        - triples_set_name (str) the name of the triples spllit (i.e. train, test, valid, or all)
        - triples_data_dict (dict of <a lot og things>): a dict that contains all data that is to be printed

    The values it returns are:
        - None

    NOTE: If it was not obvious this is a sub-routine called by other functions in this code. Don't call it to print your own dictionaries!
    '''
    degrees = triples_data_dict['degrees']
    pred_freqs = triples_data_dict['pred_freqs']
    subj_relationship_degrees = triples_data_dict['subj_relationship_degrees']
    obj_relationship_degrees = triples_data_dict['obj_relationship_degrees']
    total_relationship_degrees = triples_data_dict['total_relationship_degrees']
    subj_obj_cofreqs = triples_data_dict['subj_obj_cofreqs']
    percentiles = triples_data_dict['percentiles']
    pred_percentiles = triples_data_dict['pred_percentiles']
    subj_rel_degree_percentiles = triples_data_dict['subj_rel_degree_percentiles']
    obj_rel_degree_percentiles = triples_data_dict['obj_rel_degree_percentiles']
    total_rel_degree_percentiles = triples_data_dict['total_rel_degree_percentiles']
    subj_obj_cofreqs_percentiles = triples_data_dict['subj_obj_cofreqs_percentiles']

    print(f'{"="*10} Stats for the {triples_set_name} triples set {"="*10}')
    print(f'{"="*5} Degree percentiles {"="*5}')
    for percentile in percentiles:
        print(f'{percentile}%:\t{percentiles[percentile]}')
    print()

    print(f'{"="*5} Predicate freq percentiles {"="*5}')
    for percentile in pred_percentiles:
        print(f'{percentile}%:\t{pred_percentiles[percentile]}')
    print()

    print(f'{"="*5} (Subj,rel) degree percentiles {"="*5}')
    for percentile in subj_rel_degree_percentiles:
        print(f'{percentile}%:\t{subj_rel_degree_percentiles[percentile]}')
    print()

    print(f'{"="*5} (Obj,rel) degree percentiles {"="*5}')
    for percentile in obj_rel_degree_percentiles:
        print(f'{percentile}%:\t{obj_rel_degree_percentiles[percentile]}')
    print()

    print(f'{"="*5} (Node,rel) degree percentiles {"="*5}')
    for percentile in total_rel_degree_percentiles:
        print(f'{percentile}%:\t{total_rel_degree_percentiles[percentile]}')
    print()

    print(f'{"="*5} (Subj,obj) co-frequency percentiles {"="*5}')
    for percentile in subj_obj_cofreqs_percentiles:
        print(f'{percentile}%:\t{subj_obj_cofreqs_percentiles[percentile]}')
    print()

    print(f'{"="*5} Node : degree {"="*5}')
    for node in degrees:
        print(f'{node}:\t{degrees[node]}')
    print()
    
    print(f'{"="*5} Pred : freq {"="*5}')
    for pred in pred_freqs:
        print(f'{pred}:\t{pred_freqs[pred]}')
    print()

    print(f'{"="*5} (Subj,rel) degrees : freq {"="*5}')
    for (node, rel) in subj_relationship_degrees:
        print(f'{(node, rel)}:\t{subj_relationship_degrees[(node, rel)]}')
    print()

    print(f'{"="*5} (Obj,rel) degrees : freq {"="*5}')
    for (node, rel) in obj_relationship_degrees:
        print(f'{(node, rel)}:\t{obj_relationship_degrees[(node, rel)]}')
    print()

    print(f'{"="*5} (Node,rel) degrees : freq {"="*5}')
    for (node, rel) in total_relationship_degrees:
        print(f'{(node, rel)}:\t{total_relationship_degrees[(node, rel)]}')
    print()

    print(f'{"="*5} (Subj,obj) degrees : freq {"="*5}')
    for (subj, obj) in subj_obj_cofreqs:
        print(f'{(subj, obj)}:\t{subj_obj_cofreqs[(subj, obj)]}')
    print()
    print()

def calc_graph_stats(triples_dicts, do_print=True):
    '''
    calc_graph_stats() is a gernal function to calculate and rpint graph stats. It's really just a call to calc_triples_stats and then (optionally) to write_stats_data, and exists to make the API a bit cleaner.

    The arguments it acepts are:
        - triples_dicts (dict of str -> list<tuple<int, int int>>): A dict that maps the triple split (train, test, valid, or all) to all triples contained in that split. The "all" split contains all triples in the KG. Note that all nodes and edges are represented by their numeric (integer) IDs, not their labels, in the returned lists. In each row of this list, the integers represent, in order, the IDs of the subject, predicate, and object of a triple.
        - do_print (bool): whether the graph stats should be printed (and returned) or only returned.

    The values returned are:
        - all_triples_struct_data (dict of str -> str -> any): A dict that maps the triple split (train, test, valid, or all) and a metric / value name to the corresponding value. For example, all_triples_struct_data['train']['degrees'] would access the dictionary containing all nodes and their degrees from the train set of the triples given.

    NOTE: You can get triples_dicts from a call to get_triples() :))
    '''
    all_triples_struct_data = {}
    for name in triples_dicts:
        triples = triples_dicts[name]
        degrees, \
            pred_freqs, \
            subj_relationship_degrees, \
            obj_relationship_degrees, \
            total_relationship_degrees, \
            subj_obj_cofreqs, \
            percentiles, \
            pred_percentiles, \
            subj_rel_degree_percentiles, \
            obj_rel_degree_percentiles, \
            total_rel_degree_percentiles, \
            subj_obj_cofreqs_percentiles = calc_triples_stats(triples)
        
        all_triples_struct_data[name] = {
            'degrees': degrees,
            'pred_freqs': pred_freqs,
            'subj_relationship_degrees': subj_relationship_degrees,
            'obj_relationship_degrees': obj_relationship_degrees,
            'total_relationship_degrees': total_relationship_degrees,
            'subj_obj_cofreqs': subj_obj_cofreqs,
            'percentiles': percentiles,
            'pred_percentiles': pred_percentiles,
            'subj_rel_degree_percentiles': subj_rel_degree_percentiles,
            'obj_rel_degree_percentiles': obj_rel_degree_percentiles,
            'total_rel_degree_percentiles': total_rel_degree_percentiles,
            'subj_obj_cofreqs_percentiles': subj_obj_cofreqs_percentiles,
        }
        if do_print:
            write_stats_data(name, all_triples_struct_data[name])
    return all_triples_struct_data

def get_triples_by_idx(triples_dicts, triples_set):
    '''
    get_triples_by_idx() creates a dictionary from a lsit of triples that maps that triple's index to it. In other words, it makes explicit the use of the triples index as an ID for the entire triple. Note that since Python>3.7 dictioanries are ordered, and since no randomness is ever used in dictionary creation, these indicies and IDs should relaibly take the same values.

    The arguments it acepts are:
        - triples_dicts (dict of str -> list<tuple<int, int int>>): A dict that maps the triple split (train, test, valid, or all) to all triples contained in that split. The "all" split contains all triples in the KG. Note that all nodes and edges are represented by their numeric (integer) IDs, not their labels, in the returned lists. In each row of this list, the integers represent, in order, the IDs of the subject, predicate, and object of a triple.
        - triples_set (str) the split of the triples in triples_dicts (i.e. train, test, valid, or all) for which the index -> triples map is wanted.

    The values returned are:
        - idx_to_triples (dict of int -> tuple<int, int int>): A dict that maps an index to the single triple identified (uniquely) by that index. You can see this dictionary as a pseudo database if you'd ike, subject ot the usual key rules of relational databases. But it is just a regular Python dictionary.

    NOTE: You can get triples_dicts from a call to get_triples() :))
    '''
    idx_to_triples = {}
    for idx, triple in enumerate(triples_dicts[triples_set]):
        idx_to_triples[idx] = triple
    return idx_to_triples

def load_custom_dataset(dataset_name):
    '''
    load_custom_dataset() loads a custom dataset from disk (where by custoom I mean any dataset that is not present by default in PyKEEN; see https://github.com/pykeen/pykeen#datasets). It outputs data in the same format as pykeen.datasets.get_dataset() -- just so that all my code here can interact with datasets using the same API (and a standaard API that others will, hopefuly, understand and appreciate the use of).

    The arguments it accepts are:
        - dataset_name (str): the same of the custom dataset to load. NOTE that this assumes all custom datasets will be located in "./custom_datasets/" -- if they are not, please put them there rather than modifying this code if it all possible, it will make integration much easier on you.


    Please also note that datasets MUST be pre-stratified into train-test-valid splits -- this code will not do that for you, and relies on the files `custom_datasets/{dataset_name}.train`, `custom_datasets/{dataset_name}.test`, and `custom_datasets/{dataset_name}.valid` all existing. If your data is not stratified, PyKEEN offers a very easy way to do that, with documnetation here: https://pykeen.readthedocs.io/en/stable/byo/data.html.

    Finally, please note that all data must be in 3-col TSV format (tab-separated values) to be laoded, as this is what PyKEEN expects and we are loading the data with PyKEEN as a backend.
    '''
    train_path = f'custom_datasets/{dataset_name}.train'
    test_path = f'custom_datasets/{dataset_name}.test'
    valid_path = f'custom_datasets/{dataset_name}.valid'
    factory_dict = {
        'training': TriplesFactory.from_path(train_path),
        'testing': TriplesFactory.from_path(test_path),
        'validation': TriplesFactory.from_path(valid_path)
    }
    dataset = Custom_Dataset(factory_dict)
    return dataset
