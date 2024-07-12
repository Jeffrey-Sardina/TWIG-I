# TWIG-I
Jeffrey Seathrún Sardina and Alok Debnath

## What is TWIG-I?
Topologically-Weighted Intelligence Generation for Inference (TWIG-I) is an embedding-free, graph-structure-based link predictor build to work on knowledge graphs. To put it simply, **TWIG-I models the link prediction task as a regression task**. The input features are hand-crafted graph structure features, and the output value is a plausibility score for each triple.

To be more exact, TWIG-I uses a constant set of handpicked features to describe every triple in a KG, and these features (unlike embeddings in Knowledge Graph embeding systems) are never uptated or changed. Instead, all leaning is done by the neural network that learns to assign a plausibility score to every feature vector. This means that in TWIG-I, all learnable parameters are shared globally. Hence, it's besically a linear regressor predicing plausibility of every triple.

TWIG-I is meant as an alternative to other link predictors such as Knowledge Graph Embeddings and Graph Neural Networks which create latent embedding for every node (and, sometimes, edge) in a knowledge graph. Full details on TWIG-I, its uses, its strengths and weaknesses, and its palce in the literature can be found in the paper (see the citation at the bottom of this README).

Finally, **TWIG-I is completely GenAI free**. No GenAI tools were used to create TWIG-I, its documentation, or to help in the communication of it. Under no circumtances are we willing for TWIG-I to be used as input to training a GenAI algorithm --  we as authors retain copyright on TWIG-I and are unwilling for it, its paper, or its documentation to be used for this purpose.

## Architecture of the Library
TWIG-I itself is a sort of library for learning to solve the LP task without embeddings. It does this using pre-defined structural features (a total of 22) that describe the local degrees and predicate frequencies in and around around a triple.

As much as is possible, all functionality is placed into its own module files. These are
- **load_data.py** -- contains all logic for loading, pre-processing, and normalising data used for training, testing, and validation.
- **loss.py** -- contains implementations of all loss functions
- **negative_sampler.py** -- contains all implementations of negative samplers; i.e. the part of TWIG that creates randomly corrupted triples to uses as negative examples during training.
- **run_exp.py** -- contains all code needed to run TWIG from the command line, including accepting command line arguments, and orchestrating all other modules to make TWIG work.
- **run_from_checkpoint.py** - contains all code needed to re-run TWIG-I from a checkpoint made during training.
- **trainer.py** -- contains implementations for processing batches, generating negatives, and calculating losses during training and testing. It also reports all test statistics and reports on validation data during training.
- **twig_nn.py** -- contains the TWIG neural network implementation.
- **utils.py** -- contains implementations of vavious utility functions, mostly for for loading and processing graph data and putting it into a form TWIG can recognise.

All of these files are fully documented, so if you are curious about how anything works see the function comments in those files. If you don't know where to start, the answer is simple: **run_exp.py**. It calls all o the other files as sub-routines to corrdinate the entire learning process from data loading to writing the results. From there, you can look at any specific module that you want.

## Reproducing the Results of the TWIG-I Paper
To reproduce the results of the TWIG-I paper, use the files
- **../batch-twigi-hyp-search.sh** -- for the hyperparmater search on all datasets
- **.../batch-twigi.sh** -- to run TWIG-I on its optimal hyperparameters
- **../batch-twigi-finetune.sh** -- to run TWIG-I with finetuning. Note that **you will have to edit the checkpoint IDs** to use your own based on the random IDs generated while running TWIG on its optimal hyperparameters.

## Using TWIG-I on New Datasets (Standard for Finetuning)
There are 2 ways to use TWIG-I on a new dataset. If you dataset is given in PyKEEN's default list (https://github.com/pykeen/pykeen#datasets), you can simply pass it by name to TWIG-I and all will work out-of-the-box since TWIG-I uses PyKEEN as a backend data loader.

If you have a custom dataset (such as a KG you made youself), you need to add them into `custom_datasets/`. To be specific, you your new KG is named MyKG, you need to create the following files:
- custom_datasets/MyKG.train
- custom_datasets/MyKG.test
- custom_datasets/MyKG.valid

Each file must be a tab-separated values file where each row represents a triple in the order subject, predicate, object. Once you have done this, you can have TWIG-I use your dataset by passing "MyKG" to it as the datasets parameter, exactly as you would with a PyKEEN dataset.

Note: I **highly** recommend that you use the `../TWIG-I_pipeline.sh` and `../TWIG-I_from_checkpoint.sh` for your training / finetuning as these files abstract away much of the complexitty that you do not need to deal with directly. They also take care of automatic logging and the such.

## Extending TWIG-I
We're all researchers, right? So you are probably here thinking: "This is super cool (well, I hope you are thinking that at least!). I have a super awesome idea that uses TWIG-I how do I implement it?"

This section will teach you how to do that. A word, however -- while I have made every effort to amke this code interoperatble, modular, and extensible, it is still research code and is not as well-build as other systems (say PyKEEN) which ahve large development teams. If you have any issues, please raise an issue on GitHub and I'd be happy to help out!

### Making a New Triple Scoring Model
TWIG-I scores triples given a single triple feature vector, and outputting a single score (higher scores represnt that a triple is more plausibly true). If you want to add a new model, you can do this in `twig_nn.py`. Just copy on of the existing models and edit however you would like.

As long as you accept `n_local` many features as input in your firest layer and output (for each for in the input) a single scalr-valued score, it does not matter what operations you perform, and all of your changes can be made in (and only in) that one file.

### Removing Features from the Default Feature Vector
Want to do a feature ablation? We have you covered! It's quite easy -- just run `../TWIG-I_pipeline.sh` and pass the names of the features you **do not** want to use to its `$fts_blacklist` parameter on the command line. Easy as! As a quick note, all possible features (22 total) that are used by default are (in order):

Features describing the structure of the given triple (6):
- s_deg
- o_deg
- p_freq
- s_p_cofreq
- o_p_cofreq
- s_o_cofreq

Features decribing the neighbourhood of the subject node in the given triple (8):
- s_min_deg_neighbour
- s_max_deg_neighbour
- s_mean_deg_neighbour
- s_num_neighbours
- s_min_freq_rel
- s_max_freq_rel
- s_mean_freq_rel
- s_num_rels

Features decribing the neighbourhood of the object node in the given triple (8):
- o_min_deg_neighbour
- o_max_deg_neighbour
- o_mean_deg_neighbour
- o_num_neighbours
- o_min_freq_rel
- o_max_freq_rel
- o_mean_freq_rel
- o_num_rels

### Adding new Features to TWIG-I
Adding new features is quite straighforward. If you want to do this you need to modify the data loader. Since data loading is defined exactly once, these cchanges should propagate to everything, inclluding the negative sampler, seamlessly.

**load_data.py** -- Modify the `Structure_Loader` class to add you feature to the feature vecor it outputs from `__call__`. You may want to modify `build_neighbour_cache` if you want to precalculate some features to a cache as an optimisation strategy, but this is not necessary.

As long as you keep the general structure of the TWIG-I code, you should be able to do feature ablation (see above) on your custom features just as well as on the default features!

### Adding a New Negative Sampling Strategy
To add a new negative sampling strategy, you need to modify `negative_sampler.py`. Specifically, you must extend the `Negative_Sampler` class and implement the `get_negatives` function. Note that you abolsutely MUST define two modes for your negative sampler:
- training, where the sampler generates some number of negatives per positive
- testing, where it generates all possible negatives (possibly with filtering of "negatives" that have been oserved in the train / test / valid sets). This way, you can use the standard evaluation mode for LP.

If you want, you can always create an `Optimised_Negative_Sampler` object as an attribute of your custom negative sampler, and use it for the testing case. This way, you only need to think about how you want to generate negatives for training, and you can simply defer the call to the `Optimised_Negative_Sampler`during th testing phase in your negative sampler. For examples of how to write a negative sampler, feel free to take a look at `Optimised_Negative_Sampler`.

If your negative sampler needs more arguments to its init function that are normally given, you cna add those to it in `run_exp.py`.

### Adding a New Loss Function
To add a new loss function, you need to edit the `loss.py` file. All losses right now are assumed to be pairwise losses -- i.e. comparing the score of each negative to its positive in 1:1 pairs and penalising cases where the negative outscores the positive. If you want to implement a new pairwise loss function, it will be super easy: just copy one of the losses in the file and write your own equation inside!

Once you have your loss, tell the `trainer.py` file to use it by replacing `loss_fn = ...` with the name of the getter for your function (i.e. `loss_fn = get_my_new_loss(your_hyperparams)`)

If you want to implement something that is ot a pairwise loss (i.e. pointwise or setwise losses, or other losses), you will need to not only modify `loss.py` as above, but may also need to modify how scores are calcualted and send to the loss function in `do_batch` in `trainer.py`. This depends heavily on hte loss you are implementing, so I will leave it there for now.

## Understanding TWIG-I at Scale
### Memory Scaling
I've put a huge amount of effort into making TWIG-I scale in terms of memory -- bad implementations of TWIG-I take up TONS of memory (i know this from direct experience...). As such, it's memory scalability is actually pretty deccent (though I am sure it could be improved further here and there). 

TWIG'I's memory usage is determined by a few key components:

**The Input Feature Vectors.**
All data is laoded as (pre-calculated) input feature vectors. These consist of 22 features, plus a 23rd element that is a flag for the ID of the triple from which the features come. (This flag is stripped before the feature vectors are used in learning so that it cannot lead to data leakage, but is needed so that the negative sample knows what triple to generate negatives for.) This means that we have (using float32) 92 btyes of memory per triple. At scale, this means that
    - a KG of 10,000 triples will require around 0.87 MB
    - a KG of 100,000 triples will require around 8.7 MB
    - a KG of 100 million triples will require around 8.6 GB

We note that we must draw a constrast to Knoweldge Graph Embedding (KGE) methods. KGE mothods scale by the number of *nodes and edges* in a KG, regardless of the density of that graph. Is a KG has 1,000 nodes and 10 edges, it will take the same memory for a KGE model to learn it whether those form 2,000 triples or 200,000 triples.

For TWIG-I, it does not matter at all how many nodes or edges are present, but it *oes* matter how many triples are present. As such, TWIG-I will often require more memory that KGEs on denser graphs, and less memory than KGEs on sparser graphs.

**The Structure Loader / Negative Sampler.**
The memory consumed by the structure load (which is a backend for the negative sampler as well as used in directly loading positive triples) depends massively in implementation. If you try to cache all possible negatives -- you will run out of memory on anything with over 10K triples. However, using the current caching strategy, we get very clean memory consumption. For this, we cache features for each node and node-relation or node-node pair. This, therefore, depends on the number of nodes and relations in a KG, but not on the number of triples.

For each node, we store 8 features (from the coarse-grained features) plus its degree, for a total of 9 features per node. For each predicate we store 1 value (its frequency). For each ordered pair of nodes we store their co-frequency (1 value for each pair observed) and for each node-predicate pair we store their co-frequency (2 values, as information on if the node is at subject or object is also included here). Thhis means that we have a cache that scales as so:
- 9 * #nodes
- 1 * #relations
- 1 * #node-pairs
- 2 * #node-rel-pairs

Note how the second two terms will be largeer in more dense graphs -- that is a pattern with TWIG-I,. that it gets richer (but more) information from more dense graphs. We also see that TWIG-I's current negative sampling approach scales much more strongly in terms of the number of nodes than the number of relations.

While the totaly number of node pairs and node-relation pairs are quite hard to predict in a KG, we know every KG's count of nodes and relations quite easily. Looking at those terms only (which will give us an underestiamte, but hopefully not too low of one), we see the following memory patterns (assuming 32-bit integets are used, and doubling memry to account for the keys in the Python dictionary for all these elements in the cache):
- for a graph with 10,000 nodes and 10 predicates, we use approximately 704 KB
- for a graph with 100,000 nodes and 100 predicates, we use approximately 6.86 MB
- for a graph with 1 million nodes and 1,000 predicates, we use approximately 68.6 MB

This does not account for the node-node and node-relation pair data (or for the overhead of the Python dictionary oobject), but it's likely safe to assume that even on a large KG of around 1M nodes that well under 1GB of memory will be used by the structure loader cache.

Fineally, for those who are *really* interested, this memory is stored in a cache in RAM -- it is used to construct vectors on the GPU. This means that this memory usage will almost certainly not contribute to GPU-related OOMs. The memory used by a batch is dependent on the batch size and the number of negatives wanted per positive -- to get this, multiple batch size by (1 + npp) to get the nuber of (positive and negatve) feature vectors created per batch. You can then ccalculate estiamted per-batch additional on-GPU memory using this, plus the number of features (defaul 22; the triple ID flag will be been stripped at this point), the number of bytes per feature (default 4, as float32 is used).

### What Memory does NOT Scale By
TWIG's memory usage is **not** significantly impacted by other components (that you might be tempted to think have an influence on its memory use)
- **The TWIG-I neural architecture.** TWIG-I's neural architecutre has (including bias terms) 351 parameters. Using float32 as a datatype, this means that around 1.4KB of memory for all those parameters. Add on a bit for gradients and the such, and this is safely remain under 10KB of memory. Unless you *massively* increase the size of this netowrk (to, say 1 million parameters), it is pretty much guaranteed to enver be a memory bottleneck. And there is no evidennce that such a large NN would every be needed or beneficial at the moment.

### Runtime Scaling
I have not heavily optimised TWIG-I for runtime, except to ensure it can run FB15k-237 and WN18RR in a reasonable amount of tie (under a day). You could probably get a 2x to 4x speedup by going through the code with (likely many) edits / re-writes for optimum speed. This section will go into the areas that (I think) are the most impactful on runtime (at the per-epoch level).

**The Negative Sampler**. The negative sampler is by far the biggest runtime bottleneck -- after all, without it, all we have is a 22-fearure input vector at a tiny (351-parameeter), 3-layer neural network. On top of that, 99% of the default negative sampler runs on CPU, not GPU, wich means it will be a bottleneck in an otherwise all-GPU pipeline. These operations are actually quite fast dictioanry lookups, but en masse they take a lot of time. As such, any attempt to optimise runtime should probably begin with a critical re-analysis of the negative sampler.

I actually have an improvment on this in the works (the Vector Negative Sampler) which could possibly lead to massive improvements in training speed. However, the default negative sampler format (of currupting triples, and thereby acccessing the feature cache in RAM / CPU memory) must be used in evaluation for fairness and comparision to other link predictors. This means that there will probably always be some sort of a bottleneck there (even if we can manage to significantly reduce it).

**The Loss Calculation**. I **don't** mean the implementation in the `loss.py` file, which I would expect to be quite fast. What I mean here is specifically the way that `do_batch` in `trainer.py` prepared for a pairwise loss: is extends the tensor of all scores of positive triple to match the size of the tensor of the scores of all negatives (see the call to `repeat_interleave`). While this results in the code for loss calculation to be extremely simple, it adds a lot of memory and requires Tensor creation at every step, which I expect takes some time. It may not be an issue -- but I have a feeling this part of the code could be optimised further.

### Other Extensions
I'm not sure what other ideas you may have -- and that's awesome! But it also means I have not planned for TWIG-I to be extended that way (yet), and may mean a bit more heavy lifting in the code than the above options would require. Feel free to raise an isssue / question on GitHub if you have other ideas but are unsure of how to implement them, and I'd be happy to help!

## FAQs
What versions of Python / PyTorch / etc does thi work on? Short answer -- anything "recent" (as of 2022 - 2024) *should* work. I've done a few on  afew different computers (meaning slightly different graphics cards, etc) and installing the latest libraries, so far, has always worked. I *highly* suggest making a conda env just for TWIG-I to avoid potential conflicts with other packages or projects, btu that's up to you. To be specific, what I use (and what works for me) is a conda environment with:
- Python 3.9
- torch 2.0.1 with Cuda cu11
- pykeen 1.10.1

Will TWIG-I be maintained? Yes! At least for a time, as this is under active deleopment as a part of Jeffrey's PhD thesis.

Are future TWIG-I releases planned? Sort of. I do plan to keep developing TWIG-I, but am unsure what for that will take, what opportunities will open up as a result, etc. Future releases will likely have increased research tooling, such as graph structure amalysers and enhanced ablation study capacity.

Can I help contribute to TWIG-I? Definitely -- that's whhy it's open source! We don't have a contributor policy, however -- at least, no more than "try to match the style of the existing code". I also don't have automated testing, which means merging will be slow to make sure everything still works mantually. If you have an idea you have implemented, raise a PR and we can take it from there!

Can I use TWIG-I in my research? Definitely! Just make sure to cite it (see the citation below).

Can I use TWIG-I for commercial pruposes? Currently TWIG-I is made available under a non-commercial license. If you do want to use TWIG-I in a commerical setting, send me an email and we can talk.

Do you have any other TWIG-I adjacent work? Yep -- It's called TWIG ;)). TWIG-I is actuially a re-imagining of TWIG. You can find TWIG here: https://github.com/Jeffrey-Sardina/TWIG-release-1.0. The differece is that TWIG learns to **simulate and predict the output of KGEs**, where as **TWIG-I does link prediction itself**. They are based on a very similar codebase.

Can I pip / apt-get / conda / etc install TWIG-I? Not yet, sorry! I'll try for pip at some poiont, but I have never messed with such dark magics before.

## Citation
If you have found this helpful, or used TWIG-I in your work, please drop us a citation at:
```
Citation bibtex pending
```
