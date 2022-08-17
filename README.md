# VilLain: Unsupervised Hypergraph Embedding via Virtual Label Propagation
Source code for the submitted paper [VilLain: Unsupervised Hypergraph Embedding via Virtual Label Propagation](README.md).

*Groupwise interactions arise in various scenarios in real-world systems: collaborations of researchers, co-purchases of products, and discussions in online Q&A sites, to name a few. Such higher-order relations are naturally modeled as hypergraphs, which consist of hyperedges (i.e., any-sized subsets of nodes). An important task is to represent the nodes in a hypergraph as structure-preserving embeddings (i.e., low-dimensional vectors), which can be readily used for various downstream tasks (e.g., node classification and hyperedge prediction). Importantly, it is desirable that such embeddings can be obtained even with minimal requirements, i.e., without relying on labels, features, augmentations, and negative samples. However, previous methods resort to additional information or manipulation of the data, which can be problematic in realistic setups.*
*In this work, we propose VilLain, a novel hypergraph embedding method based on the propagation of virtual labels (v-labels). Aiming to reproduce higher-order homogeneity, which we observe in real-world hypergraphs, VilLain first learns, for each node, the probability distribution over v-labels. Then, VilLain obtains embeddings by propagating the distributions over the hypergraph. With additional schemes for automatic combination of multiple embeddings while reducing redundancy, VilLain is: **(a) Requirement-free:** not requiring extra information (e.g., labels and features) or data manipulation (e.g., augmentation and negative sampling), **(b) Versatile:** giving embeddings that are not specialized to specific tasks but generalizable to diverse downstream tasks, **(c) Accurate:** being up to 66.4% and 29.3% more accurate than its competitors for node classification and hyperedge prediction tasks, respectively.*

## Datasets
* Datasets used in the paper is in [data](data).
* Input file is the hypergraph which can be either in the form of Type 1 or Type 2:
```
Type 1: A PyTorch tensor shaped (# nodes) X (# hyperedges)

e.g.,
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], dtype=torch.float64)
```
```
Type 2: A pair (V_idx, E_idx) of PyTorch tensors lengthed (# nonzero elements of the incidence matrix)
        Node V_idx[i] is contained in hyperedge E_idx[i].

e.g.,
[tensor([     0,      1,      2,  ...,  80632,   5121, 260208]),
 tensor([    0,     0,     0,  ..., 31963, 31963, 31963])]
```

## How to Run
* To run demos, execute following commend in [code](code):
```
./run.sh
```
* In VilLain, there are several hyperparameters:
```
dim:            embedding dimension (default: 128)
learning_rate:  learning rate (default: 0.01)
num_step:       number of propagation steps for training, i.e., k in the paper (default: 4)
gen_step:       number of propagation steps for inference, i.e., k' in the paper (default: 10 or 100)
```
* To run with specified hyperparameters, execute following commends in [code](https://github.com/geonlee0325/HashNWalk/tree/main/code):
```
python main.py --dataset [DATASET NAME] --dim [DIMENSION] --learning_rate [LEARNING RATE] --num_step [# PROPAGATION STEP (TRAINING)] --gen_step [# PROPAGATION STEP (INFERENCE)]

e.g.,
python main.py --dataset cora --dim 128 --learning_rate 0.01 --num_step 4 --gen_step 10
```
* Embeddings will be saved in [code/embs](code/embs) as a PyTorch tensor:
```
e.g.,
tensor([[-1.1989, -0.4377, -1.0248,  ...,  0.1267,  0.0519,  0.0599],
        [ 0.2100,  1.9606,  2.8593,  ..., -0.0185, -0.0200, -0.2071],
        [-1.3471,  2.2053, -1.0844,  ...,  0.0438, -0.1264,  0.0531],
        ...,
        [-0.9464, -0.8524,  1.3157,  ...,  0.1008, -0.0555, -0.1518],
        [-0.1419, -1.2256, -1.5207,  ..., -0.0369, -0.2210, -0.0169],
        [-2.3479,  1.4506, -0.5193,  ..., -0.1328,  0.0967, -0.2736]])
```
