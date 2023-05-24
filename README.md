# VilLain: Self-Supervised Learning on Hypergraphs without Features via Virtual Label Propagation
Source code for the submitted paper [VilLain: Hypergraph Embedding with No Labels and No Features via Virtual Label Propagation](README.md).

*Group interactions arise in various scenarios in real-world systems: collaborations of researchers, co-purchases of products, and discussions in online Q&A sites, to name a few. Such higher-order relations are naturally modeled as hypergraphs, which consist of hyperedges (i.e., any-sized subsets of nodes). For hypergraphs, the challenge to learn node representation when features or labels are not available is imminent, given that (a) most real-world hypergraphs are not equipped with external features while (b) most existing approaches for hypergraph learning resort to additional information. Thus, in this work, we propose VilLain, a novel self-supervised hypergraph representation learning method based on the propagation of virtual labels (v-labels). Specifically, we construct for each node a sparse probability distribution over v-labels as its feature vector, and we propagate the vectors to construct the final node embeddings. Inspired by higher-order label homogeneity, which we discover in real-world hypergraphs, we design novel self-supervised loss functions for v-labels to reproduce the higher-order structure-label pattern. We demonstrate that VilLain is: (a) Requirement-free: learning node embeddings without relying on node labels and features, (b) Versatile: giving embeddings that are not specialized to specific tasks but generalizable to diverse downstream tasks, and (c) Accurate: more accurate than its competitors for node classification, node retrieval, and hyperedge prediction tasks. Our code is available at https://anonymous.4open.science/r/VilLain-C18B.*

## Datasets
* Datasets used in the paper is in [data](data).
* Input file is the hypergraph which is in the form:
```
A pair (V_idx, E_idx) of PyTorch tensors lengthed (# nonzero elements of the incidence matrix)
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
lr:             learning rate (default: 0.01)
num_step:       number of propagation steps for training, i.e., k in the paper (default: 4)
num_step_gen:   number of propagation steps for inference, i.e., k' in the paper (default: 10 or 100)
pca:            explained variance ratio for PCA (default: 0.99)
```

* To run with specified hyperparameters, execute following commends in [code](code):
```
python main.py --dataset [DATASET NAME] --dim [DIMENSION] --lr [LEARNING RATE] --num_step [# PROPAGATION STEP (TRAINING)] --num_step_gen [# PROPAGATION STEP (INFERENCE)] --pca [eplained variance ratio]

e.g.,
python main.py --dataset cora --dim 128 --lr 0.01 --num_step 4 --num_step_gen 10 --pca 0.99
```

* Embeddings will be saved in [code/embs](code/embs) as a PyTorch tensor:
```
e.g.,
tensor([[-2.5536,  0.1871, -0.6198,  ...,  0.0245,  0.4387, -0.0706],
        [ 0.2559,  1.8595,  2.9310,  ...,  0.0047, -0.2317, -0.2191],
        [-0.8752,  2.4186, -1.1438,  ...,  0.1499,  0.0692,  0.1509],
        ...,
        [-1.2408, -0.4067,  1.0191,  ..., -0.0158, -0.0975, -0.2476],
        [ 0.6102, -2.2991, -0.5856,  ..., -0.1927, -0.1565,  0.6828],
        [-1.9437,  1.8978, -0.5164,  ..., -0.0729, -0.4142, -0.2966]])
```
