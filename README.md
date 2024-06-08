# VilLain: Self-Supervised Learning on Homogeneous Hypergraphs without Features via Virtual Label Propagation
Source code for the paper **VilLain: Self-Supervised Learning on Homogeneous Hypergraphs without Features via Virtual Label Propagation**, Geon Lee, Soo Yong Lee, and Kijung Shin, WWW 2024.

## Datasets
* Datasets used in the paper are in [data](data).
* The input file is the hypergraph which is in the form:
```
A pair (V_idx, E_idx) of PyTorch tensors lengthed (# nonzero elements of the incidence matrix)
Node V_idx[i] is contained in hyperedge E_idx[i].

e.g.,
[tensor([     0,      1,      2,  ...,  80632,   5121, 260208]),
 tensor([    0,     0,     0,  ..., 31963, 31963, 31963])]
```

## How to Run
* To run demos, execute the following command in [code](code):
```
./run.sh
```

* In VilLain, there are several hyperparameters:
```
dim:            embedding dimension (default: 128)
lr:             learning rate (default: 0.01)
num_step:       number of propagation steps for training, i.e., k in the paper (default: 4)
num_step_gen:   number of propagation steps for inference, i.e., k' in the paper (default: 10 or 100)
nl:             number of v-labels in each subspace, i.e., d/D (default: 2)
```

* To run with specified hyperparameters, execute following commends in [code](code):
```
python main.py --dataset [DATASET NAME] --dim [DIMENSION] --lr [LEARNING RATE] --num_step [# PROPAGATION STEP (TRAINING)] --num_step_gen [# PROPAGATION STEP (INFERENCE)]

e.g.,
python main.py --dataset cora --dim 128 --lr 0.01 --num_step 4 --num_step_gen 10 --nl 2
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
