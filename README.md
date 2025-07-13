# VilLain: Self-Supervised Learning on Homogeneous Hypergraphs without Features via Virtual Label Propagation
Source code for the paper **VilLain: Self-Supervised Learning on Homogeneous Hypergraphs without Features via Virtual Label Propagation**, Geon Lee, Soo Yong Lee, and Kijung Shin, WWW 2024.

* Paper: [PDF](http://dmlab.kaist.ac.kr/~kijungs/papers/villainWWW2024.pdf)
* Poster: [PDF](http://dmlab.kaist.ac.kr/~kijungs/posters/villainWWW2024.pdf)

## Datasets
* Datasets used in the paper (Amazon, Trivago, Cora, Citeseer, Pubmed, DBLP, Primary, and High) are in [data](data).
* The input file is the hypergraph which is in the form:
```
A pair (V_idx, E_idx) of PyTorch tensors lengthed (# nonzero elements of the incidence matrix)
Node V_idx[i] is contained in hyperedge E_idx[i].

e.g.,
[tensor([     0,      1,      2,  ...,  80632,   5121, 260208]),
 tensor([    0,     0,     0,  ..., 31963, 31963, 31963])]
```

* The hypergraph data can be loaded using the following command:
```
with open(os.path.join(data_path, f'H.pickle'), 'rb') as f:
    H = pkl.load(f)

V_idx = H[0]
E_idx = H[1]
```

## How to Run VilLain
* Some main hyperparameters of VilLain are:
```
dim:            embedding dimension (default: 128)
num_step:       number of propagation steps for training, i.e., k in the paper (default: 4)
num_step_gen:   number of propagation steps for inference, i.e., k' in the paper (default: 10 or 100)
nl:             number of v-labels in each subspace, i.e., d/D (default: 2)
```

* To run with specified hyperparameters, execute [main.py](code/main.py):
```
python main.py --gpu [GPU ID] --dataset [DATASET NAME] --num_step [# PROPAGATION STEP (TRAINING)] --num_step_gen [# PROPAGATION STEP (INFERENCE)] --lr [LEARNING RATE] --num_labels [NUMBER OF LABELS] --dim [DIMENSION]

e.g.,
python main.py --gpu 0 --dataset cora --num_step 4 --num_step_gen 100 --lr 0.01 --num_labels 2 --dim 128
```

* To aggregate the embeddings learned with different numbers of v-labels, execute [emb_concat.py](code/emb_concat.py):
```
python emb_concat.py --dataset [DATASET NAME] --num_step [# PROPAGATION STEP (TRAINING)] --num_step_gen [# PROPAGATION STEP (INFERENCE)] --dim [DIMENSION]

e.g.,
python emb_concat.py --dataset cora --num_step 4 --num_step_gen 100 --dim 128
```

* To run a demo, execute the following command in [code](code):
```
./run.sh
```

* Embeddings will be saved in [code/embs](code/embs) as a PyTorch tensor. The final embedding is saved as **_merged.pkl**.
