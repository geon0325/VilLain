# VilLain: Hypergraph Embedding with No Labels and No Features via Virtual Label Propagation
Source code for the submitted paper [VilLain: Hypergraph Embedding with No Labels and No Features via Virtual Label Propagation](README.md).

*Group interactions arise in various scenarios in real-world systems: collaborations of researchers, co-purchases of products, and discussions in online Q&A sites, to name a few. Such higher-order relations are naturally modeled as hypergraphs, which consist of hyperedges (i.e., any-sized subsets of nodes). An important task is to represent the nodes in a hypergraph as structure-preserving embeddings (i.e., low-dimensional vectors), which can be readily used for various downstream tasks (e.g., node classification/retrieval and hyperedge prediction). Importantly, it is desirable that such embeddings can be obtained even with minimal requirements, i.e., without relying on labels and features. However, most existing approaches for hypergraph embedding resort to additional information which can be hard to obtain in realistic setups.*
*In this work, we propose VilLain, a novel hypergraph embedding method based on the propagation of virtual labels (v-labels). Aiming to reproduce higher-order homogeneity, which we observe in real-world hypergraphs, VilLain first learns, for each node, the probability distribution over v-labels. Then, VilLain obtains embeddings by propagating the distributions over the hypergraph. With additional schemes for automatic combination of multiple embeddings, VilLain is: (a) Requirement-free: learns node embeddings without any supervision (e.g., node labels) or extra information (e.g., node attributes or even the number of unique labels), (b) Versatile: giving embeddings that are not specialized to specific tasks but generalizable to diverse downstream tasks, (c) Accurate: being up to 79.8%, 73.6%, and 6.5% more accurate than its competitors for node classification, node retrieval, and hyperedge prediction tasks, respectively, and (d) Concise: yielding up to 682.7X more concise node representations with similar node classification accuracy using an extra space-saving scheme.*

## Datasets
* Datasets used in the paper is in [data](data).
* Input file is the hypergraph which is in the form:

	# A pair (V_idx, E_idx) of PyTorch tensors lengthed (# nonzero elements of the incidence matrix)
	# Node V_idx[i] is contained in hyperedge E_idx[i].

	# e.g.,
	# [tensor([     0,      1,      2,  ...,  80632,   5121, 260208]),
 	#  tensor([    0,     0,     0,  ..., 31963, 31963, 31963])]

## How to Run
* To run demos, execute following commend in [code](code):

	# ./run.sh

* In VilLain, there are several hyperparameters:


	# dim:            embedding dimension (default: 128)
	# lr:             learning rate (default: 0.01)
	# num_step:       number of propagation steps for training, i.e., k in the paper (default: 4)
	# num_step_gen:   number of propagation steps for inference, i.e., k' in the paper (default: 10 or 100)
	# pca:            explained variance ratio for PCA (default: 0.99)

* To run with specified hyperparameters, execute following commends in [code](code):

	# python main.py --dataset [DATASET NAME] --dim [DIMENSION] --lr [LEARNING RATE] --num_step [# PROPAGATION STEP (TRAINING)] --num_step_gen [# PROPAGATION STEP (INFERENCE)] --pca [eplained variance ratio]

	# e.g.,
	# python main.py --dataset cora --dim 128 --lr 0.01 --num_step 4 --num_step_gen 10 --pca 0.99

* Embeddings will be saved in [code/embs](code/embs) as a PyTorch tensor:

	# e.g.,
	# tensor([[-2.5536,  0.1871, -0.6198,  ...,  0.0245,  0.4387, -0.0706],
	#         [ 0.2559,  1.8595,  2.9310,  ...,  0.0047, -0.2317, -0.2191],
	#         [-0.8752,  2.4186, -1.1438,  ...,  0.1499,  0.0692,  0.1509],
	#         ...,
	#         [-1.2408, -0.4067,  1.0191,  ..., -0.0158, -0.0975, -0.2476],
	#         [ 0.6102, -2.2991, -0.5856,  ..., -0.1927, -0.1565,  0.6828],
	#         [-1.9437,  1.8978, -0.5164,  ..., -0.0729, -0.4142, -0.2966]])
