import os
import math
import argparse
import random
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cora', type=str, help='dataset')
    parser.add_argument("--emb_path", default='embs', type=str, help='embedding path')
    parser.add_argument("--k", default=4, type=int, help='k (training)')
    parser.add_argument("--K", default=10, type=int, help='k (inference)')
    parser.add_argument("--config", default=None, type=str, help='configuration')
    parser.add_argument("--split", default='-1', type=str, help='split')
    return parser.parse_args()

args = parse_args()

emb_files = os.listdir(args.emb_path)

data_files = []

for emb_file in emb_files:
    if args.dataset not in emb_file:
        continue
        
    if '0.01' not in emb_file:
        continue
        
    if args.split not in emb_file:
        continue
        
    terms = emb_file.split('_')
    nl = int(terms[2])
    k = int(terms[3])
    K = int(terms[4])
    
    if k != args.k or K != args.K:
        continue
        
    data_files.append(emb_file)
    
    
Z = {}
for emb_file in data_files:
    with open(f'{args.emb_path}/{emb_file}', 'rb') as f:
        x = pkl.load(f)
        
    x = x[:,:128]
    
    dim_subspace = int(emb_file.split('_')[2])
    num_subspace = math.ceil(128 / (dim_subspace - 1))
    
    Z[dim_subspace] = x
    
print(args.k, args.K, args.split)
print(Z.keys())
y = np.concatenate((Z[2], Z[3], Z[4], Z[5], Z[6], Z[7], Z[8]), 1)

pca = PCA(n_components=128)
y = pca.fit_transform(y)
#y = (Z[2] + Z[3] + Z[4] + Z[5] + Z[6] + Z[7] + Z[8]) / 7

print(args.dataset, args.k, args.K, args.split)
with open('embs/' + args.dataset + f'_{args.k}_{args.K}_{args.split}.pkl', 'wb') as f:
    pkl.dump(y, f)
