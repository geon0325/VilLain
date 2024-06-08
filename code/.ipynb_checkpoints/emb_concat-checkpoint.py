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
    parser.add_argument("--num_step", default=4, type=int, help='k (training)')
    parser.add_argument("--num_step_gen", default=10, type=int, help='k (inference)')
    parser.add_argument("--config", default=None, type=str, help='configuration')
    parser.add_argument("--dim", default=128, type=int)
    return parser.parse_args()

args = parse_args()

emb_files = os.listdir('embs')
data_files = []

for emb_file in emb_files:
    if args.dataset not in emb_file:
        continue

    if f'dim{args.dim}' not in emb_file:
        continue
        
    if f'ns{args.num_step}_nsg{args.num_step_gen}' not in emb_file:
        continue
        
    data_files.append(emb_file)
    
data_files = sorted(data_files)
    
embs = []
for emb_file in data_files:
    with open(f'embs/{emb_file}', 'rb') as f:
        x = pkl.load(f)
    x = x[:,:args.dim]
    embs.append(x)

y = np.concatenate((embs), 1)

pca = PCA(n_components=args.dim)
y = pca.fit_transform(y)
print(y.shape)

with open('embs/' + args.dataset + f'_dim{args.dim}_ns{args.num_step}_nsg{args.num_step_gen}_merged.pkl', 'wb') as f:
    pkl.dump(y, f)
