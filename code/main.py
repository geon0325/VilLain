import os
import copy
import time
import math
import random
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
from collections import defaultdict
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim

import utils
import model

EPS = 1e-10

args = utils.parse_args()

##### GPU Settings #####
if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print(f'Device:\t{device}\n')

##### Config #####
config = f'{args.dataset}_{args.dim}_{args.lr}_{args.num_step}_{args.num_step_gen}_{args.pca}'

##### Random Seed #####
SEED = 2023
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

##### Read Data #####
H, labels = utils.load(args.dataset)
labels = labels.tolist()

V_idx, E_idx = H[0].to(device), H[1].to(device)
V, E = torch.max(V_idx)+1, torch.max(E_idx)+1

print(f'V = {V}\tE = {E}')
print(f'Sparsity = {len(V_idx)/(V*E)}')
    
################################################################################

P = torch.empty((V, 0))

remaining_dim = args.dim
dim_subspace = 2

while P.shape[1] < args.dim:
    
    num_subspace = math.ceil(remaining_dim / dim_subspace)
    print(dim_subspace, num_subspace)
    
    ##### Prepare Training #####
    our_model = model.model(V_idx, E_idx, num_subspace, dim_subspace, args.num_step, args.num_step_gen).to(device)
    optimizer = optim.AdamW(our_model.parameters(), lr=args.lr, weight_decay=1e-5)

    ##### Train Model #####
    
    best_loss, best_model, patience = 1e10, None, 50
    
    for epoch in range(1, args.epochs+1):
        our_model.train()

        loss_local, loss_global = our_model()
        loss = loss_local + loss_global
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(our_model.parameters(), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            print(f'{epoch}\t{loss_local.item()}\t{loss_global.item()}')
            with open(f'logs/{config}.txt', 'a') as f:
                f.write(f'Epoch {epoch}\tLocal {loss_local.item()}\tGlobal {loss_global.item()}\n')
        
        if epoch <= 1000:
            continue
            
        if loss.item() < best_loss:
            our_model.eval()
            best_loss = loss.item()
            best_model = copy.deepcopy(our_model.state_dict())
            cnt_wait = 0
        else:
            cnt_wait += 1
            
        if cnt_wait == patience:
            break
    
    # Process the embedding
    our_model.load_state_dict(best_model)
    best_X = our_model.get_node_embeds().detach().cpu().numpy()
        
    best_X_pca = PCA(n_components=args.pca).fit_transform(best_X)
    best_X_pca = torch.from_numpy(best_X_pca)
    
    P = torch.cat((P, best_X_pca), 1)
    print(P.shape)
    
    with open(f'logs/{config}.txt', 'a') as f:
        f.write(f'dim: {dim_subspace}\tP_shape: {P.shape[0]}X{P.shape[1]}\n')
    
    remaining_dim = args.dim - P.shape[1]
    dim_subspace += 1

P = P[:,:args.dim]
print(f'Final:\t{P.shape}')

with open(f'embs/{config}.pkl', 'wb') as f:
    pkl.dump(P, f)
