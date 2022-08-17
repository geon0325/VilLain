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
print('Device:\t', device, '\n')

##### Config #####
config = '{}_{}_{}_{}_{}'.format(args.dataset, args.dim, args.learning_rate, args.num_step, args.gen_step)

##### Random Seed #####
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

##### Read Data #####
H = utils.load(args.dataset)

if len(H) == 2:
    V_idx, E_idx = H
    V, E = torch.max(V_idx)+1, torch.max(E_idx)+1
else:
    idx = torch.nonzero(H)
    V_idx, E_idx = idx[:,0], idx[:,1]
    V, E = H.shape
    
V_idx, E_idx = V_idx.to(device), E_idx.to(device)

print('V = {}\tE = {}'.format(V,E))
    
################################################################################

P = torch.empty((V, 0))

remaining_dim = args.dim
dim_subspace = 2

while P.shape[1] < args.dim:
    
    num_subspace = math.ceil(remaining_dim / (dim_subspace-1))
    print(dim_subspace, num_subspace)
    
    ##### Prepare Training #####
    our_model = model.model(V_idx, E_idx, V, E, num_subspace, dim_subspace, args.num_step, args.gen_step, args.tau).to(device)
    optimizer = optim.AdamW(our_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    ##### Train Model #####
    
    best_loss, best_model, patience = 1e10, None, 50
    
    for epoch in range(1, args.epochs+1):
        our_model.train()

        loss_local, loss_global = our_model()

        _loss_local = sum(loss_local.values())
        _loss_global = sum(loss_global.values())
        loss = _loss_local + _loss_global

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(our_model.parameters(), 1.0)
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {}\tLocal {:.6f}\tGlobal {:6f}'.format(epoch, _loss_local.item(), _loss_global.item()))
            with open('logs/{}.txt'.format(config), 'a') as f:
                f.write('Epoch {}\tLocal {}\tGlobal {}\n'.format(epoch, _loss_local.item(), _loss_global.item()))
        
        if epoch <= 500:
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
    
    with open('logs/{}.txt'.format(config), 'a') as f:
        f.write('dim: {}\tP_shape: {}X{}\n'.format(dim_subspace, P.shape[0], P.shape[1]))
    
    remaining_dim = args.dim - P.shape[1]
    dim_subspace += 1

P = P[:,:args.dim]
print('Final:\t{}'.format(P.shape))

with open('embs/{}.pkl'.format(config), 'wb') as f:
    pkl.dump(P, f)
