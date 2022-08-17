import os
import math
import pickle as pkl
import numpy as np
import argparse
from tqdm import tqdm, trange
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", default='cora', type=str, help='dataset')
    parser.add_argument("--gpu", default='0', type=str, help='gpu number')
    parser.add_argument("--seed", default=2020, type=int, help='seed number')
    
    parser.add_argument("--dim", default=128, type=int, help='embedding dimension')
    parser.add_argument("--learning_rate", default=1e-2, type=float, help='learning rate')
    
    parser.add_argument("--num_step", default=4, type=int, help='k in the paper')
    parser.add_argument("--gen_step", default=100, type=int, help='k in the paper')
    
    # Fixed parameters
    parser.add_argument("--pca", default=0.99, type=float, help='explained variance ratio')
    parser.add_argument("--epochs", default=5000, type=int, help='number of epochs')
    parser.add_argument("--tau", default=1.0, type=float, help='softmax temperature')
    
    return parser.parse_args()

def load(dataset):
    data_path = os.path.join('..', 'data', dataset)
    
    with open(os.path.join(data_path, 'H.pickle'), 'rb') as f:
        H = pkl.load(f)
    
    return H

def generate_batches(data_size, batch_size, shuffle=True):
    data = np.arange(data_size)
    
    if shuffle:
        np.random.shuffle(data)
    
    batch_num = math.ceil(data_size / batch_size)
    batches = np.split(np.arange(batch_num * batch_size), batch_num)
    batches[-1] = batches[-1][:(data_size - batch_size * (batch_num - 1))]
    
    for i, batch in enumerate(batches):
        batches[i] = [data[j] for j in batch]
        
    return batches

def motif_incidence(H, T=1.4):
    V, E = H.shape
    line_graph = H.T.matmul(H)
    
    M = []
    
    for i in trange(E):
        a = i
        e_a = set(torch.nonzero(H.T[a]).view(-1).tolist())
        size_a = len(e_a)
        N_a = torch.nonzero(line_graph[a]).view(-1)

        for j in range(len(N_a)):
            b = N_a[j].item()
            e_b = set(torch.nonzero(H.T[b]).view(-1).tolist())
            size_b = len(e_b)
            C_ab = len(e_a & e_b)

            for k in range(j+1, len(N_a)):
                c = N_a[k].item()
                e_c = set(torch.nonzero(H.T[c]).view(-1).tolist())
                size_c = len(e_c)

                C_ca = len(e_c & e_a)
                C_bc = len(e_b & e_c)
                C_abc = len(e_a & e_b & e_c)

                if C_bc:
                    if i < j:
                        overlapness = (size_a + size_b + size_c) / len(e_a | e_b | e_c)
                        motif_index = motifs.get_index(size_a, size_b, size_c, C_ab, C_bc, C_ca, C_abc)
                        #if motif_index in [4, 10]:
                        if overlapness >= 1.4:
                            M.append(torch.max(H.T[a], torch.max(H.T[b], H.T[c])))
                else:
                    overlapness = (size_a + size_b + size_c) / len(e_a | e_b | e_c)
                    motif_index = motifs.get_index(size_a, size_b, size_c, C_ab, C_bc, C_ca, C_abc)
                    #if motif_index in [4, 10]:
                    if overlapness >= 1.4:
                        M.append(torch.max(H.T[a], torch.max(H.T[b], H.T[c])))
        M = torch.stack(M).T
        return M
