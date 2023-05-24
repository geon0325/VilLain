import os
import math
import pickle as pkl
import numpy as np
import argparse
from tqdm import tqdm, trange
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", default='cora', type=str, help='dataset')
    parser.add_argument("--gpu", default='0', type=str, help='gpu number')
    parser.add_argument("--split", default=-1, type=int, help='train test split number')
    
    parser.add_argument("--batch_size", default=50000, type=int, help='batch size')
    parser.add_argument("--epochs", default=5000, type=int, help='number of epochs')
    parser.add_argument("--learning_rate", default=1e-2, type=float, help='learning rate')
    parser.add_argument("--dim", default=128, type=int, help='embedding dimension')
    parser.add_argument("--num_labels", default=2, type=int, help='number of labels')
    parser.add_argument("--tau", default=1.0, type=float, help='softmax temperature')
    parser.add_argument("--pca", default=0.99, type=float, help='explained variance ratio')
    parser.add_argument("--gb_hard", default=False, type=str2bool, help='gumbel softmax soft/hard')
    
    parser.add_argument("--num_step", default=4, type=int, help='number of steps')
    parser.add_argument("--num_step_gen", default=10, type=int, help='number of steps (generation)')
    
    return parser.parse_args()

def load(dataset, split):
    data_path = os.path.join('..', 'data', dataset)
    
    if split == -1:
        with open(os.path.join(data_path, 'H.pickle'), 'rb') as f:
            H = pkl.load(f)
    else:
        with open(os.path.join(data_path, f'H_train_{split}.pickle'), 'rb') as f:
            H = pkl.load(f)
    
    with open(os.path.join(data_path, 'L.pickle'), 'rb') as f:
        labels = pkl.load(f)
    
    return H, labels

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
