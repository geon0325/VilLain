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
    
    parser.add_argument("--batch_size", default=50000, type=int, help='batch size')
    parser.add_argument("--epochs", default=5000, type=int, help='number of epochs')
    parser.add_argument("--lr", default=1e-2, type=float, help='learning rate')
    parser.add_argument("--dim", default=128, type=int, help='embedding dimension')
    parser.add_argument("--pca", default=0.99, type=float, help='explained variance ratio')
    
    parser.add_argument("--num_step", default=4, type=int, help='number of steps')
    parser.add_argument("--num_step_gen", default=100, type=int, help='number of steps (generation)')
    
    return parser.parse_args()

def load(dataset):
    data_path = os.path.join('..', 'data', dataset)
    
    with open(os.path.join(data_path, 'H.pickle'), 'rb') as f:
        H = pkl.load(f)
    
    with open(os.path.join(data_path, 'L.pickle'), 'rb') as f:
        labels = pkl.load(f)
    
    return H, labels
