import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
from torch_scatter import scatter_mean

import utils 
import warnings

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

EPS = 1e-10

class model(nn.Module):
    def __init__(self, V_idx, E_idx, V, E, num_subspace, dim_subspace, num_step, num_step_gen):
        super(model, self).__init__()
        self.V_idx, self.E_idx = V_idx, E_idx
        self.V, self.E = V, E
        
        self.num_subspace = num_subspace
        self.dim_subspace = dim_subspace
        
        self.node_embedding = torch.nn.Parameter(torch.FloatTensor(V, num_subspace * dim_subspace))
        
        self.num_step = num_step
        self.num_step_gen = num_step_gen
        self.tau = 1.0
        
        self.cos = nn.CosineSimilarity(dim=0, eps=EPS)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.node_embedding, mean=0.0, std=0.1)
        
    def forward(self):
        _X0_pre = self.node_embedding   
        _X0 = F.gumbel_softmax(_X0_pre.view(-1, self.num_subspace, self.dim_subspace), tau=self.tau, dim=2, hard=False)
        X0 = _X0.view(-1, self.num_subspace * self.dim_subspace)
        Xj = X0
        
        loss_local, loss_global = 0, 0
        
        for i in range(1, self.num_step+1):
            ##### Hyperedges #####
            _Yj = scatter_mean(Xj[self.V_idx], self.E_idx, 0).view(-1, self.num_subspace, self.dim_subspace)
            Yj = _Yj.view(-1, self.num_subspace * self.dim_subspace)
            
            Yj_entropy = - torch.sum(_Yj * torch.log(_Yj + EPS), 2)
            loss_local += torch.mean(torch.mean(Yj_entropy, 1))
            
            p_Y = torch.sum(_Yj, 0) / self.E
            loss_global += (- torch.mean(- torch.sum(p_Y * torch.log(p_Y + EPS), 1)) + self.discrimination(_Yj))
            
            ##### Nodes #####
            _Xj = scatter_mean(Yj[self.E_idx], self.V_idx, 0).view(-1, self.num_subspace, self.dim_subspace)
            Xj = _Xj.view(-1, self.num_subspace * self.dim_subspace)
            
            Xj_entropy = - torch.sum(_Xj * torch.log(_Xj + EPS), 2)
            loss_local += torch.mean(torch.mean(Xj_entropy, 1))
            
            p_X = torch.sum(_Xj, 0) / self.V
            loss_global += (- torch.mean(- torch.sum(p_X * torch.log(p_X + EPS), 1)) + self.discrimination(_Xj))
            
        return loss_local, loss_global
    
    def discrimination(self, X):
        num_data, num_subspace, dim_subspace = X.shape
        idx_i = torch.arange(dim_subspace).repeat(dim_subspace)
        idx_j = torch.arange(dim_subspace).repeat_interleave(dim_subspace)
        
        C = self.cos(X[:,:,idx_i], X[:,:,idx_j]).view(-1, dim_subspace, dim_subspace)
        C = torch.softmax(C, 2)
        C = torch.diagonal(C, dim1=2)
        
        return torch.mean(- torch.log(C))
 
    def get_node_embeds(self, node=True):
        with torch.no_grad():
            _X0_pre = self.node_embedding
            _X0 = F.gumbel_softmax(_X0_pre.view(-1, self.num_subspace, self.dim_subspace), tau=self.tau, dim=2, hard=False)
            X0 = _X0.view(-1, self.num_subspace * self.dim_subspace)
            Xj = X0

            if node:
                _X = torch.zeros(self.V, self.num_subspace * self.dim_subspace).to(X0.device)
            else:
                _Y = torch.zeros(self.E, self.num_subspace * self.dim_subspace).to(X0.device)

            for i in range(1, self.num_step_gen+1):
                Yj = scatter_mean(Xj[self.V_idx], self.E_idx, 0)
                Xj = scatter_mean(Yj[self.E_idx], self.V_idx, 0)
                if node:
                    _X += Xj.view(-1, self.num_subspace, self.dim_subspace).reshape(self.V, -1) 
                else:
                    _Y += Yj.view(-1, self.num_subspace, self.dim_subspace).reshape(self.E, -1)
            
            if node:
                _X /= self.num_step_gen
            else:
                _Y /= self.num_step_gen
                
            if node:
                return _X
            else:
                return _Y
    
