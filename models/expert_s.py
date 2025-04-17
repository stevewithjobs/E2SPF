import torch

from torch import nn
from torch.nn import init
import torch.nn.functional as F
import time


def merge_alladj(A, B, crossA, crossB):

    top = torch.cat((A, crossB), dim=-1)  # Shape: (batch_size, window_size, h1, w1 + w2)
    bottom = torch.cat((crossA, B), dim=-1)  # Shape: (batch_size, window_size, h2, w1 + w2)
    C = torch.cat((top, bottom), dim=-2)  # Shape: (batch_size, window_size, h1 + h2, w1 + w2)
    
    return C

class GraphConvolution(nn.Module):
    def __init__(self, window_size, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.window_size = window_size
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(
            torch.Tensor(window_size, in_features, out_features)
        )

        self.t = 0
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weights)

    def forward(self, adjacency, nodes):
        """
        :param adjacency: FloatTensor (batch_size, window_size, node_num, node_num)
        :param nodes: FloatTensor (batch_size, window_size, node_num, in_features)
        :return output: FloatTensor (batch_size, window_size, node_num, out_features)
        """
        t1 = time.time()
        batch_size = adjacency.size(0)
        window_size = adjacency.size(1)

        weights = self.weights.unsqueeze(0).expand(batch_size, self.window_size, self.in_features, self.out_features)
        weights = weights[:, -window_size:, :, :]
        
        output = adjacency.matmul(nodes).matmul(weights)
        t2 = time.time()
        self.t += t2 -t1
        # print("gcn_time", self.t)
        # self.t += 1
        return output

import torch
import torch.nn as nn

class NodeSampler(nn.Module):
    def __init__(self, window_size, node_num, sample_node_num):

        super(NodeSampler, self).__init__()
        # self.mask = torch.rand(node_num)
        self.window_size = window_size
        self.node_num = node_num
        self.gate = nn.Parameter(torch.rand(node_num))
        self.gate1 = nn.Parameter(torch.zeros(node_num))
        self.gate2 = nn.Parameter(torch.zeros(node_num))
        self.sample_node_num = sample_node_num
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        
        self.conf = self.sigmoid(self.gate)
        self.conf1 = self.sigmoid(self.gate1)
        self.conf2 = self.sigmoid(self.gate2)

        connect = self.conf >= 0.7
        mask1 = self.conf1 < 0.5
        mask2 = self.conf2 < 0.5

        return connect, mask1, mask2

class Generator(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        self.out_features = out_features
        # batch_size, window_size, node_num = bike_in_shots.size()[0: 3]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
        self.gcn1 = GraphConvolution(window_size, in_features, out_features)  
        self.gcn2 = GraphConvolution(window_size, out_features, out_features)

        self.adjtime = 0
        self.gcntime = 0
        self.t = 0
        
        # Create zero tensors for the cross connections
        self.crossA = torch.zeros((self.batch_size, window_size, node_num, node_num),  device=device)
        self.crossB = torch.zeros((self.batch_size, window_size, node_num, node_num),  device=device)

    def forward(self, bike_in_shots, bike_adj, taxi_in_shots, taxi_adj, connect):
        """
        :param bike_in_shots: FloatTensor (batch_size, window_size, node_num, in_features)
        :param bike_adj: FloatTensor (batch_size, window_size, node_num, node_num)
        :param taxi_in_shots: FloatTensor (batch_size, window_size, node_num, in_features)
        :param taxi_adj: FloatTensor (batch_size, window_size, node_num, node_num)
        :return bike_gcn_output: FloatTensor (batch_size, node_num, node_num * out_features)
        :return taxi_gcn_output: FloatTensor (batch_size, node_num, node_num * out_features)
        """
        
        # batch_size, window_size, node_num = bike_in_shots.size()[0: 3]
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
        t1 = time.time()
        bike_adj = bike_adj + self.eye
        bike_diag = bike_adj.sum(dim=-1, keepdim=True).pow(-0.5).expand(bike_adj.size()) * self.eye
        bike_adjacency = bike_diag.matmul(bike_adj).matmul(bike_diag)
        taxi_adj = taxi_adj + self.eye
        taxi_diag = taxi_adj.sum(dim=-1, keepdim=True).pow(-0.5).expand(taxi_adj.size()) * self.eye
        taxi_adjacency = taxi_diag.matmul(taxi_adj).matmul(taxi_diag)

        t2 = time.time()
        if self.t > 2:
            self.adjtime += t2 - t1

        t1 = time.time()
        in_shots = torch.cat((bike_in_shots, taxi_in_shots), dim=-2)

        adj = merge_alladj(bike_adjacency, taxi_adjacency, self.crossA, self.crossB)

        t2 = time.time()
        if self.t > 2 :
            self.gcntime += t2 - t1
        # print("gcntime", self.gcntime)
        
        gcn_output1 = self.gcn1(adj, in_shots)

        bike_gcn_output = gcn_output1[:, :, :self.node_num, :]
        # self.bike_cache = bike_gcn_output
        taxi_gcn_output = gcn_output1[:, :, self.node_num:, :]
        # self.taxi_cache = taxi_gcn_output

        self.t += 1
        
        return bike_gcn_output, taxi_gcn_output

        


