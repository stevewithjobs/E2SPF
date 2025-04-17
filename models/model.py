import torch
from torch import nn
from models.smoe import GatedSpatialMoE2d, GatedSpatialMoE2d_s

from models.expert_s import Generator, NodeSampler
from models.expert_t import LstmAttention
from models.TimesNet import Model_onetimenet
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLinearGate2d, SpatialLatentTensorGate2d
from collections import deque

import functools
import time

class Config:
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels, e_layers, c_out, batch_size):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.e_layers = e_layers
        self.c_out = c_out
        self.batch_size = batch_size

class Net(nn.Module):
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config, pred_size):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.node_num = node_num
        self.window_size = window_size

        sampling_ratio = 2 /3
        sampled_node_num = int(node_num * sampling_ratio)
        self.nodesample = NodeSampler(
            node_num = node_num, 
            sample_node_num=sampled_node_num, 
            window_size=window_size
        )

        self.generator1 = Generator(
            batch_size=batch_size,
            window_size=window_size,
            node_num=node_num,
            in_features=2,
            out_features=8
        )

        self.pred_size = pred_size
        timesnetconfig = Config(
            seq_len=window_size, 
            pred_len=self.pred_size, 
            top_k=1, 
            d_model=node_num * 16, 
            d_ff=node_num * 2,  
            num_kernels=2, 
            e_layers=1, 
            c_out=node_num * 4,
            batch_size=batch_size
        )
    
        self.timesnet = Model_onetimenet(
             timesnetconfig,
            #  smoe_config,
        )


        self.fc1 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU()
        )
        self.gcntime = 0
        self.tstime = 0
        self.t = 0

    
    def forward(self, x, window_size):
        
        ''' spatio'''
        bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori = x[0], x[1], x[2], x[3]
        '''sample'''
        
        connect, mask1, mask2 = self.nodesample()
        
        gcn_output1, gcn_output2 = self.generator1(bike_node_ori, bike_adj_ori, taxi_node_ori, taxi_adj_ori, 0)
        
        gcn_output1[:, :, mask1, :] = 0
        gcn_output2[:, :, mask2, :] = 0

        gcn_output = torch.cat((gcn_output1, gcn_output2), dim=-1)
        
        '''temporal'''     
        
        gcn_output = gcn_output.view(self.batch_size, self.window_size, -1)

        t1 = time.time()
        gcn_output = gcn_output[:, -window_size:, :]
        t2 = time.time()
        if self.t > 2 :
            self.gcntime += t2 - t1
        print("slice", self.gcntime)

        t1 = time.time()

        timesnetout, _ = self.timesnet(bike_node_ori, gcn_output)
        
        timesnetout = timesnetout.view(self.batch_size, self.pred_size, self.node_num, -1)

        bike_start = self.fc1(timesnetout[:, :, :, 0])
        bike_end = self.fc2(timesnetout[:, :, :, 1])

        taxi_start = self.fc3(timesnetout[:, :, :, 2])
        taxi_end = self.fc4(timesnetout[:, :, :, 3])

        t2 = time.time()
        if self.t > 1 :
            self.tstime += t2 - t1
        # print("tstime", self.tstime)
        self.t += 1
        return bike_start, bike_end, taxi_start, taxi_end


    