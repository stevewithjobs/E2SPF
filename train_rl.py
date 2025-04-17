import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse
from collections import deque
from utils.mdtp import MyDataset_mmsm, set_seed, metric1
from torch.utils.data.dataloader import DataLoader
from models.model import Net
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLinearGate2d, SpatialLatentTensorGate2d
import functools
from utils import mdtp
import time
import yaml
import torch.nn.functional as F
from models.expert_t import LstmAttention

from models.expert_s import Generator
from models.TimesNet import TimesBlock, Model, Model_moe, Model_withoutmoe, Model_moeconv, Model_onetimenet

import torch.distributions as distributions





parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='cuda:4', help='GPU setting')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--window_size', type=int, default=24, help='window size')
parser.add_argument('--pred_size', type=int, default=4, help='pred size')
parser.add_argument('--node_num', type=int, default=231, help='number of node to predict')
parser.add_argument('--in_features', type=int, default=2, help='GCN input dimension')
parser.add_argument('--out_features', type=int, default=16, help='GCN output dimension')
parser.add_argument('--lstm_features', type=int, default=256, help='LSTM hidden feature size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=1000, help='epoch')
parser.add_argument('--gradient_clip', type=int, default=5, help='gradient clip')
parser.add_argument('--pad', type=bool, default=False, help='whether padding with last batch sample')
parser.add_argument('--bike_base_path', type=str, default='./data/bike', help='bike data path')
parser.add_argument('--taxi_base_path', type=str, default='./data/taxi', help='taxi data path')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./best_model.pth', help='save path')
parser.add_argument('--rlsave', type=str, default='./mmsm_model/', help='save path')
parser.add_argument('--smoe_start_epoch', type=int, default=99, help='smoe start epoch')
parser.add_argument('--gpus', type=str, default='4', help='gpu')
parser.add_argument('--log', type=str, default='0.log', help='log name')


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def custom_collate_fn(batch):
    return batch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []

    def store(self, state, action, log_prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()

class MultiArmedBanditEnv:
    def __init__(self, num_arms, batch_size, pred_size, data_type='val'):

        self.num_arms = num_arms
        self.batch_size = batch_size
        self.pred_size = pred_size

        assert data_type in ['train', 'val', 'test'], "data_type 必须为 'train' 或 'test'"

        bikevolume_save_path = os.path.join(args.bike_base_path, f'BV_{data_type}.npy')
        bikeflow_save_path = os.path.join(args.bike_base_path, f'BF_{data_type}.npy')
        taxivolume_save_path = os.path.join(args.taxi_base_path, f'TV_{data_type}.npy')
        taxiflow_save_path = os.path.join(args.taxi_base_path, f'TF_{data_type}.npy')

        bike_train_data = MyDataset_mmsm(bikevolume_save_path, args.window_size, batch_size)
        taxi_train_data = MyDataset_mmsm(taxivolume_save_path, args.window_size, batch_size)
        bike_adj_data = MyDataset_mmsm(bikeflow_save_path, args.window_size, batch_size)
        taxi_adj_data = MyDataset_mmsm(taxiflow_save_path, args.window_size, batch_size)

        self.bike_train_loader = DataLoader(
            dataset=bike_train_data,
            batch_size=None,
            shuffle=False,
            pin_memory=True,
            collate_fn=custom_collate_fn 
        )
        self.taxi_train_loader = DataLoader(
            dataset=taxi_train_data,
            batch_size=None,
            shuffle=False,
            pin_memory=True,
            collate_fn=custom_collate_fn 
        )
        self.bike_adj_loader = DataLoader(
            dataset=bike_adj_data,
            batch_size=None,
            shuffle=False,
            pin_memory=True,
            collate_fn=custom_collate_fn 
        )
        self.taxi_adj_loader = DataLoader(
            dataset=taxi_adj_data,
            batch_size=None,
            shuffle=False,
            pin_memory=True,
            collate_fn=custom_collate_fn 
        )
        # cache
        self.cache = torch.zeros((batch_size, pred_size, pred_size, 231, 4),device=device)
        self.timesnet_cache = torch.zeros((batch_size, pred_size, pred_size, 231 * 4),device=device)

        # out_node
        self.out_node = None

        # loss
        self.loss = mdtp.mae_rlerror

    def _getdata(self):
        try:
            if not hasattr(self, 'bike_adj_loader_iter'):
                self.bike_adj_loader_iter = iter(self.bike_adj_loader)
                self.bike_train_loader_iter = iter(self.bike_train_loader)
                self.taxi_adj_loader_iter = iter(self.taxi_adj_loader)
                self.taxi_train_loader_iter = iter(self.taxi_train_loader)

            bike_in_adj, bike_out_adj = next(self.bike_adj_loader_iter)
            bike_in_node, bike_out_node = next(self.bike_train_loader_iter)
            taxi_in_adj, taxi_out_adj = next(self.taxi_adj_loader_iter)
            taxi_in_node, taxi_out_node = next(self.taxi_train_loader_iter)
            self.out_node = torch.cat((bike_out_node.to(device), taxi_out_node.to(device)), dim=-1)
            return (bike_in_node.to(device), bike_in_adj.to(device), taxi_in_node.to(device), taxi_in_adj.to(device))
        except StopIteration:
            self.bike_adj_loader_iter = iter(self.bike_adj_loader)
            self.bike_train_loader_iter = iter(self.bike_train_loader)
            self.taxi_adj_loader_iter = iter(self.taxi_adj_loader)
            self.taxi_train_loader_iter = iter(self.taxi_train_loader)
            return None
    
    def _getcache(self):
        return self.cache.detach()
    
    def _gettimesnetcache(self):
        return self.timesnet_cache.detach()
    
    def reset(self):
        data = self._getdata()
        self.cache = torch.zeros((self.batch_size, self.pred_size, self.pred_size, 231, 4),device=device)
        cache = self._getcache()
        return data, cache
    
    def _getreward(self, action, newcache):
        cache_loss1 = self.loss(newcache[:, :, 0, :, 0], self.out_node[:, :, 0].unsqueeze(1).expand(-1, 4, -1))      
        cache_loss2 = self.loss(newcache[:, :, 0, :, 2], self.out_node[:, :, 2].unsqueeze(1).expand(-1, 4, -1))      
 
        min_indices1 = torch.argmin(cache_loss1, dim=1)
        min_indices2 = torch.argmin(cache_loss2, dim=1)

        return min_indices1, min_indices2

    
    def _updatecache(self, newcache):
        newcache = torch.roll(newcache, 1, dims=1)
        newcache = torch.roll(newcache, -1, dims=2)
        self.cache = newcache
        return 

    def step(self, new_cache):
        # update cache
        self._updatecache(new_cache)
        cache = self._getcache()

        data = self._getdata()
        return data, cache
    
    def getrightaction(self, action, newcache):
        cache_loss = self.loss(newcache[:, :, 0, :, 0], self.out_node[:, :, 0].unsqueeze(1).expand(-1, 4, -1))      
        rightaction = torch.argmin(cache_loss, dim=1)
        _, sortaction = torch.sort(cache_loss, dim=-1, descending=False)
        return sortaction
    
    def get_acc(self, pred):
        # pred = pred[:, :, 0, :, :].view(231, 4)
        # pred = pred[:, 0, :, :].view(231, 4)
        pred = pred.view(16, 231, 4)
        # pred = pred[0, 0, :, :]
        out_node = self.out_node.view(16, 231, 4)
        bk_start_mask = pred[:, :, 0]!= out_node[:, :, 0]
        bk_end_mask = pred[:, :, 1] != out_node[:, :, 1]
        tx_start_mask = pred[:, :, 2] != out_node[:, :, 2]
        tx_end_mask = pred[:, :, 3] != out_node[:, :, 3]
        
        bike_start_metrics = metric1(pred[:, :, 0], out_node[:, :, 0], bk_start_mask)
        bike_end_metrics = metric1(pred[:, :, 1], out_node[:, :, 1], bk_end_mask)
        taxi_start_metrics = metric1(pred[:, :, 2], out_node[:, :, 2], tx_start_mask)
        taxi_end_metrics = metric1(pred[:, :, 3], out_node[:, :, 3], tx_end_mask)

        return bike_start_metrics, bike_end_metrics, taxi_start_metrics, taxi_end_metrics
    
    def get_acc_train(self, pred):
        # pred = pred[:, :, 0, :, :].view(231, 4)
        # pred = pred[:, 0, :, :].view(self.batch_size, 231, 4)
        # pred = pred[0, 0, :, :]
        out_node = self.out_node.view(self.batch_size, 231, 4)
        bk_start_mask = pred[:, :, 0]!= out_node[:, :, 0]
        bk_end_mask = pred[:,:,1] != out_node[:,:, 1]
        tx_start_mask = pred[:,:, 2] != out_node[:,:, 2]
        tx_end_mask = pred[:,:, 3] != out_node[:,:, 3]
        
        bike_start_metrics = mdtp.rmse(pred[:,:, 0], out_node[:,:, 0], bk_start_mask)
        bike_end_metrics = mdtp.rmse(pred[:,:, 1], out_node[:,:, 1], bk_end_mask)
        taxi_start_metrics = mdtp.rmse(pred[:,:, 2], out_node[:,:, 2], tx_start_mask)
        taxi_end_metrics = mdtp.rmse(pred[:,:, 3], out_node[:,:, 3], tx_end_mask)

        return bike_start_metrics, bike_end_metrics, taxi_start_metrics, taxi_end_metrics

class BanditNet(nn.Module):
    def __init__(self, num_arms, model_list, batch_size, pred_size):
        super(BanditNet, self).__init__()
        self.batch_size = batch_size
        self.pred_size = pred_size
        self.lstm = nn.LSTM(input_size=231, hidden_size=32, num_layers=1, batch_first=True)
        self.weight = nn.Parameter(torch.full((6, 4), 0.25))
        self.decide = nn.Linear(in_features=32, out_features=6)
        self.softmax = nn.Softmax(dim=-1)

        self.hidden_state = None
        self.model_list = model_list

        self.fc = nn.Linear(num_arms, num_arms) 

    def forward(self, data, cache, epsilon):
        
        lstm_input = torch.cat((data[0], data[2]), dim=-1)
        lstm_input = torch.sum(lstm_input, dim=-1)
        lstm_input = lstm_input[:, -1, :].reshape(self. batch_size, -1)
        lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
        decisoin_q = torch.mean(self.decide(lstm_out), dim=0)
        action_probs = self.softmax(decisoin_q)
        dist = distributions.Categorical(action_probs)
        action = dist.sample()

        self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())

        with torch.no_grad():
            bike_start, bike_end, taxi_start, taxi_end = self.model_list[action](data, (action+1)*4)
        new_predict = torch.stack((bike_start, bike_end, taxi_start, taxi_end), dim=-1)
        updated_cache = torch.cat((new_predict.unsqueeze(1), cache[:, -3: :, : ,:]), dim=1)

        with torch.no_grad():
            bike_start, bike_end, taxi_start, taxi_end = self.model_list[-1](data, 24)
        st_predict = torch.stack((bike_start, bike_end, taxi_start, taxi_end), dim=-1)

        return self.weight[action], new_predict[:, 0, :, :], st_predict[:, 0, :, :], decisoin_q, updated_cache, action, dist.log_prob(action)

    def forward_test(self, data, cache):
        t1 = time.time()
        t1 = time.time()
        lstm_input = torch.cat((data[0], data[2]), dim=-1)
        lstm_input = torch.sum(lstm_input, dim=-1)
        lstm_input = lstm_input[:, -1, :].reshape(self. batch_size, -1)
        lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
        
        self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())
        t2 = time.time()
        

        decisoin_q = torch.mean(self.decide(lstm_out), dim=0)

        action_probs = self.softmax(decisoin_q)

        action = torch.argmax(action_probs).item()
        
        t1 = time.time()
        with torch.no_grad():
            bike_start, bike_end, taxi_start, taxi_end = self.model_list[action](data, (action+1)*4)
        new_predict = torch.stack((bike_start, bike_end, taxi_start, taxi_end), dim=-1)
        t2 = time.time()

        updated_cache = torch.cat((new_predict.unsqueeze(1), cache[:, -3: :, : ,:]), dim=1)

        return self.weight[action], new_predict[:, 0, :, :], updated_cache
    
def save_model(model, save_path, episode):
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    model_save_path = os.path.join(save_path, f"mmsm_{episode}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == '__main__':
    set_seed(args.seed)

    base_smoe_config = SpatialMoEConfig(
            in_planes=2,
            out_planes=16,
            num_experts=32,
            gate_block=functools.partial(SpatialLatentTensorGate2d,
                                node_num = 231),
            save_error_signal=True,
            dampen_expert_error=False,
            unweighted=False,
            block_gate_grad=True,
    )
    model_list = nn.ModuleList()

    with open('model_paths.txt', 'r') as f:
        model_paths = [line.strip() for line in f if line.strip()]

    for model_path in model_paths:
        pred_model = Net(
            args.batch_size, 
            args.window_size, 
            args.node_num, 
            args.in_features, 
            args.out_features, 
            args.lstm_features, 
            base_smoe_config, 
            args.pred_size)
        
        pred_model.to(device)
        pred_model.load_state_dict(torch.load(model_path))

        for param in pred_model.parameters():
            param.requires_grad = False

        model_list.append(pred_model)

    # BanditNet
    bandit_Net = BanditNet(
        num_arms=6, 
        model_list=model_list, 
        batch_size=args.batch_size,
        pred_size=args.pred_size)
    bandit_Net.to(device)
    bandit_Net.train()
    optimizer = optim.Adam(bandit_Net.parameters(), lr=0.0001)
    
    # env
    env = MultiArmedBanditEnv(num_arms=4, batch_size=args.batch_size, pred_size=args.pred_size)

    num_episodes = 400
    epsilon_start = 1.0 
    epsilon_end = 0.01   
    epsilon_decay = 0.995  
    buffer = RolloutBuffer()
    for episode in range(num_episodes):
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        data, cache = env.reset()
        total_loss = 0
        t = 0
        step_count = 0
        while data != None:
            # weight, new_predict, st_predict, decisoin_q, new_cache = bandit_Net(data, cache, epsilon)  
            weight, new_predict, st_predict, decisoin_q, new_cache, action, log_prob = bandit_Net(data, cache, epsilon)

            # if random.random() < epsilon: 
            #     decision = torch.randint(0, decisoin_q.size(0), (1,))  
            # else:
            #     decision = torch.argmax(decisoin_q, dim=0)  

            data, cache = env.step(new_cache=new_cache)
            bike_start_metrics, bike_end_metrics, taxi_start_metrics, taxi_end_metrics = env.get_acc_train(new_predict)
            st_bike_start_metrics, st_bike_end_metrics, st_taxi_start_metrics, st_taxi_end_metrics = env.get_acc_train(st_predict)
            
            weight = weight.view(1, 4, 1, 1)
            predict_w = torch.sum(cache[:, :, 0, :, :] * weight, dim=1)
            weight_u = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
            weight_u = weight_u.view(1, 4, 1, 1)
            predict_u = torch.sum(cache[:, :, 0, :, :] * weight_u, dim=1)
            u_bike_start_metrics, u_bike_end_metrics, u_taxi_start_metrics, u_taxi_end_metrics = env.get_acc_train(predict_u)
            w_bike_start_metrics, w_bike_end_metrics, w_taxi_start_metrics, w_taxi_end_metrics = env.get_acc_train(st_predict)
            
            reward_m = (st_bike_start_metrics - bike_start_metrics) + (st_bike_end_metrics - bike_end_metrics) + \
                        (st_taxi_start_metrics - taxi_start_metrics) + (st_taxi_end_metrics - taxi_end_metrics)
            reward_w =  (u_bike_start_metrics - w_bike_start_metrics) + (u_bike_end_metrics - w_bike_end_metrics) + \
                        (u_taxi_start_metrics - w_taxi_start_metrics) + (u_taxi_end_metrics - w_taxi_end_metrics)
            
            reward = reward_m + 0.1 * reward_w

            # optimizer.zero_grad()

            # loss = -torch.log(decisoin_q) * reward

            # loss.backward()  
            # optimizer.step()  
            # total_loss += loss.item()
            buffer.store(None, action, log_prob, reward)

            step_count += 1

            if step_count % 16 == 0: 
                optimizer.zero_grad()
                loss = 0
                rewards = torch.tensor(buffer.rewards, dtype=torch.float32, device=device)
                log_probs = torch.stack(buffer.log_probs)
                loss = -(log_probs * rewards).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                buffer.clear()

        print(f"Loss: {total_loss:.4f}")
        print(f"Episode {episode + 1}/{num_episodes}")
        save_model(bandit_Net, args.rlsave, episode)