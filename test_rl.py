import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse
from collections import deque
from utils.mdtp import MyDataset_rl, set_seed
from torch.utils.data.dataloader import DataLoader
from models.model import Net
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLinearGate2d, SpatialLatentTensorGate2d
import functools
from utils import mdtp
import time
import yaml
from train_rl import BanditNet, MultiArmedBanditEnv 



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
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
parser.add_argument('--rlsave', type=str, default='./mmsm_model/mmsm_0.pth', help='save path')
parser.add_argument('--smoe_start_epoch', type=int, default=99, help='smoe start epoch')
parser.add_argument('--gpus', type=str, default='4', help='gpu')
parser.add_argument('--log', type=str, default='0.log', help='log name')


args = parser.parse_args()
config = yaml.safe_load(open('config.yml'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    
    bike_start_loss = []
    bike_end_loss = []
    bike_start_rmse = []
    bike_end_rmse = []
    bike_start_mape = []
    bike_end_mape = []
    taxi_start_loss = []
    taxi_end_loss = []
    taxi_start_rmse = []
    taxi_end_rmse = []
    taxi_start_mape = []
    taxi_end_mape = []

    infer_time = 0
    action_time = 0
    tmp = 0
    modeltime = 0
    t = 0
    action_list = []
    right_action_list = []

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
        num_arms=4, 
        model_list=model_list, 
        batch_size=args.batch_size,
        pred_size=args.pred_size)
    bandit_Net.load_state_dict(torch.load(args.rlsave))
    bandit_Net.eval()
    bandit_Net.to(device)

    # env
    env = MultiArmedBanditEnv(num_arms=4, batch_size=args.batch_size, pred_size=args.pred_size, data_type='test')
    
    data, cache = env.reset()
    t = 0
    tt = 0
    while data != None:
        t1 = time.time()
        with torch.no_grad():
            weight, new_predict, new_cache = bandit_Net.forward_test(data, cache)
        t2 = time.time()
        # action = torch.argmax(q_values1, dim=1)
        if t > 1:
            tt += t2 - t1  

        # if t > 4:
        #     weights = torch.tensor([0.4, 0.3, 0.2, 0.1], device=device)

        #     weights = weights.view(1, 4, 1, 1)

        #     # final_predict = new_cache[:, :, 0, :, :] * weights
        #     final_predict = new_cache[:, 0, 0, :, :]

        #     final_predict = torch.sum(final_predict, dim=1)
        # else:
        #     final_predict = new_predict
        weight = weight.view(1, 4, 1, 1)
        final_predict = new_cache[:, :, 0, :, :] * weight
        final_predict = torch.sum(final_predict, dim=1)
        
        bike_start_metrics, bike_end_metrics, taxi_start_metrics, taxi_end_metrics = env.get_acc(final_predict)
        data, cache = env.step(new_cache=new_cache)

        bike_start_loss.append(bike_start_metrics[0])
        bike_end_loss.append(bike_end_metrics[0])
        bike_start_rmse.append(bike_start_metrics[1])
        bike_end_rmse.append(bike_end_metrics[1])
        bike_start_mape.append(bike_start_metrics[2])
        bike_end_mape.append(bike_end_metrics[2])

        taxi_start_loss.append(taxi_start_metrics[0])
        taxi_end_loss.append(taxi_end_metrics[0])
        taxi_start_rmse.append(taxi_start_metrics[1])
        taxi_end_rmse.append(taxi_end_metrics[1])
        taxi_start_mape.append(taxi_start_metrics[2])
        taxi_end_mape.append(taxi_end_metrics[2])
        log = 'Iter: {:02d}\nTest Bike Start MAE: {:.4f}, Test Bike End MAE: {:.4f}, ' \
                'Test Taxi Start MAE: {:.4f}, Test Taxi End MAE: {:.4f}, \n' \
                'Test Bike Start RMSE: {:.4f}, Test Bike End RMSE: {:.4f}, ' \
                'Test Taxi Start RMSE: {:.4f}, Test Taxi End RMSE: {:.4f}, \n' \
                'Test Bike Start MAPE: {:.4f}, Test Bike End MAPE: {:.4f}, ' \
                'Test Taxi Start MAPE: {:.4f}, Test Taxi End MAPE: {:.4f}'
        print(
            log.format(t, bike_start_metrics[0]*config['bike_volume_max'], bike_end_metrics[0]*config['bike_volume_max'], taxi_start_metrics[0]*config['taxi_volume_max'], taxi_end_metrics[0]*config['taxi_volume_max'],
                        bike_start_metrics[1]*config['bike_volume_max'], bike_end_metrics[1]*config['bike_volume_max'], taxi_start_metrics[1]*config['taxi_volume_max'], taxi_end_metrics[1]*config['taxi_volume_max'],
                        bike_start_metrics[2]*100, bike_end_metrics[2]*100, taxi_start_metrics[2]*100, taxi_end_metrics[2]*100, ))

        t += 1

    log1 = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'

    print(log1.format((np.mean(bike_start_loss))*config['bike_volume_max'], (np.mean(bike_end_loss))*config['bike_volume_max'],
                        (np.mean(taxi_start_loss))*config['taxi_volume_max'], (np.mean(taxi_end_loss))*config['taxi_volume_max'],
                        (np.mean(bike_start_rmse))*config['bike_volume_max'], (np.mean(bike_end_rmse))*config['bike_volume_max'],
                        (np.mean(taxi_start_rmse))*config['taxi_volume_max'], (np.mean(taxi_end_rmse))*config['taxi_volume_max'],
                        (np.mean(bike_start_mape))*100, (np.mean(bike_end_mape))*100,
                        (np.mean(taxi_start_mape))*100, (np.mean(taxi_end_mape))*100))
    print(tt)
