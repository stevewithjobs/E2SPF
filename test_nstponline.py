import json
import time
import os
import yaml
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader
from utils.mdtp import MyDataset, metric1, MyDataset_nstp, MyDataset_nstponline
from models.model import Net
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLinearGate2d, SpatialLatentTensorGate2d
import torch
import functools
import queue



# these parameter settings should be consistent with train.py
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='GPU setting')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--window_size', type=int, default=24, help='window size')
parser.add_argument('--pred_size', type=int, default=4, help='pred size')
parser.add_argument('--node_num', type=int, default=231, help='number of node to predict')
parser.add_argument('--in_features', type=int, default=2, help='GCN input dimension')
parser.add_argument('--out_features', type=int, default=16, help='GCN output dimension')
parser.add_argument('--lstm_features', type=int, default=256, help='LSTM hidden feature size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=20, help='epoch')
parser.add_argument('--gradient_clip', type=int, default=5, help='gradient clip')
parser.add_argument('--pad', type=bool, default=False, help='whether padding with last batch sample')
parser.add_argument('--bike_base_path', type=str, default='./data/bike', help='bike data path')
parser.add_argument('--taxi_base_path', type=str, default='./data/taxi', help='taxi data path')
parser.add_argument('--seed', type=int, default=99, help='random seed')


parser.add_argument('--save', type=str, default='./best_model.pth', help='save path')


args = parser.parse_args()
config = yaml.safe_load(open('config.yml'))

def custom_collate_fn(batch):
    return batch

def main():
    device = torch.device(args.device)

    bikevolume_test_save_path = os.path.join(args.bike_base_path, 'BV_test.npy')
    bikeflow_test_save_path = os.path.join(args.bike_base_path, 'BF_test.npy')
    taxivolume_test_save_path = os.path.join(args.taxi_base_path, 'TV_test.npy')
    taxiflow_test_save_path = os.path.join(args.taxi_base_path, 'TF_test.npy')

    bike_test_data = MyDataset_nstponline(bikevolume_test_save_path, args.window_size, args.batch_size, args.pred_size)
    taxi_test_data = MyDataset_nstponline(taxivolume_test_save_path, args.window_size, args.batch_size, args.pred_size)
    bike_adj_data = MyDataset_nstponline(bikeflow_test_save_path, args.window_size, args.batch_size, args.pred_size)
    taxi_adj_data = MyDataset_nstponline(taxiflow_test_save_path, args.window_size, args.batch_size, args.pred_size)

    bike_test_loader = DataLoader(
        dataset=bike_test_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )
    taxi_test_loader = DataLoader(
        dataset=taxi_test_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn ,
        num_workers=0,
    )
    bike_adj_loader = DataLoader(
        dataset=bike_adj_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )
    taxi_adj_loader = DataLoader(
        dataset=taxi_adj_data,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )

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
    model = Net(args.batch_size, args.window_size, args.node_num, args.in_features, args.out_features, args.lstm_features, base_smoe_config, args.pred_size)

    model.to(device)
    model.load_state_dict(torch.load(args.save),strict=False)
    model.eval()
    print('model load successfully!')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')

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
    

    

    bike_start_que = queue.Queue()
    bike_end_que = queue.Queue()
    taxi_start_que = queue.Queue()
    taxi_end_que = queue.Queue()
    pass_flag = False
    count = 0 
    tmp = 0
    total_time = 0
    # torch.set_num_threads(1)
    preds = []
    trues = []
    for iter, (bike_adj, bike_node, taxi_adj, taxi_node) in enumerate(zip(bike_adj_loader, bike_test_loader,
                                                                              taxi_adj_loader, taxi_test_loader)):
         
        # if iter % 5 != 0:
        #     continue
        # if bike_start_que.empty() and pass_flag:
        #     print(1)
        bike_in_shots, bike_out_shots = bike_node
        bike_adj = bike_adj[0]
        bike_out_shots = bike_out_shots.permute(3, 0, 1, 2)
        taxi_in_shots, taxi_out_shots = taxi_node
        taxi_adj = taxi_adj[0]
        taxi_out_shots = taxi_out_shots.permute(3, 0, 1, 2)
        bike_in_shots, bike_out_shots, bike_adj = bike_in_shots.to(device), bike_out_shots.to(device), bike_adj.to(
            device)
        taxi_in_shots, taxi_out_shots, taxi_adj = taxi_in_shots.to(device), taxi_out_shots.to(device), taxi_adj.to(
            device)
        test_x = (bike_in_shots, bike_adj, taxi_in_shots, taxi_adj)

        with torch.no_grad():
            t1 = time.time()
            bike_start, bike_end, taxi_start, taxi_end = model(test_x, 24)
            t2 = time.time()
            if iter > 1:
                total_time += t2 - t1
        
        want = 0    
        bk_start_mask = bike_out_shots[0][:, want, :] != bike_start[:, want, :]
        bk_end_mask = bike_out_shots[1][:, want, :] != bike_end[:, want, :]
        tx_start_mask = taxi_out_shots[0][:, want, :] != taxi_start[:, want, :]
        tx_end_mask = taxi_out_shots[1][:, want, :] != taxi_end[:, want, :]

        bike_start_metrics = metric1(bike_start[:, want, :], bike_out_shots[0][:, want, :], bk_start_mask)
        bike_end_metrics = metric1(bike_end[:, want, :], bike_out_shots[1][:, want, :], bk_end_mask)
        taxi_start_metrics = metric1(taxi_start[:, want, :], taxi_out_shots[0][:, want, :], tx_start_mask)
        taxi_end_metrics = metric1(taxi_end[:, want, :], taxi_out_shots[1][:, want, :], tx_end_mask)


        pred = torch.stack((taxi_start, taxi_end, bike_start, bike_end), dim=-1)
        true = torch.stack((taxi_out_shots[0], taxi_out_shots[1], bike_out_shots[0], bike_out_shots[1]), dim=-1)
        preds.append(pred)
        trues.append(true)


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
            log.format(iter, bike_start_metrics[0]*config['bike_volume_max'], bike_end_metrics[0]*config['bike_volume_max'], taxi_start_metrics[0]*config['taxi_volume_max'], taxi_end_metrics[0]*config['taxi_volume_max'],
                       bike_start_metrics[1]*config['bike_volume_max'], bike_end_metrics[1]*config['bike_volume_max'], taxi_start_metrics[1]*config['taxi_volume_max'], taxi_end_metrics[1]*config['taxi_volume_max'],
                       bike_start_metrics[2]*100, bike_end_metrics[2]*100, taxi_start_metrics[2]*100, taxi_end_metrics[2]*100, ))
    print("-----------------------------------------------------------------")

    log1 = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'

    print(log1.format(
        (np.mean(taxi_start_loss)) * config['taxi_volume_max'], 
        (np.mean(taxi_start_rmse)) * config['taxi_volume_max'], 
        (np.mean(taxi_start_mape)) * 100,
        
        (np.mean(taxi_end_loss)) * config['taxi_volume_max'], 
        (np.mean(taxi_end_rmse)) * config['taxi_volume_max'], 
        (np.mean(taxi_end_mape)) * 100,

        (np.mean(bike_start_loss)) * config['bike_volume_max'], 
        (np.mean(bike_start_rmse)) * config['bike_volume_max'], 
        (np.mean(bike_start_mape)) * 100,

        (np.mean(bike_end_loss)) * config['bike_volume_max'], 
        (np.mean(bike_end_rmse)) * config['bike_volume_max'], 
        (np.mean(bike_end_mape)) * 100
    ))
    
    print("inference time: ", total_time)
    print(count)

    preds = torch.cat(preds, dim=0)  
    trues = torch.cat(trues, dim=0) 

if __name__ == "__main__":
    main()
