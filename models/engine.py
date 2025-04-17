import torch
import torch.nn as nn
import functools

from torch import optim
from models.model import Net
import yaml
from utils import mdtp
from models.smoe_config import SpatialMoEConfig
from models.gate import SpatialLinearGate2d, SpatialLatentTensorGate2d
from models.loss import routing_classification_loss_by_error
from utils.mdtp import metric1, getCrossloss
from collections import deque
import time


config = yaml.safe_load(open('config.yml'))


def adjust_learning_rate(optimizer, lr, wd):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd


class Trainer:
    def __init__(self, batch_size, window_size, node_num, in_features, out_features, lstm_features, device, lrate,
                 wdecay, clip, smoe_start_epoch, pred_size):
        # self.model = Net(batch_size, window_size, node_num, in_features, out_features, lstm_features)
        self.base_smoe_config = SpatialMoEConfig(
            in_planes=2,
            out_planes=150,
            num_experts=231,
            gate_block=functools.partial(SpatialLatentTensorGate2d,
                                node_num = 231),
            save_error_signal=True,
            dampen_expert_error=False,
            unweighted=False,
            block_gate_grad=True,
            routing_error_quantile=0.7,
            pred_size=pred_size,
            windows_size=window_size
        )
        
        self.model = Net(batch_size, window_size, node_num, in_features, out_features, lstm_features, smoe_config=self.base_smoe_config, pred_size=pred_size)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total parameters: {total_params}')
        

        self.model.to(device)
        nodesample_params = [param for name, param in self.model.named_parameters() if 'nodesample' in name]
        other_params = [param for name, param in self.model.named_parameters() if 'nodesample' not in name]

        self.optimizer = optim.Adam([
            {'params': nodesample_params, 'lr': 0.001}, 
            {'params': other_params, 'lr': lrate, 'betas': (0.9, 0.99), 'weight_decay': wdecay}
        ], lr=lrate, weight_decay=wdecay, betas=(0.9, 0.99))
        self.loss = mdtp.mae_weight

        self.maeloss = mdtp.mae_weight
        self.rmseloss = mdtp.rmse_weight
        self.mapeloss = mdtp.mape_weight
        self.rmse_per_node = mdtp.rmse_per_node

        self.batch_size = batch_size
        self.window_size = window_size
        self.node_num = node_num
        self.clip = clip
        self.smoe_start_epoch = smoe_start_epoch
        self.smoe_start = False

        self.gcn1_grad = 0
        self.gcn2_grad = 0

    def train(self, input, real, lr, wd, iter):
        self.model.train()
        self.optimizer.zero_grad()

        bike_start_predict, bike_end_predict, taxi_start_predict, taxi_end_predict = self.model(input, 24)

        bike_start_predict.retain_grad()
        bike_end_predict.retain_grad()
        taxi_start_predict.retain_grad()
        taxi_end_predict.retain_grad()

        bike_real, taxi_real = real[0], real[1]

        mask = torch.ones_like(bike_real[0], dtype=torch.bool)
        start_mask = mask
        end_mask = mask

        bike_start_loss = self.loss(bike_start_predict, bike_real[0], start_mask)
        bike_end_loss = self.loss(bike_end_predict, bike_real[1], end_mask)
        bike_start_loss_rmse = self.rmseloss(bike_start_predict, bike_real[0], start_mask)
        bike_end_loss_rmse = self.rmseloss(bike_end_predict, bike_real[1], end_mask)
        bike_start_loss_mape = self.mapeloss(bike_start_predict, bike_real[0], start_mask)
        bike_end_loss_mape = self.mapeloss(bike_end_predict, bike_real[1], end_mask)

        taxi_start_loss = self.loss(taxi_start_predict, taxi_real[0], start_mask)
        taxi_end_loss = self.loss(taxi_end_predict, taxi_real[1], end_mask)
        taxi_start_loss_rmse = self.rmseloss(taxi_start_predict, taxi_real[0], start_mask)
        taxi_end_loss_rmse = self.rmseloss(taxi_end_predict, taxi_real[1], end_mask)
        taxi_start_loss_mape = self.mapeloss(taxi_start_predict, taxi_real[0], start_mask)
        taxi_end_loss_mape = self.mapeloss(taxi_end_predict, taxi_real[1], end_mask)
        
        loss =  0.3 * (taxi_start_loss_rmse + taxi_end_loss_rmse) + (bike_start_loss_rmse + bike_end_loss_rmse) * 0.7 
        loss.backward(retain_graph=True) 
        
        '''maskloss'''

        grad1 =  torch.abs(bike_start_predict.grad) + torch.abs(bike_end_predict.grad)
        grad2 = torch.abs(taxi_start_predict.grad) + torch.abs(taxi_end_predict.grad)

        maskloss1 = mdtp.getMaskloss(self.model.nodesample.conf1, grad1)
        maskloss2 = mdtp.getMaskloss(self.model.nodesample.conf2, grad2)
        maskloss = maskloss1 + maskloss2
        maskloss.backward()
        
        '''rc_loss'''
        if self.smoe_start:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            rc_loss = routing_classification_loss_by_error(self.model, scaler, self.base_smoe_config.routing_error_quantile)
            if rc_loss: 
                rc_loss_avg = sum(rc_loss)
                rc_loss_avg = rc_loss_avg
                rc_loss_avg.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        adjust_learning_rate(self.optimizer, lr, wd)
        mae = (bike_start_loss.item(), bike_end_loss.item(), taxi_start_loss.item(), taxi_end_loss.item())
        rmse = (bike_start_loss_rmse.item(), bike_end_loss_rmse.item(), taxi_start_loss_rmse.item(), taxi_end_loss_rmse.item())
        mape = (bike_start_loss_mape.item(), bike_end_loss_mape.item(), taxi_start_loss_mape.item(), taxi_end_loss_mape.item())
    
        return mae, rmse, mape

    def val(self, input, real, epoch):
        self.model.eval()
        t1 = time.time()
        bike_start_predict, bike_end_predict, taxi_start_predict, taxi_end_predict = self.model(input, 24)
        t2 = time.time()
        bike_real, taxi_real = real[0], real[1]

        mask = torch.ones_like(bike_real[0], dtype=torch.bool)
        start_mask = mask
        end_mask = mask

        bike_start_loss = self.loss(bike_start_predict, bike_real[0], start_mask)
        bike_end_loss = self.loss(bike_end_predict, bike_real[1], end_mask)
        bike_start_loss_rmse = self.rmseloss(bike_start_predict, bike_real[0], start_mask)
        bike_end_loss_rmse = self.rmseloss(bike_end_predict, bike_real[1], end_mask)
        bike_start_loss_mape = self.mapeloss(bike_start_predict, bike_real[0], start_mask)
        bike_end_loss_mape = self.mapeloss(bike_end_predict, bike_real[1], end_mask)

        taxi_start_loss = self.loss(taxi_start_predict, taxi_real[0], start_mask)
        taxi_end_loss = self.loss(taxi_end_predict, taxi_real[1], end_mask)
        taxi_start_loss_rmse = self.rmseloss(taxi_start_predict, taxi_real[0], start_mask)
        taxi_end_loss_rmse = self.rmseloss(taxi_end_predict, taxi_real[1], end_mask)
        taxi_start_loss_mape = self.mapeloss(taxi_start_predict, taxi_real[0], start_mask)
        taxi_end_loss_mape = self.mapeloss(taxi_end_predict, taxi_real[1], end_mask)

        mae = (bike_start_loss.item(), bike_end_loss.item(), taxi_start_loss.item(), taxi_end_loss.item())
        rmse = (bike_start_loss_rmse.item(), bike_end_loss_rmse.item(), taxi_start_loss_rmse.item(), taxi_end_loss_rmse.item())
        mape = (bike_start_loss_mape.item(), bike_end_loss_mape.item(), taxi_start_loss_mape.item(), taxi_end_loss_mape.item())
        
        return (mae, rmse, mape), (t2-t1)

