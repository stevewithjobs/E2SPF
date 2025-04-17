import torch
from torch.utils.data import Dataset
import numpy as np
import random
import os
import scipy.sparse as sp
from scipy.sparse import linalg
import torch.nn as nn


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MyDataset(Dataset):
    def __init__(self, path, window_size, batch_size, pad_with_last_sample):
        super(MyDataset, self).__init__()
        original_data = np.load(path)
        self.batch_size = batch_size
        if pad_with_last_sample:
            num_padding = (batch_size - (len(original_data) % batch_size)) % batch_size + window_size
            x_padding = np.repeat(original_data[-1:], num_padding, axis=0)
            self.data = torch.from_numpy(np.concatenate([original_data, x_padding], axis=0)).float()
        else:
            self.data = torch.from_numpy(original_data).float()
        self.window_size = window_size
        self.num = self.data.size(0) - window_size

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item: item + self.window_size], self.data[item + self.window_size]
    
class MyDataset_nstp(Dataset):
    def __init__(self, path, window_size, batch_size, pad_with_last_sample, pred_size):
        super(MyDataset_nstp, self).__init__()
        original_data = np.load(path)
        self.batch_size = batch_size
        if pad_with_last_sample:
            num_padding = (batch_size - (len(original_data) % batch_size)) % batch_size + window_size   
            x_padding = np.repeat(original_data[-1:], num_padding, axis=0)
            self.data = torch.from_numpy(np.concatenate([original_data, x_padding], axis=0)).float()
        else:
            self.data = torch.from_numpy(original_data).float()
        self.window_size = window_size
        self.num = self.data.size(0) - window_size 
        self.pred_size = pred_size

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        in_start = item
        in_end = in_start + self.window_size
        out_start = in_end
        out_end = out_start + self.pred_size
        return self.data[in_start:in_end], self.data[out_start:out_end]
    
class MyDataset_nstponline(Dataset):
    def __init__(self, path, window_size, batch_size, pred_size, data_fraction=1.0):
        super(MyDataset_nstponline, self).__init__()
    
        self.data = np.load(path)
        self.window_size = window_size
        self.batch_size = batch_size
        self.total_length = self.data.shape[0]

        total_data_length = self.data.shape[0] 
        self.total_length = int(total_data_length * data_fraction) 

        self.pred_size = pred_size


        self.subset_size = (self.total_length - self.window_size - self.pred_size) // self.batch_size
        self.total_length = self.subset_size * self.batch_size  

    def __len__(self):
        return self.subset_size  

    def __getitem__(self, index):
        batch_input = []
        batch_target = []

        for i in range(self.batch_size):
          
            start_idx = i * self.subset_size + index     
            end_idx = start_idx + self.window_size
            batch_input.append(torch.tensor(self.data[start_idx:end_idx], dtype=torch.float))  

   
            pred_start_idx = end_idx 
            pred_end_idx = pred_start_idx + self.pred_size
            batch_target.append(torch.tensor(self.data[pred_start_idx:pred_end_idx], dtype=torch.float)) 

      
        return torch.stack(batch_input), torch.stack(batch_target)
    
class MyDataset_rl(Dataset):
    def __init__(self, path, window_size, batch_size):
        super(MyDataset_rl, self).__init__()

        self.data = np.load(path)
        self.window_size = window_size
        self.batch_size = batch_size
        self.total_length = self.data.shape[0]

        self.subset_size = (self.total_length - self.window_size) // self.batch_size
        self.total_length = self.subset_size * self.batch_size  

    def __len__(self):
        return 1

    def __getitem__(self, index):
        batch_input = []
        for i in range(self.batch_size):
            start_idx = i * self.subset_size + index     
            end_idx = start_idx + self.window_size + self.subset_size
            batch_input.append(torch.tensor(self.data[start_idx:end_idx], dtype=torch.float))  

        return torch.stack(batch_input)
    
class MyDataset_mmsm(Dataset):
    def __init__(self, path, window_size, batch_size):
        super(MyDataset_mmsm, self).__init__()

        self.data = np.load(path)
        self.window_size = window_size
        self.batch_size = batch_size
        self.total_length = self.data.shape[0]

        self.subset_size = (self.total_length - self.window_size) // self.batch_size
        self.total_length = self.subset_size * self.batch_size  

    def __len__(self):
        return self.subset_size

    def __getitem__(self, index):
        batch_input = []
        batch_output = []
        for i in range(self.batch_size):

            start_idx = i * self.subset_size + index     
            end_idx = start_idx + self.window_size
            batch_input.append(torch.tensor(self.data[start_idx:end_idx], dtype=torch.float))  
            batch_output.append(torch.tensor(self.data[end_idx], dtype=torch.float)) 

        return torch.stack(batch_input), torch.stack(batch_output)




# def masked_mse(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = (preds - labels) ** 2
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)


# def masked_rmse(preds, labels, null_val=np.nan):
#     return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


# def masked_mae(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds - labels)
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)


# def masked_mape(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds - labels) / labels
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)


# def mae(preds, labels, threshold):
#     mask = labels > threshold
#     if torch.sum(mask) != 0:
#         avg_mae = torch.mean(torch.abs(labels[mask] - preds[mask]))
#         return avg_mae
#     else:
#         return torch.tensor(0)

# def rmse(preds, labels, threshold):
#     mask = labels > threshold
#     if torch.sum(mask) != 0:
#         avg_rmse = torch.sqrt(torch.mean((labels[mask] - preds[mask]) ** 2))
#         return avg_rmse
#     else:
#         return torch.tensor(0)

# def mape(preds, labels, threshold):
#     mask = labels > threshold
#     if torch.sum(mask) != 0:
#         avg_mape = torch.mean(torch.abs(labels[mask] - preds[mask]) / labels[mask])
#         return avg_mape
#     else:
#         return torch.tensor(0)



def mae(preds, labels, mask):
    if torch.sum(mask) != 0:
        avg_mae = torch.mean(torch.abs(labels[mask] - preds[mask]))
        return avg_mae
    else:
        return torch.tensor(0)

def rmse(preds, labels, mask):
    if torch.sum(mask) != 0:
        avg_rmse = torch.sqrt(torch.mean((labels[mask] - preds[mask]) ** 2))
        return avg_rmse
    else:
        return torch.tensor(0)

def mape(preds, labels, mask):
    if torch.sum(mask) != 0:
        zero_positions = (labels == 0)
        mask = torch.logical_and(mask, ~zero_positions)
        mask = ~zero_positions
        avg_mape = torch.mean(torch.abs(labels[mask] - preds[mask]) / labels[mask])
        return avg_mape
    else:
        return torch.tensor(0)
    
def mae_weight(preds, labels, mask, a=1, r=1):
    assert preds.shape == labels.shape == mask.shape
    num_steps = preds.size(1)
    weights = a * torch.pow(r, torch.arange(num_steps, dtype=torch.float32)).to(preds.device)
    # weights = weights.view(1, num_steps, 1)
    
    if torch.sum(mask) != 0:
        abs_diff = torch.abs(labels - preds)
        # weighted_diff = abs_diff * weights * mask
        # avg_mae = torch.sum(weighted_diff) / torch.sum(mask)
        mae_per_step = torch.sum(abs_diff * mask, dim=(0, 2)) / torch.sum(mask, dim=(0, 2))  
        weighted_mae = mae_per_step * weights  
        avg_mae = torch.sum(weighted_mae) / torch.sum(weights)
        return avg_mae
    else:
        return torch.tensor(0.0, device=preds.device)


def rmse_weight(preds, labels, mask, a=1, r=1):

    assert preds.shape == labels.shape == mask.shape
    num_steps = preds.size(1)
    weights = a * torch.pow(r, torch.arange(num_steps, dtype=torch.float32)).to(preds.device)
    # weights = weights.view(1, num_steps, 1)
    
    if torch.sum(mask) != 0:
        squared_diff = (labels - preds) ** 2
        # weighted_squared_diff = squared_diff * weights * mask
        # avg_rmse = torch.sqrt(torch.sum(weighted_squared_diff) / torch.sum(mask))
        rmse_per_step = torch.sqrt(torch.sum(squared_diff * mask, dim=(0, 2)) / torch.sum(mask, dim=(0, 2)))  
        weighted_rmse = rmse_per_step * weights 
        avg_rmse = torch.sum(weighted_rmse) / torch.sum(weights)
        return avg_rmse
    else:
        return torch.tensor(0.0, device=preds.device)


def mape_weight(preds, labels, mask, a=1, r=1):

    assert preds.shape == labels.shape == mask.shape
    num_steps = preds.size(1)
    weights = a * torch.pow(r, torch.arange(num_steps, dtype=torch.float32)).to(preds.device)
    # weights = weights.view(1, num_steps, 1)
    
    if torch.sum(mask) != 0:
        zero_positions = (labels == 0)
        mask = torch.logical_and(mask, ~zero_positions)  
        abs_diff = torch.abs(labels - preds)
        relative_diff = abs_diff / torch.clamp(labels, min=1e-8)  
        mape_per_step = torch.sum(relative_diff * mask, dim=(0, 2)) / torch.sum(mask, dim=(0, 2)) 
        weighted_mape = mape_per_step * weights  
        avg_mape = torch.sum(weighted_mape) / torch.sum(weights)
        return avg_mape
    else:
        return torch.tensor(0.0, device=preds.device)

def mae_rlerror(preds, labels):
    
    assert preds.shape == labels.shape
    
    abs_diff = torch.abs(labels - preds)
    
    mae = torch.mean(abs_diff, dim = 2)

    
    return mae

def rmse_rlerror(preds, labels):

    assert preds.shape == labels.shape
    
    squared_diff = (labels - preds) ** 2
    
    mse = torch.mean(squared_diff, dim=1)
    mse = torch.sum(mse, dim=1)
    
    rmse = torch.sqrt(mse)
    
    return rmse

def rmse_rlerror(preds, labels):

    assert preds.shape == labels.shape
    
    squared_diff = (labels - preds) ** 2
    
    mse_per_sample = torch.mean(squared_diff, dim=list(range(1, squared_diff.dim())))
    
    rmse_per_sample = torch.sqrt(mse_per_sample)
    
    return rmse_per_sample

def metric1(pred, real, threshold):
    m = mae(pred, real, threshold).item()
    r = rmse(pred, real, threshold).item()
    p = mape(pred, real, threshold).item()
    return m, r, p

def getCrossloss(tensor):

    grad = tensor.grad

    num_top_regions = int(0.3 * len(grad)) 
    top_indices = torch.topk(grad.abs(), num_top_regions).indices

    mask = torch.zeros_like(tensor)
    mask[top_indices] = 1

    loss = ((tensor * mask) ** 2).sum()

    return loss
def rmse_per_node(preds, labels, mask, a=1, r=0.5):

    assert preds.shape == labels.shape == mask.shape
    num_steps = preds.size(1)
    num_nodes = preds.size(2)
    
    weights = a * torch.pow(r, torch.arange(num_steps, dtype=torch.float32)).to(preds.device) 
    weights = weights.view(num_steps, 1) 

    if torch.sum(mask) != 0:

        squared_diff = (labels - preds) ** 2  # (batch_size, num_steps, num_nodes)
        
        rmse_per_step = torch.sqrt(torch.sum(squared_diff * mask, dim=0) / torch.sum(mask, dim=0))  # (num_steps, num_nodes)
        weighted_rmse = rmse_per_step * weights  # (num_steps, num_nodes)
        
        avg_rmse_per_node = torch.sum(weighted_rmse, dim=0) / torch.sum(weights)
        return avg_rmse_per_node  
    else:
        return torch.zeros(num_nodes, device=preds.device)
    
def getMaskloss(conf, grad):
    grad = grad.sum(dim=(0, 1))
    q = grad.quantile(1/2, dim=-1, keepdim=True)
    label = (grad >= q).float() 

    criterion = torch.nn.BCELoss()
    loss = criterion(conf, label)
    
    return loss

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def metric1_per_node(pred, real, threshold):
    # Initialize lists to store metrics for each node
    mae_per_node = []
    rmse_per_node = []
    mape_per_node = []

    # Iterate over each node in the tensor (assuming the second dimension represents nodes)
    for node_index in range(pred.shape[1]):
        # Extract the predictions and real values for the current node
        pred_node = pred[:, node_index]
        real_node = real[:, node_index]

        # Compute metrics for the current node
        mae_node = mae(pred_node, real_node, threshold)
        rmse_node = rmse(pred_node, real_node, threshold)
        mape_node = mape(pred_node, real_node, threshold)

        # Store the metrics
        mae_per_node.append(mae_node.item())
        rmse_per_node.append(rmse_node.item())
        mape_per_node.append(mape_node.item())

    # Convert lists to tensors for easier handling
    mae_per_node = torch.tensor(mae_per_node)
    rmse_per_node = torch.tensor(rmse_per_node)
    mape_per_node = torch.tensor(mape_per_node)

    return mae_per_node, rmse_per_node, mape_per_node