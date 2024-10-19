#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utilities3 import *
import sys
from timeit import default_timer
from Adam import Adam
from loguru import logger
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Import required wavelet packages
try:
    import ptwt, pywt
    from ptwt.conv_transform_3 import wavedec3, waverec3
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    from pytorch_wavelets import DWT, IDWT
except ImportError:
    print('Wavelet convolution requires <Pytorch Wavelets>, <PyWavelets>, <Pytorch Wavelet Toolbox> \n \
                    For Pytorch Wavelet Toolbox: $ pip install ptwt \n \
                    For PyWavelets: $ conda install pywavelets \n \
                    For Pytorch Wavelets: $ git clone https://github.com/fbcotter/pytorch_wavelets \n \
                                          $ cd pytorch_wavelets \n \
                                          $ pip install .')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--var", required=True)
    parser.add_argument("-p", "--process", required=True)
    parser.add_argument("-r", "--resolution", required=True)
    parser.add_argument("-n", "--normalization", default="False")
    args = parser.parse_args()

    print(args.var)
    print(args.process)

    var = args.var
    process = args.process
    res = int(args.resolution)
    norm_flag = args.normalization

    torch.manual_seed(0)
    np.random.seed(0)

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d-%H_%M_%S")
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{level}</cyan> | {message}"
    logger.add(sys.stdout, format=fmt)

casename = 'sameinit'
ntrain = 4000
ntest = 2000
T_in = 10
T = 1
learning_rate = 0.0005  
scheduler_step = 100
scheduler_gamma = 0.1
epochs = 100 
batch_size = 16  
width = 32 
step = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define wavelet parameters
level = 1
wavelet = 'db4'
mode = 'symmetric'
size = [res, res]

# Define WaveConv2d class
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, mode='symmetric'):
        super(WaveConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2d accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2d accepts size of 2D signal as a list')
        self.wavelet = wavelet
        self.mode = mode
        dummy_data = torch.randn(1, 1, *self.size)
        dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet)
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]

        # Parameter initialization
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()  
            if x.shape[-1] > self.size[-1]:
                factor = int(np.log2(x.shape[-1] // self.size[-1]))
                dwt = DWT(J=self.level + factor, mode=self.mode, wave=self.wavelet).to(x.device)
                x_ft, x_coeff = dwt(x)
            elif x.shape[-1] < self.size[-1]:
                factor = int(np.log2(self.size[-1] // x.shape[-1]))
                dwt = DWT(J=self.level - factor, mode=self.mode, wave=self.wavelet).to(x.device)
                x_ft, x_coeff = dwt(x)
            else:
                dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
                x_ft, x_coeff = dwt(x)

            out_ft = torch.zeros_like(x_ft, device=x.device)
            out_coeff = [torch.zeros_like(coeffs, device=x.device) for coeffs in x_coeff]

            # Multiply the final approximate Wavelet modes
            out_ft = self.mul2d(x_ft, self.weights1)
            # Multiply the final detailed wavelet coefficients
            out_coeff[-1][:, :, 0, :, :] = self.mul2d(x_coeff[-1][:, :, 0, :, :].clone(), self.weights2)
            out_coeff[-1][:, :, 1, :, :] = self.mul2d(x_coeff[-1][:, :, 1, :, :].clone(), self.weights3)
            out_coeff[-1][:, :, 2, :, :] = self.mul2d(x_coeff[-1][:, :, 2, :, :].clone(), self.weights4)

            # Return to physical space
            idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
            x = idwt((out_ft, out_coeff))
        return x

# Define WNO2d class
class WNO2d(nn.Module):
    def __init__(self, level, size, wavelet, mode, width):
        super(WNO2d, self).__init__()

        self.level = level
        self.size = size
        self.wavelet = wavelet
        self.mode = mode
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations

        self.conv0 = WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet, self.mode)
        self.conv1 = WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet, self.mode)
        self.conv2 = WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet, self.mode)
        self.conv3 = WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet, self.mode)
        self.conv4 = WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet, self.mode)
        

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)

        
        self.bn0 = nn.BatchNorm2d(self.width)
        self.bn1 = nn.BatchNorm2d(self.width)
        self.bn2 = nn.BatchNorm2d(self.width)
        self.bn3 = nn.BatchNorm2d(self.width)
        self.bn4 = nn.BatchNorm2d(self.width)

       
        self.activation = nn.GELU()

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        
        dropout_rate = 0.05  
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.bn0(x)  
        x = self.activation(x)
        x = self.dropout(x)  

        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)

        
        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2
        x = self.bn4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

def gen4ddata(data, TT):
    data_4d = np.zeros((data.shape[0] - TT, data.shape[1], data.shape[2], TT))

    for i in range(data_4d.shape[0]):
        for j in range(TT):
            data_4d[i, :, :, j] = data[i + j, :, :]

    return data_4d

# Load data
if __name__ == '__main__':

    if process == 'ITest':
        norm_init2 = {}
        norm_init2['xvel'] = [0.56544, -0.33976]
        norm_init2['temp'] = [0.64661, 270.75000]

        dataset_train_path = f'./r{res}_init2/pr_dns_{var}.npy'
        dataset_3d = np.load(dataset_train_path)
        dataset_4d = gen4ddata(dataset_3d, 20)
        del dataset_3d
        train_a = dataset_4d[:ntrain, :, :, :T_in].astype(np.float32)
        train_u = dataset_4d[:ntrain, :, :, T_in:T + T_in].astype(np.float32)
        test_a = dataset_4d[-ntest:, :, :, :T_in].astype(np.float32)
        test_u = dataset_4d[-ntest:, :, :, T_in:T + T_in].astype(np.float32)
        del dataset_4d

    else:
        norm_init2 = {}
        norm_init2['xvel'] = []
        norm_init2['temp'] = []

        if process == 'STest':
            norm_init2['xvel'] = [0.28377, -0.13178]
            norm_init2['temp'] = [0.83325, 270.56335]

        dataset_train_path = f'./r{res}_init2/pr_dns_{var}.npy'
        dataset_3d = np.load(dataset_train_path)
        dataset_4d = gen4ddata(dataset_3d, 20)
        del dataset_3d
        train_a = dataset_4d[:ntrain, :, :, :T_in].astype(np.float32)
        train_u = dataset_4d[:ntrain, :, :, T_in:T + T_in].astype(np.float32)
        test_a = dataset_4d[-ntest:, :, :, :T_in].astype(np.float32)
        test_u = dataset_4d[-ntest:, :, :, T_in:T + T_in].astype(np.float32)
        del dataset_4d

   
    mean = np.mean(train_a, axis=(0,1,2,3), keepdims=True)
    std = np.std(train_a, axis=(0,1,2,3), keepdims=True)
    train_a = (train_a - mean) / std
    train_u = (train_u - mean) / std
    test_a = (test_a - mean) / std
    test_u = (test_u - mean) / std

    train_a = torch.from_numpy(train_a).to(device)
    train_u = torch.from_numpy(train_u).to(device)
    test_a = torch.from_numpy(test_a).to(device)
    test_u = torch.from_numpy(test_u).to(device)

    print(train_a.shape)
    print(train_u.shape)
    print(test_a.shape)
    print(test_u.shape)

   
    dataset = torch.utils.data.TensorDataset(train_a, train_u)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    validation_split = 0.1
    shuffle_dataset = True
    random_seed = 42

    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    test_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

    del train_a
    del train_u
    del test_a

    
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    # Training
    train_l2 = []
    test_l2 = []

    if process == 'Train':
        model = WNO2d(level, size, wavelet, mode, width).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        myloss = nn.MSELoss()

        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 20  
        trigger_times = 0

        log_path = f'./logs/WNO_log_{var}_r_{res}_nf'
        print(log_path)
        if os.path.exists(log_path):
            os.remove(log_path)
        ii = logger.add(log_path)

        for ep in range(epochs):
            model.train()
            train_l2_step = 0
            for xx, yy in train_loader:
                xx = xx.to(device)
                yy = yy.to(device)

                optimizer.zero_grad()
                with autocast():
                    im = model(xx)
                    loss = myloss(im, yy)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_l2_step += loss.item()
                
                torch.cuda.empty_cache()

            scheduler.step()

            # Validation
            model.eval()
            val_l2_step = 0
            with torch.no_grad():
                for xx, yy in validation_loader:
                    xx = xx.to(device)
                    yy = yy.to(device)
                    with autocast():
                        im = model(xx)
                        loss = myloss(im, yy)
                    val_l2_step += loss.item()
                    
                    torch.cuda.empty_cache()

            l2_train = train_l2_step / len(train_indices)
            l2_val = val_l2_step / len(val_indices)
            logger.info(f'Epoch {ep}, Training Loss: {l2_train:.5f}, Validation Loss: {l2_val:.5f}')

            # Early stopping check
            if l2_val < best_val_loss:
                best_val_loss = l2_val
                trigger_times = 0
                # Save the best model
                torch.save(model.state_dict(), f'./models/best_WNO_model_{var}_r_{res}_init1.pth')
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    logger.info('Early stopping!')
                    break

        logger.remove(ii)

    elif process == 'Test':
        test_l2_step = 0
        myloss = nn.MSELoss()
        model = WNO2d(level, size, wavelet, mode, width).to(device)
        model.load_state_dict(torch.load(f'./models/best_WNO_model_{var}_r_{res}_init1.pth', map_location=device))
        model.eval()

        preds = torch.zeros(test_u.shape).to(device, dtype=torch.float)

        print(f'{preds.shape=}')
        index = 0
        with torch.no_grad():
            s_time = time.time()
            infer_time = 0
            for xx, yy in test_loader1:
                xx = xx.to(device, dtype=torch.float)
                yy = yy.to(device, dtype=torch.float)

                t1 = time.time()
                with autocast():
                    out = model(xx)
                t2 = time.time()
                infer_time = infer_time + t2 - t1

                preds[index] = out
                index = index + 1
                loss = myloss(out, yy)
                test_l2_step += loss.item()
                
                torch.cuda.empty_cache()

            e_time = time.time()

            test_score = test_l2_step / ntest
            print(f'{test_score=:.5f} time={e_time - s_time:.2f}')
            tests = torch.stack([preds, test_u]).view(2, -1, 1, res, res).cpu().numpy()

            
            tests = tests * std + mean

        np.save(f"./tests/WNO_out_{var}_r_{res}_init1", tests)

    # You can similarly update 'STest' and 'ITest' processes if needed.
