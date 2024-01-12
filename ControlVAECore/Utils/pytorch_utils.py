'''
*************************************************************************

BSD 3-Clause License

Copyright (c) 2023,  Visual Computing and Learning Lab, Peking University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************
'''
from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'LRELU': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
    'ELU': nn.ELU(),
}

device = None


def init_gpu(use_gpu=False, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        # torch.backends.cudnn.benchmark = True
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def from_numpy_cpu(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float()

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

class scheduler(nn.Module):
    '''
    a custom scheduler
    '''
    def __init__(self, begin_point, end_point, begin_value, end_value, skip = 1):
        super(scheduler, self).__init__()
        self.begin_point = begin_point
        self.end_point = end_point
        self.begin_value = begin_value
        self.end_value = end_value
        self.skip = skip
        self.cnt = nn.Parameter(torch.tensor(0.0), requires_grad= False)
        self.skip_cnt = 1
        self.value = begin_value
    
    def step(self):
        self.skip_cnt += 1
        if self.skip_cnt == self.skip:
            self.skip_cnt = 0
            self.cnt += 1
            if self.cnt > self.end_point:
                self.value = self.end_value
            else:
                t = self.cnt / (self.end_point - self.begin_point)
                self.value = self.begin_value * (1-t) + self.end_value* t
                
                
                
                
def build_mlp(input_dim, output_dim, hidden_layer_num, hidden_layer_size, activation):
    activation_type = str_to_activation[activation]
    layers = []
    for i in range(hidden_layer_num):
        if i==0:
            layers.append(nn.Linear(input_dim, hidden_layer_size))
        else:
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        layers.append(activation_type)
    layers.append(nn.Linear(hidden_layer_size, output_dim))
    return nn.Sequential(*layers)

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean