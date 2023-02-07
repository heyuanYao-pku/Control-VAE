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
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
import math
from ControlVAECore.Utils.pytorch_utils import *

class Encoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, var, **params) -> None:
        super(Encoder, self).__init__()
        self.activation = str_to_activation[params['activation']]
        self.hidden_size = params['hidden_layer_size']
        self.fc_layers = []
        self.fc_layers.append(nn.Linear(input_size + condition_size, self.hidden_size))
        for i in range(params['hidden_layer_num']):
            self.fc_layers.append(nn.Linear(input_size + self.hidden_size, self.hidden_size))    
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.mu = nn.Linear(input_size + self.hidden_size, output_size)
        self.logvar = nn.Linear(input_size + self.hidden_size, output_size)
        self.var = var
        if self.var is not None:
            self.log_var = math.log(var)*2
    def encode(self, x, c):
        res = c
        for layer in self.fc_layers:
            if res is not None:
                res = layer(torch.cat([x,res], dim = -1))
            else:
                res = layer(x)
            res = self.activation(res)
            
        latent = torch.cat([x,res], dim = -1)
        mu = self.mu(latent)
        if self.var is not None:
            logvar = torch.ones_like(mu)*self.log_var
        else:
            logvar = self.logvar(latent)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        exp = torch.randn_like(std)
        return mu + exp * std
    
    def forward(self, x, c):
        mu, logvar = self.encode(x,c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class SimpleLearnablePriorEncoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, fix_var = None, **kargs) -> None:
        super(SimpleLearnablePriorEncoder, self).__init__()
        self.prior = Encoder(
            input_size= condition_size,
            condition_size= 0,
            output_size= output_size,
            var = fix_var,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        
        self.post = Encoder(
            input_size= input_size,
            condition_size= condition_size,
            output_size= output_size,
            var = fix_var,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        self.var = fix_var
        
    def kl_loss(self, mu_prior, mu_post):
        return 0.5*(mu_prior - mu_post)**2/(self.var**2)
    
    def encode_prior(self, n_observation):
        return self.prior(n_observation, None)
    def encode_post(self, n_observation, n_target):
        return self.post(n_target, n_observation)
    
    def forward(self, n_observation, n_target):
        _, mu_prior, logvar_prior = self.encode_prior(n_observation)
        latent_code, mu_post, logvar_post = self.encode_post(n_observation, n_target)
        
        return latent_code + mu_prior, mu_prior+mu_post, mu_prior
    

class StandardVAEEncoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, fix_var, **kargs) -> None:
        super(StandardVAEEncoder, self).__init__()
        print('StandardVAEEncoder')
        self.encoder = Encoder(
            input_size= input_size,
            condition_size= condition_size,
            output_size= output_size,
            var = fix_var,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        self.var = fix_var
        self.latent_dim = output_size
        if self.var is not None:
            self.log_var = math.log(fix_var)*2
    def kl_loss(self, mu_prior, mu_post):
        return 0.5*(mu_prior - mu_post)**2/(self.var**2)
    
    def encode_prior(self, x):
        assert len(x.shape) == 2
        shape = (x.shape[0], self.latent_dim)
        mu = torch.zeros(shape, dtype = x.dtype, device= x.device)
        return torch.randn_like(mu)*self.var, mu, torch.ones_like(mu)*self.log_var
    
    def forward(self, normalized_obs, normalized_target):
        z, mu, logvar = self.encoder(normalized_target,normalized_obs)
        return z, mu, torch.zeros_like(mu)
    
    
class LearnablePriorEncoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, fix_var = None, **kargs) -> None:
        super(LearnablePriorEncoder, self).__init__()
        
        self.prior = build_mlp(
            input_dim = input_size,
            # output_dim = kargs['encoder_hidden_layer_size'],
            output_dim= output_size,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        
        self.posterior = build_mlp(
            input_dim = input_size + condition_size,
            # output_dim = kargs['encoder_hidden_layer_size'],
            output_dim= output_size,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        
        # Maybe here should add an activation before it, just try it.... although I don't think it matters...
        self.mu = nn.Linear(kargs['encoder_hidden_layer_size'], output_size)
        self.var = fix_var
        if self.var is not None:
            self.log_var = math.log(self.var)*2
        else:
            self.log_var = nn.Linear(kargs['encoder_hidden_layer_size'], output_size)
    
    def feature2muvar(self, feature):
        mu = feature#self.mu(feature)
        if self.var is not None:
            logvar = torch.ones_like(mu) * self.log_var
        else:
            logvar = self.log_var(feature)
        return mu, logvar
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        exp = torch.randn_like(mu)
        return mu + exp * std
    
    def kl_loss(self, mu_prior, mu_post):
        return 0.5*(mu_prior - mu_post)**2/(self.var**2)
    
    def encode_prior(self, normalized_observation):
        feature = self.prior(normalized_observation)
        mu, logvar = self.feature2muvar(feature)
        return mu, logvar
    
    def encode_posterior(self, normalized_observation, normalized_target):
        feature = self.posterior(torch.cat([normalized_observation, normalized_target], dim = -1))
        mu, logvar = self.feature2muvar(feature)
        return mu, logvar
    
    def forward(self, normalized_observation, normalized_target):
        """encode observation and target into posterior distribution

        Args:
            normalized_observation (tensor): observation
            normalized_target (tensor): target
        
        Returns:
            Tuple(tensor, tensor, tensor): latent code sampled from posterior distribution,
                mean of prior distribution, mean of posterior distribution
        """
        mu_prior, logvar_prior = self.encode_prior(normalized_observation)
        mu_post, logvar_post = self.encode_posterior(normalized_observation, normalized_target)
        
        latent_code = self.reparameterize(mu_prior+mu_post, logvar_post)
        return latent_code, mu_prior+mu_post, mu_prior



class GatingMixedDecoder(nn.Module):
    def __init__(self, latent_size, condition_size, output_size, **kargs):
        super(GatingMixedDecoder, self).__init__()

        input_size = latent_size+condition_size
        hidden_size = kargs['actor_hidden_layer_size']
        inter_size = latent_size + hidden_size
        num_experts = kargs['actor_num_experts']
        num_layer = kargs['actor_hidden_layer_num']
        self.activation = str_to_activation[kargs['actor_activation']]
        self.decoder_layers = []
        
        # put in list then initialize and register
        for i in range(num_layer + 1):
            layer = (
                nn.Parameter(torch.empty(num_experts, inter_size if i!=0 else input_size, hidden_size if i!=num_layer else output_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size if i!=num_layer else output_size)),
                self.activation if i < num_layer else None 
            )
            self.decoder_layers.append(layer)

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            stdv = 1. / math.sqrt(weight.size(1))
            weight.data.uniform_(-stdv, stdv)
            bias.data.uniform_(-stdv, stdv)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        gate_hsize = kargs['actor_gate_hidden_layer_size']
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )
    
    def forward(self, z, c):
        assert len(c.shape) > 1
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=-1)), dim=-1)
        layer_out = c
        for (weight, bias, activation) in self.decoder_layers:
            input = z if layer_out is None else torch.cat((z, layer_out), dim=-1)
            input = F.layer_norm(input, input.shape[1:])
            mixed_bias = contract('be,ek->bk',coefficients, bias)
            mixed_input = contract('be,bj,ejk->bk', coefficients, input, weight)
            out = mixed_input + mixed_bias
            layer_out = activation(out) if activation is not None else out
        return layer_out
    
