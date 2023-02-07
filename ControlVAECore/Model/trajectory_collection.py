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
import numpy as np
import torch
import operator
from ..Utils import pytorch_utils as ptu

class TrajectorCollector():
    def __init__(self, **kargs) -> None:
        self.reset(**kargs)
    
    def reset(self, venv, **kargs):
        # set environment and actor
        self.env = venv
        
        # set property
        self.with_noise = kargs['runner_with_noise']
        
        
    # @torch.no_grad
    def trajectory_sampling(self, sample_size, actor):
        cnt = 0
        res = []
        while cnt < sample_size:
            trajectory =self.sample_one_trajectory(actor) 
            res.append(trajectory)
            cnt+= len(trajectory['done'])
        
        # res = functools.reduce(operator.add, map(collections.Counter, res))
        res_dict = {}
        for key in res[0].keys():
            res_dict[key] = np.concatenate( list(map(operator.itemgetter(key), res)) , axis = 0)
        return res_dict
    
    def eval_one_trajectory(self, actor):
        saver = self.env.get_bvh_saver()
        observation, info = self.env.reset()
        while True: 
            saver.append_no_root_to_buffer()

            # when eval, we do not hope noise...
            action = actor.act_determinastic(observation)
            action = ptu.to_numpy(action).flatten()
            new_observation, rwd, done, info = self.env.step(action)
            observation = new_observation
            if done:
                break
        return saver
    
    # @torch.no_grad
    def sample_one_trajectory(self, actor):
        observation, info = self.env.reset() 
        
        states, targets, actions, rwds, dones, frame_nums = [[] for i in range(6)]
         
        while True: 
            if self.with_noise:
                action_distribution = actor.act_distribution(observation)
                action = action_distribution.sample()
            else:
                action = actor.act_determinastic(observation)
            
            if np.random.choice([True, False], p = [0.4, 0.6]):
                action = actor.act_prior(observation)
                action = action + torch.randn_like(action) * 0.05
            action = ptu.to_numpy(action).flatten()
            
            states.append(observation['state'])
            actions.append(action)
            targets.append(observation['target'])
            
            new_observation, rwd, done, info = self.env.step(action)
            
            rwd = actor.cal_rwd(observation = new_observation['observation'], target = observation['target'])
            rwds.append(rwd)
            dones.append(done)
            frame_nums.append(info['frame_num'])
            
            observation = new_observation
            if done:
                break
        return {
            'state': states,
            'action': actions,
            'target': targets,
            'done': dones,
            'rwd': rwds,
            'frame_num': frame_nums
        }
            
            
            
            