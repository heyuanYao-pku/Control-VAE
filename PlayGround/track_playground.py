
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
from ControlVAECore.Env.vclode_track_env import VCLODETrackEnv
from PlayGround.random_playground import RandomPlayground
import torch
import ControlVAECore.Utils.pytorch_utils as ptu
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import argparse

class TrackPlayground(RandomPlayground):
    # def get_action(self, **obs_info):
    #     return self.act_determinastic(obs_info), {}

    def get_action(self, **obs_info):
        n_observation = self.obsinfo2n_obs(obs_info)
        target = obs_info['target']
        n_target = self.normalize_obs(target)
        
        # latent, mu, logvar = self.encoder.encode_post(n_observation, n_target)
        latent_code, mu_post, mu_prior = self.encode(n_observation, n_target)
        action = self.decode(n_observation, mu_post)

        # info = {'mu': mu, 'logvar': logvar}
        info = {}
        return action, info
    
    def yield_step(self):
        super().yield_step()
        self.env.load_character_state(self.env.ref_character, self.env.motion_data.state[self.env.counter])
    
    
    def visualize_motion_connection(self):
        observation = self.env.motion_data.observation
        l1 = len(observation)//2
        observation = observation[:l1][::2]
        n_observation = self.obsinfo2n_obs({'observation' : observation})
        l = len(n_observation)
        
        res = []
        idx = torch.arange(0, l)
        distribution = torch.distributions.Normal(0, self.kargs['encoder_fix_var'])
        for i in range(l):
            latent, mu, logvar = self.encoder.encode_post(n_observation[i].tile([l,1]), n_observation)
            logp = distribution.log_prob(mu)
            logp = logp.sum(1).view(1, -1)
            res.append(ptu.to_numpy(logp))
        res = np.concatenate(res, axis = 0)
        sns.heatmap(res, cmap = 'RdBu_r', vmin = -100, vmax = 0)
        plt.show()
        
        plt.figure()
        data = res.sum(0)
        data /= len(data)
        plt.plot(np.arange(len(data)), data)
        plt.show()
    
    def visualize_random(self):
        plt.figure()
        res = []
        for i in range(10000):
            self.env.reset()
            res.append(self.env.counter)
        res = np.array(res)
        res = res[res < len(self.env.p)//2]
        plt.hist(res)
        plt.show()
        
    def visualize_p(self):
        l1 = len(self.env.p)
        data = self.env.p[:l1//2][::2]
        plt.figure()
        plt.plot(np.arange(len(data)), data)
        plt.show()
        return data
if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--use', type = str, default = 'run')
    args = RandomPlayground.build_arg(parser)
    args['show'] = True
    env = VCLODETrackEnv(**args)
    playground = TrackPlayground(323, 57, 120, env, **args)
    # load
    import tkinter.filedialog as fd
    data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
    playground.try_load(data_file)
    
    if args['use'] == 'run':
        # playground.init_viewer()
        playground.run()
    elif args['use'] == 'visualize':
        playground.visualize_motion_connection()
        playground.visualize_p()
        playground.visualize_random()
    elif args['use'] == 'visualize_wm':
        playground.init_viewer()
        playground.save()