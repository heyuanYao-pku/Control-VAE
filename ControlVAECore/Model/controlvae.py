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
import random
from typing import List,Dict
from numpy import dtype
import torch
from torch import nn
import torch.distributions as D
from ControlVAECore.Model.trajectory_collection import TrajectorCollector
from ControlVAECore.Model.world_model import SimpleWorldModel
from ControlVAECore.Utils.mpi_utils import gather_dict_ndarray
from ControlVAECore.Utils.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
from modules import *
from ..Utils.motion_utils import *
from ..Utils import pytorch_utils as ptu
import time
import sys
from ControlVAECore.Utils.radam import RAdam
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_world_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

# whether this process should do tasks such as trajectory collection....
# it's true when it's not root process or there is only root process (no subprocess)
should_do_subprocess_task = mpi_rank > 0 or mpi_world_size == 1

class ControlVAE(nn.Module):
    """
    A ContorlVAE agent which includes encoder, decoder and world model
    """
    def __init__(self, observation_size, action_size, delta_size, env, **kargs):
        super().__init__()
        
        # components of controlVAE
        self.encoder = SimpleLearnablePriorEncoder(
            input_size= observation_size,
            condition_size= observation_size,
            output_size= kargs['latent_size'],
            fix_var = kargs['encoder_fix_var'],
            **kargs).to(ptu.device)
        self.agent = GatingMixedDecoder(
            # latent_size= kargs['latent_size'],
            condition_size= observation_size,
            output_size=action_size,
            **kargs
        ).to(ptu.device)
        
        # statistics, will be used to normalize each dimention of observation
        statistics = env.stastics
        self.obs_mean = nn.Parameter(ptu.from_numpy(statistics['obs_mean']), requires_grad = False).to(ptu.device)
        self.obs_std = nn.Parameter(ptu.from_numpy(statistics['obs_std']), requires_grad= False).to(ptu.device)
        
        # world model
        self.world_model = SimpleWorldModel(observation_size, action_size, delta_size, env.dt, statistics, **kargs).to(ptu.device)
        
        # optimizer
        self.wm_optimizer = RAdam(self.world_model.parameters(), kargs['world_model_lr'], weight_decay=1e-3)
        self.vae_optimizer = RAdam( list(self.encoder.parameters()) + list(self.agent.parameters()), kargs['controlvae_lr'])
        self.beta_scheduler = ptu.scheduler(0,8,0.009,0.09,500*8)
        
        #hyperparameters....
        self.action_sigma = 0.05
        self.max_iteration = kargs['max_iteration']
        self.collect_size = kargs['collect_size']
        self.sub_iter = kargs['sub_iter']
        self.save_period = kargs['save_period']
        self.evaluate_period = kargs['evaluate_period']
        self.world_model_rollout_length = kargs['world_model_rollout_length']
        self.controlvae_rollout_length = kargs['controlvae_rollout_length']
        self.world_model_batch_size = kargs['world_model_batch_size']
        self.controlvae_batch_size = kargs['controlvae_batch_size']
        
        # policy training weights                                    
        self.weight = {}
        for key,value in kargs.items():
            if 'controlvae_weight' in key:
                self.weight[key.replace('controlvae_weight_','')] = value
        
        # for real trajectory collection
        self.runner = TrajectorCollector(venv = env, actor = self, runner_with_noise = True)
        self.env = env    
        self.replay_buffer = ReplayBuffer(self.replay_buffer_keys, kargs['replay_buffer_size']) if mpi_rank ==0 else None
        self.kargs = kargs
    #--------------------------------for MPI sync------------------------------------#
    def parameters_for_sample(self):
        '''
        this part will be synced using mpi for sampling, world model is not necessary
        '''
        return {
            'encoder': self.encoder.state_dict(),
            'agent': self.agent.state_dict()
        }
    def load_parameters_for_sample(self, dict):
        self.encoder.load_state_dict(dict['encoder'])
        self.agent.load_state_dict(dict['agent'])
    
    #-----------------------------for replay buffer-----------------------------------#
    @property
    def world_model_data_name(self):
        return ['state', 'action']
    
    @property
    def policy_data_name(self):
        return ['state', 'target']
    
    @property
    def replay_buffer_keys(self):
        return ['state', 'action', 'target']

    #----------------------------for training-----------------------------------------#
    def train_one_step(self):
        
        time1 = time.perf_counter()
        
        # data used for training world model
        name_list = self.world_model_data_name
        rollout_length = self.world_model_rollout_length
        data_loader = self.replay_buffer.generate_data_loader(name_list, 
                            rollout_length+1, # needs additional one state...
                            self.world_model_batch_size, 
                            self.sub_iter)
        for batch in  data_loader:
            world_model_log = self.train_world_model(*batch)
        
        time2 = time.perf_counter()
        
        # data used for training policy
        name_list = self.policy_data_name
        rollout_length = self.controlvae_rollout_length
        data_loader = self.replay_buffer.generate_data_loader(name_list, 
                            rollout_length, 
                            self.controlvae_batch_size, 
                            self.sub_iter)
        for batch in data_loader:
            policy_log = self.train_policy(*batch)
        
        # log training time...
        time3 = time.perf_counter()      
        world_model_log['training_time'] = (time2-time1)
        policy_log['training_time'] = (time3-time2)
        
        # merge the training log...
        return self.merge_dict([world_model_log, policy_log], ['WM','Policy'])
    
    def mpi_sync(self):
        
        # sample trajectories
        if should_do_subprocess_task:
            with torch.no_grad():
                path : dict = self.runner.trajectory_sampling( math.floor(self.collect_size/max(1, mpi_world_size -1)), self )
                self.env.update_val(path['done'], path['rwd'], path['frame_num'])
        else:
            path = {}

        tmp = np.zeros_like(self.env.val)
        mpi_comm.Allreduce(self.env.val, tmp)        
        self.env.val = tmp / mpi_world_size
        self.env.update_p()
        
        res = gather_dict_ndarray(path)
        if mpi_rank == 0:
            paramter = self.parameters_for_sample()
            mpi_comm.bcast(paramter, root = 0)
            self.replay_buffer.add_trajectory(res)
            info = {
                'rwd_mean': np.mean(res['rwd']),
                'rwd_std': np.std(res['rwd']),
                'episode_length': len(res['rwd'])/(res['done']!=0).sum()
            }
        else:
            paramter = mpi_comm.bcast(None, root = 0)
            self.load_parameters_for_sample(paramter)    
            info = None
        return info
    
    
    def train_loop(self):
        """training loop, MPI included
        """
        for i in range(self.max_iteration):
            # if i ==0:
            info = self.mpi_sync() # communication, collect samples and broadcast policy
            
            if mpi_rank == 0:
                print(f"----------training {i} step--------")
                sys.stdout.flush()
                log = self.train_one_step()   
                log.update(info)       
                self.try_save(i)
                self.try_log(log, i)

            if should_do_subprocess_task:
                self.try_evaluate(i)
                
    # -----------------------------------for logging----------------------------------#
    @property
    def dir_prefix(self):
        return 'Experiment'
    
    def save_before_train(self, args):
        """build directories for log and save
        """
        import os, time, yaml
        time_now = time.strftime("%Y%m%d %H-%M-%S", time.localtime())
        dir_name = args['experiment_name']+'_'+time_now
        dir_name = mpi_comm.bcast(dir_name, root = 0)
        
        self.log_dir_name = os.path.join(self.dir_prefix,'log',dir_name)
        self.data_dir_name = os.path.join(self.dir_prefix,'checkpoint',dir_name)
        if mpi_rank == 0:
            os.makedirs(self.log_dir_name)
            os.makedirs(self.data_dir_name)

        mpi_comm.barrier()
        if mpi_rank > 0:
            f = open(os.path.join(self.log_dir_name,f'mpi_log_{mpi_rank}.txt'),'w')
            sys.stdout = f
            return
        else:
            yaml.safe_dump(args, open(os.path.join(self.data_dir_name,'config.yml'),'w'))
            self.logger = SummaryWriter(self.log_dir_name)
            
    def try_evaluate(self, iteration):
        if iteration % self.evaluate_period == 0:
            bvh_saver = self.runner.eval_one_trajectory(self)
            bvh_saver.to_file(os.path.join(self.data_dir_name,f'{iteration}_{mpi_rank}.bvh'))
        pass    
    
    def try_save(self, iteration):
        if iteration % self.save_period ==0:
            check_point = {
                'self': self.state_dict(),
                'wm_optim': self.wm_optimizer.state_dict(),
                'vae_optim': self.vae_optimizer.state_dict(),
                'balance': self.env.val
            }
            torch.save(check_point, os.path.join(self.data_dir_name,f'{iteration}.data'))
    
    def try_load(self, data_file):
        data = torch.load(data_file, map_location=ptu.device)
        self.load_state_dict(data['self'], strict = False)
        self.wm_optimizer.load_state_dict(data['wm_optim'])
        self.vae_optimizer.load_state_dict(data['vae_optim'])
        if 'balance' in data:
            self.env.val = data['balance']
            self.env.update_p()
        return data
        
    def try_log(self, log, iteration):
        for key, value in log.items():
            self.logger.add_scalar(key, value, iteration)
        self.logger.flush()
    
    def cal_rwd(self, **obs_info):
        observation = obs_info['observation']
        target = obs_info['target']
        error = pose_err(torch.from_numpy(observation), torch.from_numpy(target), self.weight, dt = self.env.dt)
        error = sum(error).item()
        return np.exp(-error/20)
    
    
    @staticmethod
    def add_specific_args(arg_parser):
        arg_parser.add_argument("--latent_size", type = int, default = 64, help = "dim of latent space")
        arg_parser.add_argument("--max_iteration", type = int, default = 20001, help = "iteration for controlVAE training")
        arg_parser.add_argument("--collect_size", type = int, default = 2048, help = "number of transition collect for each iteration")
        arg_parser.add_argument("--sub_iter", type = int, default = 8, help = "num of batch in each iteration")
        arg_parser.add_argument("--save_period", type = int, default = 100, help = "save checkpoints for every * iterations")
        arg_parser.add_argument("--evaluate_period", type = int, default = 100, help = "save checkpoints for every * iterations")
        arg_parser.add_argument("--replay_buffer_size", type = int, default = 50000, help = "buffer size of replay buffer")

        return arg_parser
    
    #--------------------------API for encode and decode------------------------------#
    
    def encode(self, normalized_obs, normalized_target, **kargs):
        """encode observation and target into posterior distribution

        Args:
            normalized_obs (Optional[Tensor,np.ndarray]): normalized current observation
            normalized_target (Optional[Tensor, np.ndarray]): normalized current target 

        Returns:
            Tuple(tensor, tensor, tensor): 
                latent coder, mean of prior distribution, mean of posterior distribution 
        """
        return self.encoder(normalized_obs, normalized_target)
    
    def decode(self, normalized_obs, latent, **kargs):
        """decode latent code into action space

        Args:
            normalized_obs (tensor): normalized current observation
            latent (tensor): latent code

        Returns:
            tensor: action
        """
        action = self.agent(latent, normalized_obs)        
        return action
    
    def normalize_obs(self, observation):
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None,...]
        return ptu.normalize(observation, self.obs_mean, self.obs_std)
    
    def obsinfo2n_obs(self, obs_info):
        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']
        else:
            if 'observation' in obs_info:
                observation = obs_info['observation']
            else:
                observation = state2ob(obs_info['state'])
            n_observation = self.normalize_obs(observation)
        return n_observation
    
    def act_tracking(self, **obs_info):
        """
        try to track reference motion
        """
        target = obs_info['target']
        
        n_target = self.normalize_obs(target)
        n_observation = self.obsinfo2n_obs(obs_info)
        
        latent_code, mu_post, mu_prior = self.encode(n_observation, n_target)
        action = self.decode(n_observation, latent_code)
        info = {
            "mu_prior": mu_prior,
            "mu_post": mu_post
        }
        return action, info
    
    def act_prior(self, obs_info):
        """
        try to track reference motion
        """
        n_observation = self.obsinfo2n_obs(obs_info)
        latent_code, mu_prior, logvar = self.encoder.encode_prior(n_observation)
        action = self.decode(n_observation, latent_code)
        
        return action
    
    #----------------------------------API imitate PPO--------------------------------#
    def act_determinastic(self, obs_info):
        action, _ = self.act_tracking(**obs_info)
        return action
                
    def act_distribution(self, obs_info):
        """
        Add noise to the output action
        """
        action = self.act_determinastic(obs_info)
        action_distribution = D.Independent(D.Normal(action, self.action_sigma), -1)
        return action_distribution
    
    #--------------------------------------Utils--------------------------------------#
    @staticmethod
    def merge_dict(dict_list: List[dict], prefix: List[str]):
        """Merge dict with prefix, used in merge logs from different model

        Args:
            dict_list (List[dict]): different logs
            prefix (List[str]): prefix you hope to add before keys
        """
        res = {}
        for dic, prefix in zip(dict_list, prefix):
            for key, value in dic.items():
                res[prefix+'_'+key] = value
        return res
    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def try_load_world_model(self, data_file):
        data = torch.load(data_file, map_location=ptu.device)
        wm_dict = data['self']
        wm_dict = {k.replace('world_model.',''):v for k,v in wm_dict.items() if 'world_model' in k}
        self.world_model.load_state_dict(wm_dict)
        return data
    #--------------------------------Training submodule-------------------------------#
    
    def train_policy(self, states, targets):
        rollout_length = states.shape[1]
        loss_name = ['pos', 'rot', 'vel', 'avel', 'height', 'up_dir', 'acs', 'kl']
        loss_num = len(loss_name)
        loss = list( ([] for _ in range(loss_num)) )
        states = states.transpose(0,1).contiguous().to(ptu.device)
        targets = targets.transpose(0,1).contiguous().to(ptu.device)
        cur_state = states[0]
        cur_observation = state2ob(cur_state)
        n_observation = self.normalize_obs(cur_observation)
        for i in range(rollout_length):
            target = targets[i]
            action, info = self.act_tracking(n_observation = n_observation, target = target)
            action = action + torch.randn_like(action)*0.05
            cur_state = self.world_model(cur_state, action, n_observation = n_observation)
            cur_observation = state2ob(cur_state)
            n_observation = self.normalize_obs(cur_observation)
            
            loss_tmp = pose_err(cur_observation, target, self.weight, dt = self.env.dt)
            for j, value in enumerate(loss_tmp):
                loss[j].append(value)        
            acs_loss = self.weight['l2'] * torch.mean(torch.sum(action**2,dim = -1)) \
                + self.weight['l1'] * torch.mean(torch.norm(action, p=1, dim=-1))
            kl_loss = self.encoder.kl_loss(**info)
            kl_loss = torch.mean( torch.sum(kl_loss, dim = -1))
            loss[-2].append(acs_loss)
            loss[-1].append(kl_loss * self.beta_scheduler.value)
        
        loss_value = [ sum( (0.95**i)*l[i] for i in range(rollout_length) )/rollout_length for l in loss]
        loss = sum(loss_value)

        self.vae_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1, error_if_nonfinite= True)
        self.vae_optimizer.step()
        self.beta_scheduler.step()
        res = {loss_name[i]: loss_value[i] for i in range(loss_num)}
        res['beta'] = self.beta_scheduler.value
        res['loss'] = loss
        return res
    
    
    def train_world_model(self, states, actions):
        rollout_length = states.shape[1] -1
        loss_name = ['pos', 'rot', 'vel', 'avel']
        loss_num = len(loss_name)
        loss = list( ([] for _ in range(loss_num)) )
        states = states.transpose(0,1).contiguous().to(ptu.device)
        actions = actions.transpose(0,1).contiguous().to(ptu.device)
        cur_state = states[0]
        for i in range(rollout_length):
            next_state = states[i+1]
            pred_next_state = self.world_model(cur_state, actions[i])
            loss_tmp = self.world_model.loss(pred_next_state, next_state)
            cur_state = pred_next_state
            for j in range(loss_num):
                loss[j].append(loss_tmp[j])
        
        loss_value = [sum(i) for i in loss]
        loss = sum(loss_value)
        
        self.wm_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1, error_if_nonfinite=True)
        self.wm_optimizer.step()
        res= {loss_name[i]: loss_value[i] for i in range(loss_num)}
        res['loss'] = loss
        return res
    
    