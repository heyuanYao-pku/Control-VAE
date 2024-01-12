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
import argparse
from ControlVAECore.Env.vclode_track_env import VCLODETrackEnv
from ControlVAECore.Model.controlvae import ControlVAE
# from ControlVAECore.Model.trajectory_collection import TrajectorCollector
from ControlVAECore.Utils.motion_utils import state2ob
from ControlVAECore.Utils.pytorch_utils import build_mlp
from ControlVAECore.Utils.radam import RAdam
from PlayGround.playground_util import get_root_facing, state2speed
from random_playground import RandomPlayground
import ControlVAECore.Utils.pytorch_utils as ptu
import torch
import numpy as np
import types
from scipy.spatial.transform import Rotation

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
def random_target(env):
    speed = np.random.choice(env.speed_range)
    direction_angle = np.random.uniform(0, np.pi * 2)
    res = np.array([speed, direction_angle])
    print("speed:", speed)
    return res

def speed_target(self):
    if not hasattr(self, 'target') or self.target is None:
        self.target = random_target(self)
    return self.target

def show_arrow_drawstuff(self):
    pos = self.state[...,0,:3].copy()
    pos[1] = 0
    self.arrow.PositionNumpy = pos.flatten().astype(np.float64)
    angle = self.target[...,1]
    new_rotation = Rotation.from_rotvec(np.array([0,np.pi/2- angle,0]))
    rot =  new_rotation *self.base_rotation
    self.arrow.setRotationNumpy(rot.as_matrix().flatten())
    pass

def show_arrow_panda(self):
    pass


def after_step(self, **kargs):
    self.step_cnt += 1
    if not self.interaction:
        if self.step_cnt % self.random_count == 0:
            self.target = random_target(self)
    else:
        speed = self.interactor.state_dict['speed']
        y,x =  self.interactor.state_dict['y_axis'], self.interactor.state_dict['x_axis']
        angle = np.arctan2(-y, x)
        
        camera = self.interactor.cameractrl
        cam_dir = camera.center - camera.position
        cam_dir = np.arctan2(cam_dir[0], cam_dir[1]) + np.pi/2
        
        # # cam_dir = camera.position - camera.center
        # cam_dir = np.arctan2(-cam_dir[0], cam_dir[1]) - np.pi/2
        
        angle -= cam_dir 
        
        print("angle: ", angle*180/np.pi, "speed:", speed)
        res = np.array([speed, angle])
        self.target = res
        
    if self.show:
        self.show_arrow()


class SpeedPlayground(RandomPlayground):
    def __init__(self, observation_size, action_size, delta_size, env, **kargs):
        kargs['replay_buffer_size'] = 1000
        super().__init__(observation_size, action_size, delta_size, env, **kargs)
        self.observation_size = observation_size
        self.latent_size = kargs['latent_size']
        self.batch_size = 512
        self.collect_size = 500
        self.env.max_length = 256
        self.runner.with_noise = kargs['train'] # use act_determinastic....
        
        if mpi_rank == 0:
            self.replay_buffer.reset_max_size(2000)
        # self.replay_buffer.reset_max_size(kargs['replay_buffer_size'])
        
        # modify env.. an add-hoc way
        # self.env.speed_range = [0,1,2,3,4,5]
        self.env.speed_range = [0,0,1,2,3]
        self.env.get_target = types.MethodType(speed_target, self.env)
        self.env.after_step = types.MethodType(after_step, self.env)   
        self.env.interaction = kargs['interaction']
        self.env.show = self.show
        # self.env.random_count = 40
        self.build_high_level()
        
        self.mass = self.env.sim_character.body_info.mass_val / self.env.sim_character.body_info.sum_mass
        self.mass = ptu.from_numpy(self.mass).view(-1)

        self.dance = kargs['dance']
        if self.show:
            if self.mode == 'drawstuff':
                try:
                    from VclSimuBackend.ODESim.Loader.MeshCharacterLoader import MeshCharacterLoader
                except ImportError:
                    import VclSimuBackend
                    MeshCharacterLoader = VclSimuBackend.ODESim.MeshCharacterLoader
                MeshLoader = MeshCharacterLoader(self.env.scene.world, self.env.scene.space)
                self.env.arrow = MeshLoader.load_from_obj('./arrow.obj', 'arrow', volume_scale=1, density_scale=1)
                self.env.arrow.is_enable = False
                self.env.arrow = self.env.arrow.root_body
                self.env.base_rotation = Rotation.from_rotvec(np.array([-np.pi/2,0,0]))
                self.env.show_arrow = types.MethodType(show_arrow_drawstuff, self.env)
                # self.renderObj.track_body(self.env.arrow, False)
            elif self.mode == 'panda':
                self.env.show_arrow = types.MethodType(show_arrow_panda, self.env)
                self.env.interactor = self.panda_server
            elif self.mode == 'unity':
                self.env.interactor = self.unity_server
                self.env.show_arrow = types.MethodType(show_arrow_panda, self.env)
        # self.max_iteration = 1001
    
    def panda_keyboard_handler(self):
        pass
    
    #-----------------------------deal with parameters--------------------------------#
    def parameters_for_sample(self):
        res =  super().parameters_for_sample()
        res['high_level'] = self.high_level.state_dict()
        return res
    
    def load_parameters_for_sample(self, dict):
        super().load_parameters_for_sample(dict)
        self.high_level.load_state_dict(dict['high_level'])
    
    def try_evaluate(self, iteration):
        pass
    
    def try_save(self, iteration):
        if iteration % self.save_period == 0:
            check_point = {
                    'self': self.state_dict(),
                    'wm_optim': self.wm_optimizer.state_dict(),
                    'vae_optim': self.vae_optimizer.state_dict(),
                    'balance': self.env.val,
                    'high_level': self.high_level.state_dict(),
                    'high_level_optim': self.high_level_optim.state_dict(),
                }
            import os
            torch.save(check_point, os.path.join(self.data_dir_name,f'{iteration}.data'))

    def try_load(self, data_file):
        data = super().try_load(data_file)
        self.high_level.load_state_dict(data['high_level'])
        self.high_level_optim.load_state_dict(data['high_level_optim'])
        return data
    @property
    def dir_prefix(self):
        return 'Experiment/playground'
    
    def cal_rwd(self, **obs_info):
        return 0
    
    #------------------------------------------task-----------------------------------#    
    @property   
    def task_ob_size(self):
        return self.observation_size + 3
    
    def build_high_level(self):
        self.high_level = build_mlp(self.task_ob_size, self.latent_size, 3, 256, 'ELU').to(ptu.device)
        self.high_level_optim = RAdam(self.high_level.parameters(), lr=1e-3)
        lr = lambda epoach: max(0.99**(epoach), 1e-1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.high_level_optim, lr)
        
    @staticmethod
    def target2n_target(state, target):
        if len(state.shape) ==2:
            state = state[None,...]
        if len(target.shape) ==1:
            target = target[None,...]
        if isinstance(target, np.ndarray):
            target = ptu.from_numpy(target)
        if isinstance(state, np.ndarray):
            state = ptu.from_numpy(state)
        facing_direction = get_root_facing(state)
        facing_angle = torch.arctan2(facing_direction[:,2], facing_direction[:,0])
        delta_angle = target[:,1] - facing_angle
        res = torch.cat([target[:,0, None], torch.cos(delta_angle[:,None]), torch.sin(delta_angle[:,None])], dim = -1)
        return res
        
    
    #------------------------------------------acting-------------------------------#
    def act_task(self, **obs_info):
        
        n_observation = self.obsinfo2n_obs(obs_info)
        latent, mu, _ = self.encoder.encode_prior(n_observation)    
        n_target = self.target2n_target(obs_info['state'], obs_info['target'])
        
        task = torch.cat([n_observation, n_target], dim=1)
        offset = self.high_level(task)
        if self.dance:
            if n_target[...,2].abs()<0.5:
                latent = latent
            else:
                latent = latent + offset
        else:
            latent = mu+offset
        
        action = self.decode(n_observation, latent)
        return action, {
            'mu': mu,
            'latent': latent,
            'offset': offset
        }
    
    def act_determinastic(self, obs_info):
        return self.act_task(**obs_info)[0]
    
    #------------------------------------------training-------------------------------#
    @property
    def high_level_data_name_list(self):
        return ['state', 'target']
    
    def train_one_step(self):
        
        name_list = self.high_level_data_name_list
        rollout_length = 16
        # self.sub_iter = 2
        data_loader = self.replay_buffer.\
            generate_data_loader(   name_list,
                                    rollout_length,
                                    self.batch_size,
                                    self.sub_iter
                                )
        for batch_dict in data_loader:
            log = self.train_high_level(*batch_dict)
        self.scheduler.step()
        return log
    
    def get_loss(self, state, target):
        
        direction = get_root_facing(state)
        delta_angle = torch.atan2(direction[:,2], direction[:,0]) - target[:,1]
        direction_loss = torch.acos(torch.cos(delta_angle).clamp(min=-1+1e-4, max=1-1e-4))/ torch.pi
        
        com_vel = state2speed(state, self.mass)
        target_direction = torch.cat([torch.cos(target[:,1,None]),torch.sin(target[:,1,None])], dim=-1)
        com_vel = torch.where(target[:,0]==0, torch.norm(com_vel, dim=-1, p = 1), torch.einsum('bi,bi->b', com_vel[:,[0,2]], target_direction))
        # com_vel = torch.einsum('bi,bi->b', com_vel[:,[0,2]], target_direction)
        # com_vel = torch.norm(com_vel, dim = -1)
        speed_loss = torch.abs(com_vel - target[:,0])/target[:,0].clamp(min=1)
        
        fall_down_loss = torch.clamp(state[...,0,1], min = 0, max = 0.6)
        fall_down_loss = (0.6 - fall_down_loss)
        fall_down_loss = torch.mean(fall_down_loss)
        
        return direction_loss.mean(), speed_loss.mean(), fall_down_loss
    
    def train_high_level(self, states, targets):
        rollout_length = states.shape[1]
        cur_state = states[:,0].to(ptu.device)
        targets = targets.to(ptu.device)
        cur_observation = state2ob(cur_state)
        n_observation = self.normalize_obs(cur_observation)
        
        loss_name = ['direction', 'speed', 'fall_down', 'acs']
        loss_num = len(loss_name)
        loss = [[] for i in range(loss_num)]
        
        # speed = np.random.choice(self.env.speed_range, targets[:,0,0].shape)
        # targets[:,:,0] = ptu.from_numpy(speed)[:,None]
        for i in range(rollout_length):
            #synthetic step
            action, info = self.act_task(state = cur_state, target = targets[:,i], n_observation = n_observation)
            cur_state = self.world_model(cur_state, action, n_observation = n_observation)
            cur_observation = state2ob(cur_state)
            n_observation = self.normalize_obs(cur_observation)
            
            # cal_loss
            loss_tmp = self.get_loss(cur_state, targets[:,i])
            for j, value in enumerate(loss_tmp):
                loss[j].append(value)
            action_loss = torch.mean(info['offset']**2)
            loss[-1].append(action_loss)
        
        #optimizer step
        # weight = [1,1,100,20]
        weight = [1,0,100,20]
        loss_value = [sum(l)/rollout_length*weight[i] for i,l in enumerate(loss)]
        loss_value[0] = loss[0][-1]
        loss = sum(loss_value)
        
        
        self.high_level_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level.parameters(), 1)
        self.high_level_optim.step()
        
        # return
        res = {loss_name[i]: loss_value[i] for i in range(loss_num)}
        return res
    
    #------------------------------------------playing-------------------------------#
    def get_action(self, **obs_info):
        return self.act_task(**obs_info)
    
    def after_step(self, server_scene):
        super().after_step(server_scene)
        
        direction = self.env.target[1]
        pos = server_scene.character0.root_body_pos
        pos[1] = 0
        server_scene.characters[1].bodies[0].PositionNumpy = (pos)
        quat = (Rotation.from_rotvec([0,np.pi/2 - direction,0])).as_matrix().astype(np.float64)
        server_scene.characters[1].body_info.bodies[0].setRotationNumpy(quat.flatten())
        # FirstIndex = lambda a, val, tol: next(i for i, _ in enumerate(a) if np.isclose(_, val, tol))
        # idx = FirstIndex(self.speed_range, cur_info['speed'].item(), 1e-3)
        server_scene.characters[1].body_info.bodies[0].setLinearVel([0.6,0.6,0.6])
       
        
if __name__ == "__main__":
    if mpi_rank ==0:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', default=False,  action='store_true')
        parser.add_argument('--interaction', default=True,  action='store_true')
        parser.add_argument('--dance', default=False,  action='store_true')
        args = SpeedPlayground.build_arg(parser)
        args['experiment_name'] = 'speed_playground'
        import tkinter.filedialog as fd
        data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
    args = mpi_comm.bcast(None if mpi_rank!=0 else args, root=0)
    args['show'] = not args['train']
    # args['mode'] = 'drawstuff' if args['train'] else 'panda'
    args['mode'] = 'panda'
    if args['train']:
        args['interaction'] = False
    ptu.init_gpu(True)
    data_file = mpi_comm.bcast(None if mpi_rank!=0 else data_file, root=0)
    env = VCLODETrackEnv(**args)
    playground = SpeedPlayground(323,57,120,env,**args)
    
    
    if args['train']:
        # load controlvae
        super(SpeedPlayground, playground).try_load(data_file)    
        playground.save_before_train(args)
        playground.train_loop()
    else:
        playground.try_load(data_file)
        playground.run()