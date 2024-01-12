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
import pickle
import numpy as np
import torch
from ..Utils.motion_dataset import MotionDataSet
from scipy.spatial.transform import Rotation
try:
    from VclSimuBackend import SetInitSeed
except:
    from ModifyODE import SetInitSeed
from ..Utils.motion_utils import character_state, state2ob, state_to_BodyInfoState
from ..Utils.index_counter import index_counter


import VclSimuBackend
try:
    from VclSimuBackend.Common.MathHelper import MathHelper
    from VclSimuBackend.ODESim.Saver import CharacterToBVH
    from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
    from VclSimuBackend.ODESim.PDControler import DampedPDControler
except ImportError:
    MathHelper = VclSimuBackend.Common.MathHelper
    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    JsonSceneLoader = VclSimuBackend.ODESim.JsonSceneLoader
    DampedPDControler = VclSimuBackend.ODESim.PDController.DampedPDControler


class VCLODETrackEnv():
    """A tracking environment, also performs as a environment base... because we need a step function.
    it contains:
        a ODEScene: physics environment which contains two characters, one for simulation another for kinematic
        a MotionDataset: contains target states and observations of reference motion
        a counter: represents the number of current target pose
        a step_cnt: represents the trajectory length, will be set to zero when env is reset
    """
    def __init__(self, **kargs) -> None:
        super(VCLODETrackEnv, self).__init__()
        self.object_reset(**kargs)
    
    def object_reset(self, **kargs):
        if 'seed' in kargs:
            self.seed(kargs['seed']) # using global_seed    
        
        # init scene
        scene_fname = kargs['env_scene_fname']
        SceneLoader = JsonSceneLoader()
        self.scene = SceneLoader.file_load(scene_fname)
        # self.scene = SceneLoader.load_from_pickle_file(scene_fname)
        
        # some configuration of scene
        self.scene.characters[1].is_enable = False
        self.scene.contact_type = kargs['env_contact_type']
        self.scene.self_collision = not kargs['env_close_self_collision']
        
        self.fps = kargs['env_fps']
        self.dt = 1/self.fps
        self.substep = kargs['env_substep']
        
        # terminate condition
        self.min_length = kargs['env_min_length']
        self.max_length = kargs['env_max_length']
        self.err_threshod = kargs['env_err_threshod']
        self.err_length = kargs['env_err_length']
        # functional element
        self.stable_pd = DampedPDControler(self.sim_character)
        name_list = self.sim_character.body_info.get_name_list()
        self.head_idx = name_list.index('head')
        self.balance = not kargs['env_no_balance']
        self.use_com_height = kargs['env_use_com_height']
        self.recompute_velocity = kargs['env_recompute_velocity']
        self.random_count = kargs['env_random_count']
        
        self.motion_data : MotionDataSet = pickle.load(open(kargs['motion_dataset'],'rb'))
        self.frame_num = self.motion_data.state.shape[0]
        self.init_index = index_counter.calculate_feasible_index(self.motion_data.done,2) 
        
        if self.balance:
            self.val = np.zeros(self.frame_num)    
            self.update_p()
        return 
        
    @property
    def stastics(self):
        return self.motion_data.stastics
    
    @property
    def sim_character(self):
        return self.scene.characters[0]

    @property
    def ref_character(self):
        return self.scene.characters[1]
    
    def get_bvh_saver(self):
        bvh_saver = VclSimuBackend.ODESim.CharacterTOBVH(self.sim_character, self.fps)
        bvh_saver.bvh_hierarchy_no_root()
        return bvh_saver

    @staticmethod
    def seed( seed):
        SetInitSeed(seed)
    
    @staticmethod
    def add_specific_args(arg_parser):
        arg_parser.add_argument("--env_contact_type", type=int, default=0, help="contact type, 0 for LCP and 1 for maxforce")
        arg_parser.add_argument("--env_close_self_collision", default=False, help="flag for closing self collision", action = 'store_true')
        arg_parser.add_argument("--env_min_length", type=int, default=26, help="episode won't terminate if length is less than this")
        arg_parser.add_argument("--env_max_length", type=int, default=512, help="episode will terminate if length reach this")
        arg_parser.add_argument("--env_err_threshod", type = float, default = 0.5, help="height error threshod between simulated and tracking character")
        arg_parser.add_argument("--env_err_length", type = int, default = 20, help="episode will terminate if error accumulated ")
        arg_parser.add_argument("--env_scene_fname", type = str, default = "character_scene.pickle", help="pickle file for scene")
        arg_parser.add_argument("--env_fps", type = int, default = 20, help="fps of control policy")
        arg_parser.add_argument("--env_substep", type = int, default = 6, help="substeps for simulation")
        arg_parser.add_argument("--env_no_balance", default = False, help="whether to use distribution balance when choose initial state", action = 'store_true')
        arg_parser.add_argument("--env_use_com_height", default = False, help="if true, calculate com in terminate condition, else use head height", action = 'store_true')
        arg_parser.add_argument("--motion_dataset", type = str, default = None, help="path to motion dataset")
        arg_parser.add_argument("--env_recompute_velocity", default = True, help = "whether to resample velocity")
        arg_parser.add_argument("--env_random_count", type=int, default=96, help = "target will be random switched for every 96 frame")
        return arg_parser
    
    def update_and_get_target(self):
        self.step_cur_frame()
        return 
    
    @staticmethod
    def isnotfinite(arr):
        res = np.isfinite(arr)
        return not np.all(res)

    def cal_done(self, state, obs):
        height = state[...,self.head_idx,1]
        target_height = self.motion_data.state[self.counter][self.head_idx,1]
        if abs(height - target_height) > self.err_threshod:
            self.done_cnt +=1
        else:
            self.done_cnt = max(0, self.done_cnt - 1)
        
        if self.isnotfinite(state):
            return 2
        
        if np.any( np.abs(obs) > 50):
            return 2
        
        if self.step_cnt >= self.min_length:
            if self.done_cnt >= self.err_length:
                return 2
            if self.step_cnt >= self.max_length:
                return 1
        return 0
    
    def update_val(self, done, rwd, frame_num):
        tmp_val = self.val / 2
        last_i = 0
        for i in range(frame_num.shape[0]):
            if done[i] !=0:
                tmp_val[frame_num[i]] = rwd[i] if done[i] == 1 else 0
                for j in reversed(range(last_i, i)):
                    tmp_val[frame_num[j]] = 0.95*tmp_val[frame_num[j+1]] + rwd[j]
                last_i = i
        self.val = 0.9 * self.val + 0.1 * tmp_val
        return
        
    def update_p(self):
        self.p = 1/ self.val.clip(min = 0.01)
        self.p_init = self.p[self.init_index]
        self.p_init /= np.sum(self.p_init)
        
    def get_info(self):
        return {
            'frame_num': self.counter
        }
    
    def get_target(self):
        return self.motion_data.observation[self.counter]
    
    @staticmethod
    def load_character_state(character, state):
        character.load(state_to_BodyInfoState(state))
        aabb = character.get_aabb() # prevent under the floor....
        if aabb[2]<0:
            character.move_character_by_delta(np.array([0,-aabb[2]+1e-8,0]))
    
    def step_counter(self, random = False):
        self.counter += 1
        if self.motion_data.done[self.counter] or random:    
            self.counter = index_counter.random_select(self.init_index, p = self.p_init)
        
    def reset(self, frame = -1, set_state = True ):
        """reset enviroment

        Args:
            frame (int, optional): if not -1, reset character to this frame, target to next frame. Defaults to -1.
            set_state (bool, optional): if false, enviroment will only reset target, not character. Defaults to True.
        """
        self.counter = frame
        if frame == -1:
             self.step_counter(random=True)
        if set_state:
            self.load_character_state(self.sim_character, self.motion_data.state[self.counter])        
        self.state = character_state(self.sim_character)
        self.observation = state2ob(torch.from_numpy(self.state)).numpy()
        
        info = self.get_info()
        
        self.step_counter(random = False)
        self.step_cnt = 0
        self.done_cnt = 0
        return {
            'state': self.state,
            'observation': self.observation,
            'target': self.get_target()
        }, info

    def after_step(self, **kargs):
        """usually used to update target...
        """
        self.step_counter(random = (self.step_cnt % self.random_count)==0 and self.step_cnt!=0)
        self.target = self.get_target()
        self.step_cnt += 1
        
    
    def step_core(self, action, using_yield = False, **kargs):
        
        action = Rotation.from_rotvec(action.reshape(-1,3)).as_quat()
        action = MathHelper.flip_quat_by_w(action)

        for i in range(self.substep):
            self.stable_pd.add_torques_by_quat(action)
            if 'force' in kargs:
                self.add_force(kargs['force'])
            self.scene.damped_simulate(1)

            if using_yield:
                yield self.sim_character.save()
        
        self.state = character_state(self.sim_character, self.state if self.recompute_velocity else None, self.dt)
        self.observation = state2ob(torch.from_numpy(self.state)).numpy()
        reward = 0 
        done = self.cal_done(self.state, self.observation)
        info = self.get_info()
        self.after_step()
        
        observation = {
            'state': self.state,
            'target': self.target,
            'observation': self.observation
        }

        if not using_yield: # for convenient, so that we do not have to capture exception
            yield observation, reward, done, info
        else:
            return observation, reward, done, info  
    
    def step(self, action, **kargs):
        step_generator = self.step_core(action, **kargs)
        return next(step_generator)
    