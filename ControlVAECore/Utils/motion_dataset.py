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
from motion_utils import *
from misc import add_to_list
import pytorch_utils as ptu

import VclSimuBackend
try:
    from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
    # from VclSimuBackend.ODESim.TargetPose import TargetPose
    # from VclSimuBackend.ODESim.ODECharacter import ODECharacter
    from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
except ImportError:
    BVHToTargetBase = VclSimuBackend.ODESim.BVHToTarget.BVHToTargetBase
    SetTargetToCharacter = VclSimuBackend.ODESim.TargetPose.SetTargetToCharacter


class MotionDataSet():
    def __init__(self, fps) -> None:
        """ We orgnaize motion capture data into pickle files

        Args:
            fps (int): target fps of downsampled bvh
        """        
        # init buffer
        self.state = None
        self.observation = None
        self.done = None
        
        # init fps
        self.fps = fps
    
    @property
    def stastics(self):
        obs_mean = np.mean(self.observation, axis=0)
        obs_std = np.std(self.observation, axis =0)
        obs_std[obs_std < 1e-1] = 0.1
        
        delta = self.observation[1:] - self.observation[:-1]
        _,_,vel, avel,_,_= decompose_obs(delta)
        num = delta.shape[0]
        delta = np.concatenate([vel.reshape(num,-1,3),avel.reshape(num,-1,3)], axis = -1)
        delta = delta.reshape(num,-1)
        delta_mean = np.mean(delta, axis = 0)
        delta_std = np.std(delta, axis = 0)
        delta_std[delta_std < 1e-1] = 0.1
        return {
            'obs_mean': obs_mean,
            'obs_std': obs_std,
            'delta_mean': delta_mean,
            'delta_std': delta_std
        }
    
    def add_bvh_with_character(self, name, character, flip = False):
        if flip:
            target = BVHToTargetBase(name, self.fps, character, flip = np.array([1,0,0])).init_target()
        else:
            target = BVHToTargetBase(name, self.fps, character).init_target()
        tarset : SetTargetToCharacter = SetTargetToCharacter(character, target)

        state, ob, done = [],[],[] 
        
        for i in range(target.num_frames):
            tarset.set_character_byframe(i)
            state_tmp = character_state(character)
            ob_tmp =  state2ob(torch.from_numpy(state_tmp)).numpy()
            done_tmp = (i == (target.num_frames -1))
            state.append(state_tmp[None,...])
            ob.append(ob_tmp.flatten()[None,...])
            done.append(np.array(done_tmp).reshape(1,1))

        self.state = add_to_list(state, self.state)
        self.observation = add_to_list(ob, self.observation)
        self.done = add_to_list(done, self.done)

         
    def add_folder_bvh(self, name, character, mirror_augment = True):
        """Add every bvh in a forlder into motion dataset

        Args:
            name (str): path of bvh folder
            character (ODECharacter): the character of ode
            mirror_augment (bool, optional): whether to use mirror augment. Defaults to True.
        """                
        for file in os.listdir(name):
            if '.bvh' in file:
                print(f'add {file}')
                self.add_bvh_with_character(os.path.join(name, file), character)
        if mirror_augment:
            for file in os.listdir(name):
                if '.bvh' in file:
                    print(f'add {file} flip')
                    self.add_bvh_with_character(os.path.join(name, file), character, flip = True)
    
     