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
from ControlVAECore.Model.controlvae import ControlVAE
from ControlVAECore.Utils.misc import load_data, load_yaml
import ControlVAECore.Utils.pytorch_utils as ptu
from ControlVAECore.Env.vclode_track_env import VCLODETrackEnv
import argparse
from direct.task import Task
import time
import numpy as np
class RandomPlayground(ControlVAE):
    def __init__(self, observation_size, action_size, delta_size, env, **kargs):
        super(RandomPlayground, self).__init__(observation_size, action_size, delta_size, env, **kargs)
        self.mode = kargs['mode']
        self.show = kargs['show']
        
        self.observation = self.env.reset()[0]
        self.step_generator = None
        self.other_objects = {}

        self.add_box()    
        self.add_box(radius = 1, name = 'hugebox')
        
        self.init_viewer()
        self.cnt = 0
        self.env.reset()
        
    def init_viewer(self):
        if self.show:
            if self.mode == 'panda':
                '''
                use panda3d as viewer
                '''
                from PlayGround.panda_server_base import PandaServerBase
                self.panda_server = PandaServerBase()
            elif self.mode == 'drawstuff':
                '''
                use default viewer
                '''
                try:
                    from VclSimuBackend.Render import Renderer
                except ImportError:
                    import VclSimuBackend
                    Renderer = VclSimuBackend.Render
                self.renderObj = Renderer.RenderWorld(self.env.scene.world)
                self.renderObj.start()
                self.renderObj.track_body(self.env.sim_character.bodies[0], False)
            elif self.mode == 'unity':
                '''
                use unity as viewer
                '''
                from PlayGround.unity_server_base import UnityServer
                self.unity_server = UnityServer(self.env, self)

    def add_box(self, radius = 0.5, name = 'box'):
        import VclSimuBackend as ode
        # import ModifyODE as ode
        import numpy as np
        space, world = self.env.scene.space, self.env.scene.world
        box = ode.GeomBox(space, [radius]*3)
        body = ode.Body(world)
        box.body = body
        mass = ode.Mass()
        mass.setBox(100,radius,radius,radius)
        body.setMass(mass)
        self.other_objects[name] = body
        body.PositionNumpy = np.array([100.0,1.0,0.0])
        box.character_id = -2 - np.random.randint(0,100)
    
        
        
    def throw_box_to_character(self, name = 'box', n_delta = None):
        box = self.other_objects[name]
        character_pos = self.env.state[0][:3]
        # delta = character_pos.flatten() -  box.PositionNumpy.flatten()
        if n_delta is None:
            delta = np.random.randn(3)
            n_delta = delta / (delta**2).sum().clip(min = 1.0)
        box_pos = character_pos.flatten() - 3*n_delta
        box_pos[1] = max(0.3, box_pos[1])
        box.PositionNumpy = box_pos
        box.setLinearVel(10*n_delta)
        
    def throw_huge_box_to_character(self):
        self.throw_box_to_character('hugebox', n_delta=np.array((0,-1,0)))
    
    def get_action(self, **obs_info):
        n_observation = self.obsinfo2n_obs(obs_info)
        latent, mu, logvar  = self.encoder.encode_prior(n_observation)
        action = self.decode(n_observation, latent)
        info = {'mu': mu, 'logvar': logvar}
        return action, info
    
    def get_generator(self):
        action, info = self.get_action(**self.observation)
        action = ptu.to_numpy(action)
        return self.env.step_core(action, using_yield = True)
    
    def yield_step(self):
        if self.step_generator is None:
            self.step_generator = self.get_generator()             
        try:
            self.step_generator.__next__()
        except StopIteration as e:
            self.observation = e.value[0]
            self.step_generator = self.get_generator() 
            self.env.load_character_state(self.env.ref_character, self.env.motion_data.state[self.env.counter])
            self.step_generator.__next__()
    def post_step(self, observation):
        pass
    
    def update_panda(self, task):
        self.yield_step()
        self.yield_step() # because panda lock 60 fps...
        # if self.cnt % 900 ==0:
        #     # pass
        #     self.throw_box_to_character()
        self.panda_server.load_state(self.env.sim_character) # actually it just load pos, rot, so needn't resample vel
        self.panda_server.load_box_state(self.other_objects['box'])
        self.panda_server.load_box_state(self.other_objects['hugebox'], name = 'hugebox')
        self.cnt +=2
        return Task.cont
    
    def run(self):
        self.env.reset()
        if self.mode == 'panda':
            self.panda_server.taskMgr.add(self.update_panda, 'update_panda')            
            self.panda_server.accept('b', self.throw_huge_box_to_character)
            self.panda_server.accept('space', self.throw_box_to_character)
            
            self.panda_server.run()
            
        elif self.mode == 'drawstuff':
            cnt = 0
            while True:
                if cnt %1200 == 0:
                    self.throw_box_to_character()
                self.yield_step()
                time.sleep(1/120)
                self.post_step(self.observation)
                cnt +=1
        elif self.mode == 'unity':
            self.unity_server.run()
    
    def after_step(self, server_scene):
        character = server_scene.characters[1]
        box_body = character.bodies[1]
        box_body.PositionNumpy = self.other_objects['box'].PositionNumpy
        box_body.setQuaternionScipy( self.other_objects['box'].getQuaternionScipy())
        
        box_body = character.bodies[0]
        box_body.PositionNumpy = self.other_objects['hugebox'].PositionNumpy
        box_body.setQuaternionScipy( self.other_objects['hugebox'].getQuaternionScipy())
        
        pass
    
    def unity_step(self, server_scene, info):
        if 'throw_box' in info and info['throw_box']:
            self.throw_box_to_character()
        if 'x' in info and abs(info['x'])>0.1:
            self.throw_huge_box_to_character()
            
        self.yield_step()
        server_scene.character0.load(self.env.sim_character.save())
        self.after_step(server_scene)
      
    @staticmethod
    def build_arg(parser = None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('--mode', default = 'panda', type = str)
        parser.add_argument('--show', default = False, action='store_true')
        parser = VCLODETrackEnv.add_specific_args(parser)
        parser = ControlVAE.add_specific_args(parser)
        args = vars(parser.parse_args())
        ptu.init_gpu(True)
        # con
        config = load_yaml(initialdir ='Data/NNModel/Pretrained')
        args.update(config)

        return args


if __name__ == '__main__':
    # args
    args = RandomPlayground.build_arg()
    args['show'] = True
    env = VCLODETrackEnv(**args)
    playground = RandomPlayground(323, 57, 120, env, **args)
    # load
    import tkinter.filedialog as fd
    data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
    playground.try_load(data_file)
    playground.run()