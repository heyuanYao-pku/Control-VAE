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
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Geom, GeomNode
import direct.showutil.BuildGeometry as geom_util
from math import pi, sin, cos
from direct.task import Task
from scipy.spatial.transform import Rotation
from ControlVAECore.Env.vclode_track_env import VCLODETrackEnv
import numpy as np
from panda3d.core import LQuaternionf,LVecBase3
from Panda3dCameraCtrl import CameraCtrl
from panda3d.core import ClockObject
import panda3d.core as pc

class PandaServerBase(ShowBase):
    def __init__(self, fStartDirect=True, windowType=None):
        '''
        this is only used for my project... lots of assumptions...
        '''
        super().__init__(fStartDirect, windowType)
        self.disableMouse()        
        
        # if the model cannot be loaded, you need to pip install panda3d-gltf
        # and uncomment this:
        # gltf.patch_loader(self.loader)
        
        #self.model = self.loader.loadModel('misc/character.gltf')
        self.model = self.loader.loadModel('misc/character.bam')
        self.model.reparentTo(self.render)
        
        # self.camera.lookAt(0,0,0.9)
        self.setupCameraLight()
        self.cameractrl.center = self.model.find('**/pelvis').getPos()
        self.camera.setHpr(0,0,0)
        
        self.step_counter = 5000*6
        self.setFrameRateMeter(True)
        self.init_key_board()
        
        globalClock.setMode(ClockObject.MLimited)
        globalClock.setFrameRate(60)
        # self.clock.setDt(1/140)
        # self.oobe()
        
        self.load_ground()
        self.load_box()
        self.load_hugebox(1.0)
        self.root = self.model.find('**/DCharacter0')
        self.root.setPos(0,0,0)
        
        self._neg = [1, -1, -1, 1]
        self._perm = [3, 0, 1, 2]
        self._root_rot_corr = LQuaternionf(np.sqrt(2)/2,np.sqrt(2)/2,0,0)
        self._cnt = 0
        
        charaMaterial = pc.Material('charaMaterial')
        charaMaterial.setDiffuse((0.8,0.9,0.8, 1))
        
        for mat in self.root.findAllMaterials():
            self.root.replaceMaterial(mat, charaMaterial)
        
        #self.model1 = self.loader.loadModel('misc/character.gltf')
        self.model1 = self.loader.loadModel('misc/character.bam')
        self.root1 =  self.model1.find('**/DCharacter0')
        charaMaterial1 = pc.Material('charaMaterial1')
        charaMaterial1.setDiffuse((0.3,0.4, 1.0, 1))        
        for mat in self.root1.findAllMaterials():
            self.root1.replaceMaterial(mat, charaMaterial1)
        self.model1.reparentTo(self.render)
        
        # self.sky = self.loader.load_model("./misc/skybox.bam")
        # self.sky.reparentTo(self.render)
        # self.sky.setScale(100,100,100)
        # self.sky.set_color(135/255,206/255,235/255)
        
        xSize = self.pipe.getDisplayWidth()
        ySize = self.pipe.getDisplayHeight()
        props = pc.WindowProperties()
        props.setSize(max(xSize-200, 800), max(ySize-200, 600))
        self.win.requestProperties(props)
        
        
    def load_box(self, radius = 0.5):
        self.box = self.loader.loadModel("./misc/GroundScene.egg")
        self.box.reparentTo(self.render)
        self.box.setScale(radius/2, radius/2, radius/2)
        self.box.setTexScale(pc.TextureStage.getDefault(), radius/4, radius/4)
        self.box.setPos(0, 0, 1)
        self.box.set_color(0.5,0.5,0)
        
    def load_hugebox(self, radius = 1.0):
        self.hugebox = self.loader.loadModel("./misc/GroundScene.egg")
        self.hugebox.reparentTo(self.render)
        self.hugebox.setScale(radius/2, radius/2, radius/2)
        self.hugebox.setTexScale(pc.TextureStage.getDefault(), radius/4, radius/4)
        self.hugebox.setPos(0, 0, 2)
        self.hugebox.set_color(0.5,0.5,0)
    
    def load_ground(self):
        self.ground = self.loader.loadModel("./misc/GroundScene.egg")
        self.ground.reparentTo(self.render)
        self.ground.setScale(100, 100, 1)
        self.ground.setTexScale(pc.TextureStage.getDefault(), 50, 50)
        self.ground.setPos(0, 0, -1)
    
    def load_box_state(self, box_body, name = 'box'):
        pos = box_body.PositionNumpy
        rot = box_body.getQuaternionScipy()                
        pos = LVecBase3(-pos[0], -pos[2], pos[1])
        rot = LQuaternionf(rot[3], rot[0], rot[2], -rot[1])        
        if name == 'box':
            self.box.setPosQuat(pos, rot)
        else:
            self.hugebox.setPosQuat(pos, rot)
            
    def load_box_state_bk(self, box_body, name = 'box'):
        pos = box_body.PositionNumpy
        rot = box_body.getQuaternionScipy()
        rot = rot[...,[3,0,1,2]]
        if name =='box':
            self.box.setPosQuat(self.root, LVecBase3(*pos)*100, LQuaternionf(*rot))
        else:
            self.hugebox.setPosQuat(self.root, LVecBase3(*pos)*100, LQuaternionf(*rot))
            
    def update_state(self):
        info = self.playground.update()           
        self.load_character(info['character'])
        
    def load_character_state_to_model(self, character, model_root):
        joint_names = character.get_joint_names()
        nodes = [model_root.find('**/'+i) for i in joint_names]
        
        rot = character.joint_info.get_local_q()
                
        # self._cnt += 1
        # if self._cnt % 100 == 0:
        #     cnt = self._cnt // 100
        #     self._neg 
        #     self._perm
        #     _perms = [
        #         [3,0,1,2],
        #         [3,0,2,1],
        #         [3,1,0,2],
        #         [3,1,2,0],
        #         [3,2,1,0],
        #         [3,2,0,1],]
        #     _negs = [
        #         [1,1,1,1],
        #         [-1,1,1,1],
        #         [1,-1,1,1],
        #         [1,1,-1,1],
        #         [1,-1,-1,1],
        #         [-1,1,-1,1],
        #         [-1,-1,1,1],
        #         ]
            
        #     self._neg = _negs[cnt % len(_negs)]
        #     self._perm = _perms[(cnt // len(_negs))%(len(_perms))]
        #     print(self._neg)
        #     print(self._perm)
            
        
        rot *= self._neg 
        rot = rot[...,self._perm]
        
        for node, quat in zip(nodes, rot):
            node.setQuat(LQuaternionf(*quat))
                    
        # root
        rot = character.get_body_quat_at(0)
        rot *= self._neg
        rot = rot[self._perm]
        rot = LQuaternionf(*rot)
        rot *= self._root_rot_corr
        model_root.setQuat(rot)
        
        pos = character.get_body_pos_at(0)
        model_root.setPos(-pos[0], -pos[2], pos[1])
        
        # others
    
    def load_state(self, character):        
        self.load_character_state_to_model(character, self.root)
        
        pos = self.root.getPos()
        pos[2] = 1.1
        self.cameraRefNode.setPos(pos)
        
        return 
        x = character.get_body_name_list()
        nodes = [self.root.find('**/'+i) for i in x]
        pos = character.get_body_pos()
        rot = character.get_body_quat()
        # pos[:,0] *=-1
        # pos = pos[:,[0,2,1]]
        
        # q = Rotation.from_quat(rot)
        # q = q * Rotation.from_rotvec([0,np.pi,0])
        # rot = q.as_quat()
        
        # rot[:,0] *= -1
        rot = rot[:,[3,0,1,2]]
        
        for node,tmp_pos,quat in zip(nodes, pos, rot):
            node.setPosQuat(self.root, LVecBase3(*tmp_pos*100), LQuaternionf(*quat))

        pos = character.get_body_pos_at(0)
        pos = [pos[0], -pos[2], 1.1]
        
        self.cameraRefNode.setPos(*pos)
        pass

    def test_load(self, task):
        self.step_counter +=1
        self.env.reset(self.step_counter//3)
        self.load_state(self.env.sim_character)
        return Task.cont
    
    def handle_keyboard(self, task):
        if self.keys['escape']:
            self.close()
        
        if self.other_keyboard_handler:
            self.other_keyboard_handler(self, task)
        return Task.cont
    
    
    def keyboard_handler(self, key, state):
        if state == "press":
            if key == 'w':
                self.state_dict['x_axis'] = 1
            elif key == 's':
                self.state_dict['x_axis'] = -1
            elif key == 'a':
                self.state_dict['y_axis'] = 1
            elif key == 'd':
                self.state_dict['y_axis'] = -1
            self.state_dict['flag'] = 1
            
        elif state == "release":
            
            if key == 'w':
                self.state_dict['x_axis'] = 0
            elif key == 's':
                self.state_dict['x_axis'] = 0
            elif key == 'a':
                self.state_dict['y_axis'] = 0
            elif key == 'd':
                self.state_dict['y_axis'] = 0                
                
            self.state_dict['flag'] = 1
            
        if key in [ "1","2","3","4","5"]:
            self.state_dict['fix_speed'] = int(key[0])
        
        self.state_dict['speed'] = self.state_dict['fix_speed'] * self.state_dict['flag']
        return 
        
    def init_key_board(self):
        self.direction = None
        self.speed = 0
        self.state_dict = {
            'y_axis':0,
            'x_axis':0,
            'fix_speed':1,
            'flag':1,
            'speed':1
        } 
        
        self.accept("w-repeat", self.keyboard_handler,["w", "press"])
        self.accept("a-repeat", self.keyboard_handler,["a", "press"])
        self.accept("s-repeat", self.keyboard_handler,["s", "press"])
        self.accept("d-repeat", self.keyboard_handler,["d", "press"])
        
        self.accept("w-up", self.keyboard_handler,["w", "release"])
        self.accept("a-up", self.keyboard_handler,["a", "release"])
        self.accept("s-up", self.keyboard_handler,["s", "release"])
        self.accept("d-up", self.keyboard_handler,["d", "release"])

        # self.accept("0", self.keyboard_handler, ["0", "press"])
        self.accept("1", self.keyboard_handler, ["1", "None"])
        self.accept("2", self.keyboard_handler, ["2", "None"])
        self.accept("3", self.keyboard_handler, ["3", "None"])
        self.accept("4", self.keyboard_handler, ["4", "None"])
        self.accept("5", self.keyboard_handler, ["5", "None"])
        
    def setupCameraLight(self):
        # create a orbiting camera
        self.cameractrl = CameraCtrl(self, self.cam)
        self.cameraRefNode = self.camera # pc.NodePath('camera holder')
        self.cameraRefNode.setPos(0,0,0)
        self.cameraRefNode.setHpr(0,0,0)
        self.cameraRefNode.reparentTo(self.render)
        
        self.accept("v", self.bufferViewer.toggleEnable)

        self.d_lights = []
        # Create Ambient Light
        ambientLight = pc.AmbientLight('ambientLight')
        ambientLight.setColor((0.3, 0.3, 0.3, 1))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)
        
        # Directional light 01
        directionalLight = pc.DirectionalLight('directionalLight')
        directionalLight.setColor((0.4, 0.4, 0.4, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        # This light is facing backwards, towards the camera.
        directionalLightNP.setPos(10, 10, 10)
        directionalLightNP.lookAt((0, 0, 0), (0, 1, 0))

        directionalLightNP.wrtReparentTo(self.cameraRefNode)
        # directionalLight.setShadowCaster(True, 512, 512)
        # directionalLight.getLens().setFilmSize((3,3))
        # directionalLight.getLens().setNearFar(0.1,300)
        
        self.render.setLight(directionalLightNP)
        self.d_lights.append(directionalLightNP)
        
        # Directional light 02
        directionalLight = pc.DirectionalLight('directionalLight1')
        # directionalLight.setColorTemperature(6500)        
        directionalLight.setColor((0.4, 0.4, 0.4, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        # This light is facing forwards, away from the camera.
        directionalLightNP.setPos(-10, 10, 10)
        directionalLightNP.lookAt((0, 0, 0), (0, 1, 0))

        directionalLightNP.wrtReparentTo(self.cameraRefNode)
        # directionalLight.setShadowCaster(True, 512, 512)
        # directionalLight.getLens().setFilmSize((3,3))
        # directionalLight.getLens().setNearFar(0.1,300)
        
        self.render.setLight(directionalLightNP)
        self.d_lights.append(directionalLightNP)
        
        
        # Directional light 03
        directionalLight = pc.DirectionalLight('directionalLight1')
        directionalLight.setColorTemperature(6500)        
        # directionalLight.setColor((0.6, 0.6, 0.6, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        # This light is facing forwards, away from the camera.
        directionalLightNP.setPos(0, -10, 20)
        directionalLightNP.lookAt((0, 0, 0), (0, 1, 0))

        directionalLightNP.wrtReparentTo(self.cameraRefNode)
        directionalLight.setShadowCaster(True, 2048, 2048)
        directionalLight.getLens().setFilmSize((10,10))
        directionalLight.getLens().setNearFar(0.1,300)
        
        self.render.setLight(directionalLightNP)
        self.d_lights.append(directionalLightNP)

        self.render.setShaderAuto(True)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    VCLODETrackEnv.add_specific_args(parser)
    args = vars(parser.parse_args())
    args['motion_dataset'] = r'Data\ReferenceData\binary_data\runwalkjumpgetup.pickle'
    args['env_scene_fname'] = r'odecharacter_scene.pickle'
    env = VCLODETrackEnv(**args)
    
    # from VclSimuBackend.Render import Renderer
    # renderObj = Renderer.RenderWorld(env.scene.world)
    # renderObj.start()
    # renderObj.look_at([3,3,3], [0,1,0], [0,1,0])
    # renderObj.track_body(env.sim_character.bodies[0], False)
    
    server =  PandaServerBase()
    server.env = env
    server.taskMgr.add(server.test_load, 'testload')
        
    # movie_task = server.movie('test')
    # server.taskMgr.add(movie_task, 'record')
    server.run()