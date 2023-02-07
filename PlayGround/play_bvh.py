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
import atexit
import numpy as np
import os
import pickle

import VclSimuBackend
try:
    from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
    from VclSimuBackend.ODESim.PDControler import DampedPDControler
    from VclSimuBackend.ODESim.ODECharacter import ODECharacter
    from VclSimuBackend.Render import Renderer
    from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
    from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
except ImportError:
    JsonSceneLoader = VclSimuBackend.ODESim.JsonSceneLoader
    JsonCharacterLoader = VclSimuBackend.ODESim.JsonCharacterLoader
    DampedPDControler = VclSimuBackend.ODESim.PDController.DampedPDControler
    ODECharacter = VclSimuBackend.ODESim.ODECharacter
    Renderer = VclSimuBackend.Render
    BVHToTargetBase = VclSimuBackend.ODESim.BVHToTarget.BVHToTargetBase
    SetTargetToCharacter = VclSimuBackend.ODESim.TargetPose.SetTargetToCharacter
    

fdir = os.path.dirname(__file__)
scene_fname = r'Data\Misc\world.json'
bvh_fname = r'Data\ReferenceData\run\WalkF-mocap.bvh'
SceneLoader = JsonSceneLoader()
scene = SceneLoader.file_load(scene_fname)
character: ODECharacter = scene.character0
character.set_render_color(np.array([1.0, 0.0, 0.0], dtype=np.float64))
bvh2target = BVHToTargetBase(bvh_fname, 120, scene.character0)
target = bvh2target.init_target()
set_target = SetTargetToCharacter(scene.character0, target)
stable_pd = DampedPDControler(scene.character0)

# add duplicated character..
# with open(scene_fname, "rb") as fin:
#     character_dict = pickle.load(fin)["CharacterList"]["Characters"][0]
#     new_character = JsonCharacterLoader(scene.world, scene.space).load(character_dict)
#     scene.characters.append(new_character)

i = 0
set_target.set_character_byframe(0)
bvh_cnt = bvh2target.frame_cnt
renderObj = Renderer.RenderWorld(scene.world)
renderObj.set_joint_radius(0.05)
renderObj.start()
# renderObj.set_color([1.0, 0.0, 0.0])

def exit_func():
    renderObj.kill()
atexit.register(exit_func)

for i in range(bvh_cnt):
    set_target.set_character_byframe(i)
    renderObj.pause(5)
    i = (i + 1) % bvh_cnt
    # print('frame={}'.format(i), end='\r')
