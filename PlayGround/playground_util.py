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
import torch
import ControlVAECore.Utils.diff_quat as DiffRotation

def get_facing(state):
    assert state.shape[-1] == 13
    assert len(state.shape) == 2
    rot = state[:,3:7]
    direction = torch.zeros([rot.shape[0],3], dtype = rot.dtype, device = rot.device)
    direction[:,2] = 1
    
    facing_direction = DiffRotation.quat_apply(rot, direction)
    axis1 = facing_direction[:,0].view(-1,1)
    axis2 = facing_direction[:,2].view(-1,1)
    facing_direction = torch.cat([axis1, torch.zeros_like(axis1), axis2], dim = -1)
    facing_direction = facing_direction/ torch.linalg.norm(facing_direction, dim = -1, keepdim = True)
    return facing_direction

def get_root_facing(state):
    return get_facing(state[:,0])

def state2speed(state, mass):
    vel = state[...,:,7:10]
    com_vel = torch.einsum('bnk,n->bk', vel, mass)
    return com_vel