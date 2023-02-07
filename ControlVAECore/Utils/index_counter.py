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
class index_counter():
    def __init__(self, done_flag) -> None:
        self.done_flag = done_flag
        self.cur_frame = 0
    
    @staticmethod
    def sample_rollout(feasible_index, batch_size, rollout_length):
        """generate index for rollout sampling

        Args:
            feasible_index (np.ndarray): please make sure [i,i+rollout_length) is useful
            batch_size (int): nop
            rollout_length (int): nop
        """
        begin_idx = np.random.choice(feasible_index.flatten(), [batch_size,1])
        bias = np.arange(rollout_length).reshape(1,-1)
        res_idx = begin_idx + bias
        return res_idx
    
    @staticmethod
    def calculate_feasible_index(done_flag, rollout_length):
        res_flag = np.ones_like(done_flag).astype(int)
        terminate_idx = np.where(done_flag!=0)[0].reshape(-1,1)
        bias = np.arange(rollout_length).reshape(1,-1)
        terminate_idx = terminate_idx - bias
        res_flag[terminate_idx.flatten()] = 0
        return np.where(res_flag)[0]
    
    @staticmethod
    def random_select(feasible_index, p = None):
        return np.random.choice(feasible_index, p = p)