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
import sys
import numpy as np
import os
from typing import Any, Optional, Tuple
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.autograd import Variable, Function


fdir = os.path.dirname(__file__)
sys.path.append(os.path.join(fdir, ".."))
cpu_device = torch.device("cpu")
import VclSimuBackend
try:
    from VclSimuBackend.Common.MathHelper import RotateType
except:
    RotateType = VclSimuBackend.Common.RotateType

fdir = os.path.dirname(__file__)
cpu_device = torch.device("cpu")


class CatWithMask(Function):
    """
    This is just like:
    res = torch.zero() # x1.shape + x2.shape
    res[idx1] = x1
    res[~idx1] = x2
    idx1 is bool Tensor for x1 in res, and ~idx is bool Tensor for x2 in res

    TODO: support Tensor in dimension more than 2
    """
    @staticmethod
    def forward(ctx, *args: Any, **kwargs: Any) -> Any:
        """
        params:
        x1, torch.Tensor
        x2, torch.Tensor
        idx1: torch.Tensor with type torch.bool. shape[0] == x1.shape[0] + x2.shape[0]
        """
        x1, x2, idx1 = args
        ctx.save_for_backward(idx1)
        shape = idx1.shape if x1.ndim == 1 else (idx1.shape[0:1] + x1.shape[1:])
        result: torch.Tensor = torch.zeros(shape, dtype=x1.dtype, device=x1.device)
        result[idx1, ...] = x1
        result[~idx1, ...] = x2
        result.requires_grad = x1.requires_grad or x2.requires_grad
        return result

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[torch.Tensor, torch.Tensor, None]:
        grad: torch.Tensor = grad_outputs[0]
        # assert torch.sum(torch.isnan(grad)) == 0
        idx1: torch.Tensor = ctx.saved_tensors[0]  # idx1 means x1, and ~idx1 means x2
        return grad[idx1], grad[~idx1], None
        # for x1, the grad is 1. for x2, the grad is 1. for idx1, the grad is None


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    assert q.shape[-1] == 4
    ret: torch.Tensor = torch.cat([-1 * q[..., :3], q[..., 3:4]], dim=-1)
    return ret


def quat_multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    multiply 2 quaternions. p.shape == q.shape
    """
    assert len(p.shape) == 2 and p.shape[-1] == 4
    assert len(q.shape) == 2 and q.shape[-1] == 4

    w: torch.Tensor = p[:, 3:4] * q[:, 3:4] - torch.sum(p[:, :3] * q[:, :3], dim=1, keepdim=True)
    xyz: torch.Tensor = (
                p[:, None, 3] * q[:, :3] + q[:, None, 3] * p[:, :3] + torch.cross(p[:, :3], q[:, :3], dim=1))

    return torch.cat([xyz, w], dim=-1)
    


def quat_multiply_imp2(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    time usage is almost same with quat_multiply
    """
    p_xyz: torch.Tensor = p[:, :3]
    p_w: torch.Tensor = p[:, 3:4]

    q_xyz: torch.Tensor = q[:, :3]
    q_w: torch.Tensor = q[:, 3:4]

    w: torch.Tensor = p_w * q_w - torch.sum(p_xyz * q_xyz, dim=1, keepdim=True)
    xyz: torch.Tensor = (p_w * q_xyz + q_w * p_xyz + torch.cross(p_xyz, q_xyz, dim=1))

    return torch.cat([xyz, w], dim=-1)


def quat_multiply_imp3(q: torch.Tensor, r: torch.Tensor):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).

    Note: this version is much slower than quat_multiply
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    original_shape = q.shape

    # Compute outer product
    terms: torch.Tensor = torch.bmm(q.view(-1, 4, 1), r.view(-1, 1, 4))

    x: torch.Tensor = + terms[:, 3, 0] - terms[:, 2, 1] + terms[:, 1, 2] + terms[:, 0, 3]  # check ok
    y: torch.Tensor = + terms[:, 2, 0] + terms[:, 3, 1] - terms[:, 0, 2] + terms[:, 1, 3]  # check ok
    z: torch.Tensor = - terms[:, 1, 0] + terms[:, 0, 1] + terms[:, 3, 2] + terms[:, 2, 3]  # check ok
    w: torch.Tensor = - terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] + terms[:, 3, 3]  # check ok
    return torch.stack((x, y, z, w), dim=1).view(original_shape)


def quat_multiply_imp4(q: torch.Tensor, r: torch.Tensor):
    idx = torch.as_tensor(
        [[12, 9, 6, 3],
         [8, 13, 2, 7],
         [4, 1, 14, 11],
         [0, 5, 10, 15]]
    )
    fac = torch.as_tensor(
        [-1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1], dtype=q.dtype
    ).view(16)
    # terms = torch.outer(q,r).view(-1,16)
    terms: torch.Tensor = torch.bmm(q.view(-1, 4, 1), r.view(-1, 1, 4)).view(-1, 16)
    flatten_terms = terms * fac
    res = torch.matmul(flatten_terms[..., idx], torch.as_tensor([1, 1, 1, 1], dtype=q.dtype))
    return res.view(q.shape)


def quat_apply_ori(q: torch.Tensor, vec3: torch.Tensor) -> torch.Tensor:
    """
    param:
    q: quaternion in shape (n, 4)
    vec3: vector in shape (n, 3)
    return rotate result q * vec3 * q^{-1}
    """
    assert q.shape[-1] == 4 and vec3.shape[-1] == 3
    if vec3.shape == (3,):
        vec3 = vec3[None, :]
    ext_vec = torch.cat([vec3, torch.zeros(vec3.shape[:-1] + (1,), dtype=vec3.dtype, device=vec3.device)], dim=-1)
    tmp: torch.Tensor = quat_multiply(q, ext_vec)  # shape = (num quat, 4)
    tmp2: torch.Tensor = quat_multiply(tmp, quat_inv(q))  # shape == (num quat, 4)
    # TODO: why should add minus here...?
    return -tmp2[..., :3]


def quat_apply(q: torch.Tensor, vec3: torch.Tensor) -> torch.Tensor:
    """
    param:
    q: quaternion in shape (n, 4)
    vec3: vector in shape (n, 3)
    return rotate result q * vec3 * q^{-1}
    """
    assert q.shape[-1] == 4 and vec3.shape[-1] == 3
    if vec3.shape == (3,):
        vec3 = vec3[None, :]
    
    t = 2 * torch.cross(q[:, :3], vec3, dim=1)
    xyz: torch.Tensor = vec3 + q[:, 3, None] * t + torch.cross(q[:, :3], t, dim=1)
    # if not torch.allclose(xyz, quat_apply_ori(q, vec3), atol = 1e-15):
    #     print(torch.max(torch.abs(xyz - quat_apply_ori(q,vec3))))
    #     print(q.shape, torch.mean(torch.abs(vec3)))
    return xyz


quat_apply_imp1 = quat_apply


# def quat_apply_imp1(q: torch.Tensor, vec3: torch.Tensor) -> torch.Tensor:
#     """
#     param:
#     q: quaternion in shape (n, 4)
#     vec3: vector in shape (n, 3)
#     return rotate result q * vec3 * q^{-1}
#     """
#     assert q.shape[-1] == 4 and vec3.shape[-1] == 3
#     if vec3.shape == (3,):
#         vec3 = vec3[None, :]
#     t = 2*torch.cross(q[:, :3], vec3, dim = 1)
#     xyz: torch.Tensor = vec3 + q[:,3,None]*t  + torch.cross(q[:, :3], t, dim = 1)
#     # if not torch.allclose(xyz, quat_apply_ori(q, vec3), atol = 1e-15):
#     #     print(torch.max(torch.abs(xyz - quat_apply_ori(q,vec3))))
#     #     print(q.shape, torch.mean(torch.abs(vec3)))
#     return xyz


def quat_inv(q: torch.Tensor) -> torch.Tensor:  # TODO test
    """
    inverse of quaternions
    """
    w = -1 * q[..., -1:]
    xyz = q[..., :3]
    return torch.cat([xyz, w], dim=-1)
    # return torch.hstack([xyz, w])


def quat_angle(q: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sum(q[..., :3] ** 2, dim=-1), torch.abs(q[..., 3]))


def vec_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    normalize vectors at the last dimension
    """
    result = x / torch.linalg.norm(x, 2, -1, keepdim=True).clamp(1e-9)
    return result


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """
    normalize quaternion
    """
    return vec_normalize(q)


def quat_integrate(q: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
    """
    update quaternion, q_{t+1} = normalize(q_{t} + 0.5 * w * q_{t})
    """
    if q.shape[-1] == 1 and omega.shape[-1] == 1:
        q = q.view(q.shape[:-1])
        omega = omega.view(omega.shape[:-1])
    assert q.shape[-1] == 4 and omega.shape[-1] == 3

    
    omega = torch.cat([omega, torch.zeros(omega.shape[:-1] + (1,), dtype=omega.dtype, device=q.device)], -1)
    delta_q = 0.5 * dt * quat_multiply(omega, q)
    result = q + delta_q
    result = quat_normalize(result)
    
    return result.view(q.shape)


def quat_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # difference between 2 quaternions. ad=b, d=a^{-1} b
    a_inv = quat_inv(a)
    result = quat_multiply(a_inv, b)
    return result


def log_quat_diff_sqr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # TODO: test
    # log(a^{-1} * b)
    a_inv: torch.Tensor = quat_inv(a)
    a_inv_b: torch.Tensor = quat_multiply(a_inv, b)
    rotvec: torch.Tensor = quat_to_rotvec(a_inv_b)
    sqr_theta: torch.Tensor = torch.sum(rotvec ** 2, dim=-1)
    return sqr_theta


def log_quat_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sqr_theta: torch.Tensor = log_quat_diff_sqr(a, b)
    # assert torch.sum(torch.isnan(sqr_theta)) == 0
    theta: torch.Tensor = torch.sqrt(sqr_theta)
    # raise ValueError("for debug")
    return theta


def quat_to_matrix(q: torch.Tensor, do_normalize: bool = False) -> torch.Tensor:
    """
    Convert Quaternion to matrix. Note: q must be normalized before.
    Param: q: torch.Tensor in shape (*, 4)
    return: rotation matrix in shape (*, 3, 3)
    """
    origin_shape = q.shape
    q: torch.Tensor = q.view(-1, 4)
    if do_normalize:
        q = quat_normalize(q)

    
    x: torch.Tensor = q[..., 0]
    y: torch.Tensor = q[..., 1]
    z: torch.Tensor = q[..., 2]
    w: torch.Tensor = q[..., 3]

    x2: torch.Tensor = x ** 2
    y2: torch.Tensor = y ** 2
    z2: torch.Tensor = z ** 2
    w2: torch.Tensor = w ** 2

    xy: torch.Tensor = x * y
    zw: torch.Tensor = z * w
    xz: torch.Tensor = x * z
    yw: torch.Tensor = y * w
    yz: torch.Tensor = y * z
    xw: torch.Tensor = x * w

    res00: torch.Tensor = x2 - y2 - z2 + w2
    res10: torch.Tensor = 2 * (xy + zw)
    res20: torch.Tensor = 2 * (xz - yw)

    res01: torch.Tensor = 2 * (xy - zw)
    res11: torch.Tensor = - x2 + y2 - z2 + w2
    res21: torch.Tensor = 2 * (yz + xw)

    res02: torch.Tensor = 2 * (xz + yw)
    res12: torch.Tensor = 2 * (yz - xw)
    res22: torch.Tensor = - x2 - y2 + z2 + w2

    res: torch.Tensor = torch.vstack([res00, res01, res02, res10, res11, res12, res20, res21, res22]).T.view(
        origin_shape[:-1] + (3, 3))
    
    return res


class _QuatCatFunc(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any) -> torch.Tensor:
        qi, qj, qk, qw, i, j, k = args
        xyz = torch.zeros((qi.shape[0], 4), dtype=qi.dtype, device=qi.device)
        idx = torch.arange(end=len(i), dtype=torch.long)
        xyz[idx, i] = qi
        xyz[idx, j] = qj
        xyz[idx, k] = qk
        xyz[:, 3] = qw
        ctx.save_for_backward(i, j, k, idx)
        xyz.requires_grad = sum([node.requires_grad for node in args]) > 0
        return xyz

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_xyz: torch.Tensor = grad_outputs[0]  # shape == (*, 3)
        i, j, k, idx = ctx.saved_tensors
        return grad_xyz[idx, i], grad_xyz[idx, j], grad_xyz[idx, k], grad_xyz[:, 3], None, None, None


def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    build quaternion from rotation matrix
    """
    # TODO: check!
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Expected `matrix` to have shape (3, 3) or (N, 3, 3), got {}".format(matrix.shape))

    origin_shape = matrix.shape
    matrix: torch.Tensor = matrix.view(-1, 3, 3)

    decision_xyz: torch.Tensor = torch.diagonal(matrix, dim1=1, dim2=2)  # [:, :3]
    decision_w: torch.Tensor = decision_xyz.sum(dim=1, keepdim=True)  # [:, -1]
    decision_matrix: torch.Tensor = torch.cat([decision_xyz, decision_w], dim=-1)
    choices: torch.Tensor = decision_matrix.argmax(dim=1)

    flg: torch.Tensor = torch.as_tensor(choices != 3)  # flg.requires_grad is False
    quat_neq3: Optional[torch.Tensor] = None
    quat_eq3: Optional[torch.Tensor] = None
    ind: torch.Tensor = torch.nonzero(flg).view(-1)
    if len(ind > 0):
        i: torch.Tensor = choices[ind]
        j: torch.Tensor = torch.as_tensor((i + 1) % 3)
        k: torch.Tensor = torch.as_tensor((j + 1) % 3)

        quat_neq3_i: torch.Tensor = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]  # quat[ind, i]
        quat_neq3_j: torch.Tensor = matrix[ind, j, i] + matrix[ind, i, j]  # quat[ind, j]
        quat_neq3_k: torch.Tensor = matrix[ind, k, i] + matrix[ind, i, k]  # quat[ind, k]
        quat_neq3_3: torch.Tensor = matrix[ind, k, j] - matrix[ind, j, k]  # quat[ind, 3]
        quat_neq3 = _QuatCatFunc.apply(quat_neq3_i, quat_neq3_j, quat_neq3_k, quat_neq3_3, i, j, k)

    ind: torch.Tensor = torch.nonzero(~flg)
    if len(ind > 0):
        quat_eq3_0: torch.Tensor = matrix[ind, 2, 1] - matrix[ind, 1, 2]
        quat_eq3_1: torch.Tensor = matrix[ind, 0, 2] - matrix[ind, 2, 0]
        quat_eq3_2: torch.Tensor = matrix[ind, 1, 0] - matrix[ind, 0, 1]
        quat_eq3_3: torch.Tensor = 1 + decision_matrix[ind, -1]
        quat_eq3: torch.Tensor = torch.cat([quat_eq3_0, quat_eq3_1, quat_eq3_2, quat_eq3_3], dim=-1)

    if quat_neq3 is None:
        quat: torch.Tensor = quat_eq3
    elif quat_eq3 is None:
        quat: torch.Tensor = quat_neq3
    else:
        quat: torch.Tensor = CatWithMask.apply(quat_neq3, quat_eq3, flg)

    quat: torch.Tensor = quat_normalize(quat)
    quat: torch.Tensor = quat.view(origin_shape[:-2] + (4,))
    return quat.view(-1) if quat.size == 4 else quat


def vector_to_cross_matrix(x: torch.Tensor) -> torch.Tensor:
    """create cross-product matrix for v

    Args:
        x (torch.Tensor): a vector with shape (..., 3, 1)
    """
    assert x.shape[-2:] == (3, 1)
    x0: torch.Tensor = x[..., 0, :]
    x1: torch.Tensor = x[..., 1, :]
    x2: torch.Tensor = x[..., 2, :]

    zero00: torch.Tensor = torch.zeros_like(x1, dtype=x.dtype, device=x.device)
    zero11: torch.Tensor = torch.zeros_like(x1, dtype=x.dtype, device=x.device)
    zero22: torch.Tensor = torch.zeros_like(x1, dtype=x.dtype, device=x.device)

    mat = torch.stack((
        zero00, -x2, x1,
        x2, zero11, -x0,
        -x1, x0, zero22
    ), dim=-1).view(*x.shape[:-2], 3, 3)

    return mat


def quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Modified from scipy.spatial.transform.Rotation.from_rotvec() method
    Convert rotvec to quaternion

    return: quaternion in torch.Tensor
    """
    if rotvec.ndim not in [1, 2] or rotvec.shape[-1] != 3:
        raise ValueError("Expected `rot_vec` to have shape (3,) or (N, 3), got {}".format(rotvec.shape))

    if rotvec.shape == (3,):
        rotvec = rotvec[None, :]

    norms: torch.Tensor = torch.linalg.norm(rotvec, axis=1)
    small_angle = torch.as_tensor(norms <= 1e-3)
    large_angle = torch.as_tensor(~small_angle)

    # for hack..
    # small_angle = torch.as_tensor(norm > -10000)
    # large_angle = torch.as_tensor(~small_angle)

    scale_small = scale_large = None
    if torch.any(small_angle):
        scale_small = (0.5 - norms ** 2 / 48 + norms ** 4 / 3840)
    if torch.any(large_angle):
        scale_large = (torch.sin(norms / 2) / norms)

    if scale_small is None:
        scale = scale_large
    elif scale_large is None:
        scale = scale_small
    else:
        scale = torch.where(small_angle, scale_small, scale_large)
        # scale = CatWithMask.apply(scale_small, scale_large, small_angle)

    quat_xyz = scale[:, None] * rotvec  # [..., :3]
    quat_w = torch.cos(torch.as_tensor(0.5) * norms)[..., None]  # [..., 3, None]
    quat = torch.cat([quat_xyz, quat_w], dim=-1)

    return quat.view(-1) if rotvec.size == 3 else quat


def quat_from_vec_and_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    build quaternion from axis and angle
    TODO: Test
    """
    assert axis.shape[-1] == 3
    angle = angle.view(axis.shape[:-1] + (1,))
    axis: torch.Tensor = vec_normalize(axis)
    a_div_2: torch.Tensor = 0.5 * angle
    sin_t: torch.Tensor = torch.sin(a_div_2)
    cos_t: torch.Tensor = torch.cos(a_div_2)
    q_xyz: torch.Tensor = sin_t * axis
    quat: torch.Tensor = torch.cat([q_xyz, cos_t], dim=-1)
    return quat


def flip_quat_by_w(q: torch.Tensor) -> torch.Tensor:
    """
    flip quaternion by w
    """
    assert q.shape[-1] == 4
    
    mask = torch.as_tensor(q[..., 3] < 0, dtype=torch.int32)
    mask[mask == 1] = -1
    mask[mask == 0] = 1
    res = q * mask[..., None]
    
    return res


def flip_vec_by_dot(x: torch.Tensor) -> torch.Tensor:
    """
        make sure x[i] * x[i+1] >= 0

        numpy version:
        sign: np.ndarray = np.sum(x[:-1] * x[1:], axis=-1)
        sign[sign < 0] = -1
        sign[sign >= 0] = 1
        sign = np.cumprod(sign, axis=0, )

        x_res = x.copy() if not inplace else x
        x_res[1:][sign < 0] *= -1

        return x_res

        TODO: result with numpy is right..but is gradient right...?
    """
    if x.ndim != 1:
        with torch.no_grad():
            sign: torch.Tensor = torch.sum(x[:-1] * x[1:], dim=-1)
            sign[sign < 0] = -1
            sign[sign >= 0] = 1
            sign = torch.cumprod(sign, dim=0)

            # we can write a mask without gradient...
            mask: torch.Tensor = torch.as_tensor(sign < 0, dtype=torch.long)
            mask[mask == 1] = -1
            mask[mask == 0] = 1
            mask = torch.cat([torch.ones((1,), dtype=torch.long, device=mask.device), mask], dim=0)

        res: torch.Tensor = mask[:, None] * x
        return res

    return x


def flip_quat_by_dot(q: torch.Tensor):
    """
    flip quaternion by dot. TODO: Test
    """
    return flip_vec_by_dot(q)


def quat_to_rotvec(q: torch.Tensor, do_normalize: bool = False):  # Test OK
    """
    Modified from scipy.spatial.transform.Rotation.as_rotvec
    Convert quaternion to rot vec
    return: rotvec
    """
    assert q.shape[-1] == 4 and q.shape[0] > 0
    if do_normalize:
        q = quat_normalize(q)

    quat: torch.Tensor = flip_quat_by_w(q)
    angle: torch.Tensor = torch.as_tensor(2.0) * torch.atan2(torch.linalg.norm(quat[:, :3], dim=1), quat[:, 3])

    eps = 1e-3
    small_angle: torch.Tensor = torch.as_tensor(angle <= eps)
    large_angle: torch.Tensor = torch.as_tensor(~small_angle)

    scale_small = scale_large = None
    if torch.any(small_angle):
        scale_small = (2 + angle ** 2 / 12 + 7 * angle ** 4 / 2880)
    if torch.any(large_angle):
        scale_large = angle / torch.sin(angle / 2)
    if scale_small is None:
        scale = scale_large
    elif scale_large is None:
        scale = scale_small
    else:
        scale = torch.where(small_angle,scale_small, scale_large)
        # scale = torch.where(small_angle, scale_small, scale_large)
        # scale = CatWithMask.apply(scale_small, scale_large, small_angle)

    rotvec: torch.Tensor = scale[:, None] * quat[:, :3]

    return rotvec.view(-1) if q.shape == (4,) else rotvec


def quat_to_vec6d(q: torch.Tensor, do_normalize: bool = False) -> torch.Tensor:
    assert q.shape[-1] == 4
    mat: torch.Tensor = quat_to_matrix(q, do_normalize)
    res: torch.Tensor = mat[..., :2].contiguous()
    return res


def normalize_vec6d(x: torch.Tensor):  # TODO: Test
    assert x.shape[-2:] == (3, 2)
    x: torch.Tensor = x / torch.linalg.norm(x, dim=-2, keepdims=True)

    first_col: torch.Tensor = x[..., 0].contiguous()
    second_col: torch.Tensor = x[..., 1].contiguous()
    last_col: torch.Tensor = torch.cross(first_col, second_col, dim=-1)
    last_col: torch.Tensor = last_col / torch.linalg.norm(last_col, dim=-1, keepdims=True)

    second_col: torch.Tensor = torch.cross(-first_col, last_col, dim=-1)
    second_col: torch.Tensor = second_col / torch.linalg.norm(second_col, dim=-1, keepdims=True)

    return first_col, second_col, last_col


def normalize_vec6d_cat(x: torch.Tensor) -> torch.Tensor:
    first_col, second_col, last_col = normalize_vec6d(x)
    result: torch.Tensor = torch.cat([first_col[..., None], second_col[..., None]], dim=-1)
    return result


def vec6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    convert vector 6d to rotation matrix. Input dim: (*, 3, 2). Output dim: (*, 3, 3)
    """
    first_col, second_col, last_col = normalize_vec6d(x)
    mat: torch.Tensor = torch.cat([first_col[..., None], second_col[..., None], last_col[..., None]], dim=-1)
    return mat


def vec6d_to_quat(x: torch.Tensor) -> torch.Tensor:
    first_col, second_col, last_col = normalize_vec6d(x)
    mat: torch.Tensor = torch.cat([first_col[..., None], second_col[..., None], last_col[..., None]], dim=-1)
    quat: torch.Tensor = quat_from_matrix(mat.view(-1, 3, 3)).view(x.shape[:-2] + (4,))
    return quat

def matrix_to_angle(x: torch.Tensor) -> torch.Tensor:  # check ok with axis angle format
    # acos((tr(R)-1)/2)
    assert x.shape[-2:] == (3, 3)
    diag: torch.Tensor = torch.diagonal(x, dim1=-1, dim2=-2)
    trace: torch.Tensor = torch.sum(diag, dim=-1)
    trace_inside: torch.Tensor = 0.5 * (trace - 1)
    trace_inside: torch.Tensor = torch.clamp(trace_inside, -1.0, 1.0)  # avoid NaN in acos function
    angle: torch.Tensor = torch.acos(trace_inside)
    return angle


def matrix_inv(x: torch.Tensor) -> torch.Tensor:
    # assume input is normalized.
    assert x.shape[-2:] == (3, 3)
    return torch.transpose(x, -1, -2)


def diff_angle_between_vec6d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape[-2:] == (3, 2)
    assert y.shape[-2:] == (3, 2)
    x_mat = torch.cat([node[..., None] for node in normalize_vec6d(x)], dim=-1)
    y_mat = torch.cat([node[..., None] for node in normalize_vec6d(y)], dim=-1)
    return diff_angle_bewteen_matrix(x_mat, y_mat)


def diff_angle_bewteen_matrix(x_mat: torch.Tensor, y_mat: torch.Tensor) -> torch.Tensor:
    x_mat_inv = matrix_inv(x_mat)
    mat_dup = x_mat_inv @ y_mat
    mat_angle = matrix_to_angle(mat_dup)
    return mat_angle


def symmetric_orthogonalization(x: torch.Tensor) -> torch.Tensor:
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.
    x: should have size [batch_size, 9]
    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).

    for rotation matrix R in SO(3), SVD (R) == R.

    get code from `An Analysis of {SVD} for Deep Rotation Estimation, NIPS 2020`
    """
    init_shape = x.shape
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r.view(init_shape)


def normalize_pred_rot(y_pred: torch.Tensor, rotate_type: RotateType) -> torch.Tensor:
    """
    the output of rotation should be normalized.
    """
    if rotate_type == RotateType.Vec6d:
        y_pred = normalize_vec6d_cat(y_pred)
    elif rotate_type == RotateType.SVD9d:
        y_pred = symmetric_orthogonalization(y_pred)
    elif rotate_type == RotateType.AxisAngle:
        y_pred = y_pred
    elif rotate_type == RotateType.Quaternion:
        y_pred = quat_normalize(y_pred)
    else:
        raise NotImplementedError

    return y_pred


def compute_delta_angle(y_pred: torch.Tensor, y_gt: torch.Tensor, rotate_type: RotateType):
    if rotate_type == RotateType.Vec6d:
        delta_angle = torch.abs(diff_angle_between_vec6d(y_pred, y_gt))
    elif rotate_type == RotateType.SVD9d or rotate_type == RotateType.Matrix:
        delta_angle = torch.abs(diff_angle_bewteen_matrix(y_pred, y_gt))
    elif rotate_type == RotateType.AxisAngle:
        delta_angle = torch.linalg.norm(y_pred - y_gt, dim=-1)
    elif rotate_type == RotateType.Quaternion:
        raise NotImplementedError
    else:
        raise NotImplementedError

    return delta_angle


def convert_to_quat(x: torch.Tensor, rotate_type: RotateType) -> torch.Tensor:
    if rotate_type == RotateType.SVD9d:
        quat = quat_from_matrix(x.view(-1, 3, 3)).view(x.shape[:-2] + (4,))
    elif rotate_type == RotateType.Matrix:
        quat = quat_from_matrix(x.view(-1, 3, 3)).view(x.shape[:-2] + (4,))
    elif rotate_type == RotateType.Vec6d:
        quat = vec6d_to_quat(x)
    elif rotate_type == RotateType.Quaternion:
        quat = x
    else:
        raise NotImplementedError
    return quat


quat_from_other_rotate = convert_to_quat


def quat_to_other_rotate(q: torch.Tensor, rotate_type: RotateType) -> torch.Tensor:
    origin_shape = q.shape
    q: torch.Tensor = q.view(-1, 4)
    if rotate_type == RotateType.SVD9d or rotate_type == RotateType.Matrix:
        ret: torch.Tensor = quat_to_matrix(q)
    elif rotate_type == RotateType.AxisAngle:
        ret: torch.Tensor = quat_to_rotvec(q)
    elif rotate_type == RotateType.Vec6d:
        ret: torch.Tensor = quat_to_vec6d(q)
    elif rotate_type == RotateType.Quaternion:
        ret: torch.Tensor = q
    else:
        raise NotImplementedError
    ret: torch.Tensor = ret.view(*origin_shape[:-1], *ret.shape[1:])
    return ret

