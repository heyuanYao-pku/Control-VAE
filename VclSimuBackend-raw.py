# cython:language_level=3
from argparse import ArgumentParser, Namespace
import copy
import datetime
from enum import IntEnum
import json
import logging
import numpy as np
import os
import pickle
import platform
import random
from scipy.ndimage import filters
from scipy import signal
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import time
import tkinter as tk
from tkinter import filedialog
from typing import Dict, Any, List, Tuple, Union, Iterable, Optional, IO, Set

import ModifyODE as ode
from MotionUtils import (
    quat_to_rotvec_fast, quat_to_matrix_fast, quat_to_vec6d_fast,
    quat_from_rotvec_fast,
    six_dim_mat_to_quat_fast,
    quat_inv_single_fast,
    quat_apply_forward_one2many_fast,
    quat_apply_single_fast,
    decompose_rotation_single_fast,
    quat_apply_forward_fast,
    quat_multiply_forward_fast
)


class Common:
    class GetFileNameByUI:
        @staticmethod
        def get_file_name_by_UI(initialdir="./", filetypes=[("all_file_types", "*.*")]):
            root = tk.Tk()
            root.withdraw()
            return filedialog.askopenfilename(initialdir=initialdir, filetypes = filetypes)

        @staticmethod
        def get_multi_file_name_by_UI():
            root = tk.Tk()
            root.withdraw()
            return filedialog.askopenfilenames()

    class Helper:
        _empty_str_list = ["", "null", "none", "nullptr", "no", "not", "false", "abort"]
        _true_str_list = ["yes", "true", "confirm", "ok", "sure", "ready"]

        def __init__(self):
            pass

        @staticmethod
        def get_curr_time() -> str:
            return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        @staticmethod
        def print_total_time(starttime):
            endtime = datetime.datetime.now()
            delta_time = endtime - starttime
            # logging.info(f"start time: {starttime.strftime('%Y-%m-%d %H:%M:%S')}")
            # logging.info(f"end time: {endtime.strftime('%Y-%m-%d %H:%M:%S')}")
            # logging.info(f"delta time: {delta_time}")
            # logging.info(f"seconds = {delta_time.seconds}")

            print(f"\n\nstart time: {starttime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"end time: {endtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"delta time: {delta_time}")
            print(f"seconds = {delta_time.seconds}", flush=True)

        @staticmethod
        def is_str_empty(s: str) -> bool:
            return s.lower() in Common.Helper._empty_str_list

        @staticmethod
        def str_is_true(s: str) -> bool:
            return s.lower() in Common.Helper._true_str_list

        @staticmethod
        def conf_loader(fname: str) -> Dict[str, Any]:
            with open(fname, "r") as f:
                conf: Dict[str, Any] = json.load(f)
            filename_conf: Dict[str, str] = conf["filename"]
            for k, v in filename_conf.items():
                if k.startswith("__"):
                    continue
                filename_conf[k] = os.path.join(os.path.dirname(fname), v)

            return conf

        @staticmethod
        def set_torch_seed(random_seed: int):
            import torch
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            random.seed(random.seed)
            os.environ['PYTHONHASHSEED'] = str(random_seed)

        @staticmethod
        def load_numpy_random_state(result: Dict[str, Any]) -> None:
            random_state = result.get("random_state")
            if random_state is not None:
                random.setstate(random_state)

            np_rand_state = result.get("np_rand_state")
            if np_rand_state is not None:
                np.random.set_state(np_rand_state)

        @staticmethod
        def save_numpy_random_state() -> Dict[str, Any]:
            result = {
                "random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
            }
            return result

        @staticmethod
        def mirror_name_list(name_list: List[str]):
            indices = [i for i in range(len(name_list))]
            def index(name):
                try:
                    return name_list.index(name)
                except ValueError:
                    return -1

            for i, n in enumerate(name_list):
                # rule 1: left->right
                idx = -1
                if n.find('left') == 0:
                    idx = index('right' + n[4:])
                elif n.find('Left') == 0:
                    idx = index('Right' + n[4:])
                elif n.find('LEFT') == 0:
                    idx = index('RIGHT' + n[4:])
                elif n.find('right') == 0:
                    idx = index('left' + n[5:])
                elif n.find('Right') == 0:
                    idx = index('Left' + n[5:])
                elif n.find('RIGHT') == 0:
                    idx = index('LEFT' + n[5:])
                elif n.find('L') == 0:
                    idx = index('R' + n[1:])
                elif n.find('l') == 0:
                    idx = index('r' + n[1:])
                elif n.find('R') == 0:
                    idx = index('L' + n[1:])
                elif n.find('r') == 0:
                    idx = index('l' + n[1:])

                indices[i] = idx if idx >= 0 else i

            return indices


    class RotateType(IntEnum):
        Matrix = 1
        AxisAngle = 2
        Vec6d = 3
        SVD9d = 4
        Quaternion = 5


    class MathHelper:

        unit_vec6d = np.array([
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0
        ], dtype=np.float64)

        @staticmethod
        def count_ones(x: int) -> int:
            count: int = 0
            while x > 0:
                if x & 1 == 1:
                    count += 1
                x >>= 1
            return count

        @staticmethod
        def array_distance(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
            """
            input: m x 2, n x 2
            output: m x n
            """
            m, k = arr1.shape
            n, k = arr2.shape
            arr1_power: np.ndarray = arr1 ** 2
            arr1_power_sum: np.ndarray = np.sum(arr1_power, axis=1)
            arr1_power_sum: np.ndarray = np.tile(arr1_power_sum, (n, 1))
            arr1_power_sum: np.ndarray = arr1_power_sum.T
            arr2_power: np.ndarray = arr2 ** 2
            arr2_power_sum: np.ndarray = np.sum(arr2_power, axis=-1)
            arr2_power_sum: np.ndarray = np.tile(arr2_power_sum, (m, 1))
            dis: np.ndarray = arr1_power_sum + arr2_power_sum - (2 * np.dot(arr1, arr2.T))
            dis: np.ndarray = np.sqrt(dis)

            def test_func():
                test_dis = np.zeros((m, n))
                for i in range(m):
                    for j in range(n):
                        test_dis[i, j] = np.linalg.norm(arr1[i] - arr2[j])
                print(np.max(test_dis - dis))

            # test_func()
            return dis

        @staticmethod
        def quat_from_other_rotate(x: np.ndarray, rotate_type) -> np.ndarray:
            if rotate_type == Common.RotateType.Matrix or rotate_type == Common.RotateType.SVD9d:
                return Common.MathHelper.matrix_to_quat(x)
            elif rotate_type == Common.RotateType.Quaternion:
                return x
            elif rotate_type == Common.RotateType.AxisAngle:
                return Common.MathHelper.quat_from_axis_angle(x)
            elif rotate_type == Common.RotateType.Vec6d:
                return Common.MathHelper.vec6d_to_quat(x)
            else:
                raise NotImplementedError

        @staticmethod
        def quat_to_other_rotate(quat: np.ndarray, rotate_type):
            if rotate_type == Common.RotateType.SVD9d or rotate_type == Common.RotateType.Matrix:
                return Common.MathHelper.quat_to_matrix(quat)
            elif rotate_type == Common.RotateType.Vec6d:
                return Common.MathHelper.quat_to_vec6d(quat)
            elif rotate_type == Common.RotateType.Quaternion:
                return quat
            elif rotate_type == Common.RotateType.AxisAngle:
                return Common.MathHelper.quat_to_axis_angle(quat)
            else:
                raise NotImplementedError

        @staticmethod
        def get_rotation_dim(rotate_type):
            if rotate_type == Common.RotateType.Vec6d:
                return 6
            elif rotate_type == Common.RotateType.AxisAngle:
                return 3
            elif rotate_type == Common.RotateType.SVD9d:
                return 9
            elif rotate_type == Common.RotateType.Matrix:
                return 9
            elif rotate_type == Common.RotateType.Quaternion:
                return 4
            else:
                raise NotImplementedError

        @staticmethod
        def get_rotation_last_shape(rotate_type) -> Tuple:
            if rotate_type == Common.RotateType.Vec6d:
                last_shape = (3, 2)
            elif rotate_type == Common.RotateType.AxisAngle:
                last_shape = (3,)
            elif rotate_type == Common.RotateType.SVD9d:
                last_shape = (3, 3)
            elif rotate_type == Common.RotateType.Matrix:
                last_shape = (3, 3)
            elif rotate_type == Common.RotateType.Quaternion:
                last_shape = (4,)
            else:
                raise NotImplementedError

            return last_shape

        @staticmethod
        def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            """
            multiply 2 quaternions. p.shape == q.shape
            """
            assert 2 == len(p.shape) and 4 == p.shape[-1]
            assert 2 == len(q.shape) and 4 == q.shape[-1]

            w: np.ndarray = p[:, 3:4] * q[:, 3:4] - np.sum(p[:, :3] * q[:, :3], axis=1, keepdims=True)
            xyz: np.ndarray = p[:, None, 3] * q[:, :3] + q[:, None, 3] * p[:, :3] + np.cross(p[:, :3], q[:, :3], axis=1)

            return np.concatenate([xyz, w], axis=-1)

        @staticmethod
        def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float):
            """
            update quaternion, q_{t+1} = normalize(q_{t} + 0.5 * w * q_{t})
            """
            init_q_shape = q.shape
            if q.shape[-1] == 1 and omega.shape[-1] == 1:
                q = q.reshape(q.shape[:-1])
                omega = omega.reshape(omega.shape[:-1])
            assert 4 == q.shape[-1] and 3 == omega.shape[-1]

            omega = np.concatenate([omega, np.zeros(omega.shape[:-1] + (1,))], axis=-1)

            delta_q = 0.5 * dt * Common.MathHelper.quat_multiply(omega, q)
            result = q + delta_q
            result /= np.linalg.norm(result, axis=-1, keepdims=True)
            return result.reshape(init_q_shape)

        @staticmethod
        def quat_to_matrix(q: np.ndarray) -> np.ndarray:
            assert q.shape[-1] == 4
            return Rotation(q.reshape(-1, 4), copy=False, normalize=False).as_matrix().reshape(q.shape[:-1] + (3, 3))

        @staticmethod
        def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
            assert mat.shape[-2:] == (3, 3)
            return Rotation.from_matrix(mat.reshape((-1, 3, 3))).as_quat().reshape(mat.shape[:-2] + (4,))

        @staticmethod
        def quat_to_vec6d(q: np.ndarray) -> np.ndarray:
            """
            input quaternion in shape (..., 4)
            return: in shape (..., 3, 2)
            """
            assert q.shape[-1] == 4
            mat: np.ndarray = Rotation(q.reshape(-1, 4)).as_matrix()  # shape == (N, 3, 3)
            mat: np.ndarray = mat.reshape(q.shape[:-1] + (3, 3))
            return mat[..., :2]

        @staticmethod
        def vec6d_to_quat(x: np.ndarray, normalize: bool = True) -> np.ndarray:
            """
            input 6d vector in shape (..., 3, 2)
            return in shape (..., 4)
            """
            assert x.shape[-2:] == (3, 2)
            if normalize:
                x = x / np.linalg.norm(x, axis=-2, keepdims=True)

            last_col: np.ndarray = np.cross(x[..., 0], x[..., 1], axis=-1)
            last_col = last_col / np.linalg.norm(last_col, axis=-1, keepdims=True)

            mat = np.concatenate([x, last_col[..., None]], axis=-1)
            quat: np.ndarray = Rotation.from_matrix(mat.reshape((-1, 3, 3))).as_quat().reshape(x.shape[:-2] + (4,))
            return quat

        @staticmethod
        def normalize_angle(a: np.ndarray) -> np.ndarray:
            """
            Covert angles to [-pi, pi)
            """
            res: np.ndarray = a.copy()
            res[res >= np.pi] -= 2 * np.pi
            res[res < np.pi] += 2 * np.pi
            return res

        @staticmethod
        def normalize_vec(a: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(a, axis=-1, keepdims=True)
            norm[norm == 0] = 1
            return a / norm

        @staticmethod
        def up_vector() -> np.ndarray:
            """
            return (0, 1, 0)
            """
            return np.array([0.0, 1.0, 0.0])

        @staticmethod
        def ego_forward_vector() -> np.ndarray:
            """
            return (0, 0, 1)
            """
            return np.array([0.0, 0.0, 1.0])

        @staticmethod
        def unit_vector(axis: int) -> np.ndarray:  # shape == (3,)
            res = np.zeros(3)
            res[axis] = 1
            return res

        @staticmethod
        def unit_quat_scipy() -> np.ndarray:  # shape == (4,)
            return Common.MathHelper.unit_quat()

        @staticmethod
        def unit_quat_scipy_list() -> List[float]:
            return [0.0, 0.0, 0.0, 1.0]

        @staticmethod
        def quat_from_scipy_to_ode(q: np.ndarray) -> np.ndarray:
            return Common.MathHelper.xyzw_to_wxyz(q)

        @staticmethod
        def quat_from_ode_to_scipy(q: np.ndarray) -> np.ndarray:
            return Common.MathHelper.wxyz_to_xyzw(q)

        @staticmethod
        def quat_from_ode_to_unity(q: np.ndarray) -> np.ndarray:
            return Common.MathHelper.wxyz_to_xyzw(q)

        @staticmethod
        def unit_quat_ode() -> np.ndarray:
            return np.array(Common.MathHelper.unit_quat_ode_list())

        @staticmethod
        def unit_quat_ode_list() -> List[float]:
            return [1.0, 0.0, 0.0, 0.0]

        @staticmethod
        def unit_quat_unity() -> np.ndarray:
            return np.asarray(Common.MathHelper.unit_quat_unity_list())

        @staticmethod
        def unit_quat_unity_list() -> List[float]:
            return [0.0, 0.0, 0.0, 1.0]

        @staticmethod
        def unit_quat() -> np.ndarray:
            return np.array([0.0, 0.0, 0.0, 1.0])

        @staticmethod
        def unit_quat_arr(shape: Union[int, Iterable, Tuple[int]]) -> np.ndarray:
            if type(shape) == int:
                shape = (shape, 4)

            res = np.zeros(shape, dtype=np.float64)
            res[..., -1] = 1
            return res.reshape(shape)

        @staticmethod
        def ode_quat_to_rot_mat(q: np.ndarray) -> np.ndarray:
            return Rotation(Common.MathHelper.quat_from_ode_to_scipy(q)).as_matrix()

        @staticmethod
        def rot_mat_to_ode_quat(mat: np.ndarray) -> np.ndarray:
            return Common.MathHelper.quat_from_scipy_to_ode(Rotation.from_matrix(mat).as_quat())

        @staticmethod
        def vec_diff(v_in: np.ndarray, forward: bool, fps: float):
            v = np.empty_like(v_in)
            frag = v[:-1] if forward else v[1:]
            frag[:] = np.diff(v_in, axis=0) * fps
            v[-1 if forward else 0] = v[-2 if forward else 1]
            return v

        @staticmethod
        def vec_axis_to_zero(v: np.ndarray, axis: Union[int, List[int], np.ndarray]) -> np.ndarray:
            res: np.ndarray = v.copy()
            res[..., axis] = 0
            return res

        @staticmethod
        def xz_vector_to_xyz(xz: np.ndarray) -> np.ndarray:
            xyz = np.zeros((xz.shape[:-1] + (3,)))
            xyz[..., [0, 2]] = xz
            return xyz

        @staticmethod
        def flip_quat_by_w(q: np.ndarray) -> np.ndarray:
            res = q.copy()
            idx: np.ndarray = res[..., -1] < 0
            res[idx, :] = -res[idx, :]
            return res

        @staticmethod
        def flip_quat_arr_by_w(*args):
            return [Common.MathHelper.flip_quat_by_w(i) for i in args]

        @staticmethod
        def flip_vector_by_dot(x: np.ndarray, inplace: bool = False) -> np.ndarray:
            """
            make sure x[i] * x[i+1] >= 0
            """
            if x.ndim == 1:
                return x

            sign: np.ndarray = np.sum(x[:-1] * x[1:], axis=-1)
            sign[sign < 0] = -1
            sign[sign >= 0] = 1
            sign = np.cumprod(sign, axis=0, )

            x_res = x.copy() if not inplace else x
            x_res[1:][sign < 0] *= -1

            return x_res

        @staticmethod
        def flip_vec3_by_dot(x: np.ndarray, inplace: bool = False) -> np.ndarray:
            assert x.shape[-1] == 3
            return Common.MathHelper.flip_vector_by_dot(x, inplace)

        @staticmethod
        def flip_quat_by_dot(q: np.ndarray, inplace: bool = False) -> np.ndarray:
            if q.shape[-1] != 4:
                raise ValueError

            return Common.MathHelper.flip_vector_by_dot(q, inplace)

        @staticmethod
        def flip_quat_arr_by_dot(*args) -> List[np.ndarray]:
            return [Common.MathHelper.flip_quat_by_dot(i) for i in args]

        @staticmethod
        def flip_quat_pair_by_dot(q0s: np.ndarray, q1s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            q0 will not be changed.
            q1 will be flipped to the same semi sphere as q0
            """
            assert q0s.shape[-1] == 4
            assert q1s.shape == q0s.shape

            dot_value = np.sum(q0s * q1s, axis=-1, keepdims=True) < 0
            dot_res = np.concatenate([dot_value] * 4, axis=-1)
            q1: np.ndarray = q1s.copy()
            q1[dot_res] = -q1[dot_res]
            return q0s, q1

        @staticmethod
        def quat_equal(q1: np.ndarray, q2: np.ndarray) -> bool:
            return np.all(np.abs(Common.MathHelper.flip_quat_by_w(q1) - Common.MathHelper.flip_quat_by_w(q2)) < 1e-5)

        @staticmethod
        def proj_vec_to_plane(a: np.ndarray, v: np.ndarray):
            """
            Project Vector to Plane
            :param a: original vector
            :param v: Normal vector of Plane
            :return: a_new(result of projection)
            """
            # k: coef.
            # a_new = a - k * v
            # a_new * v = 0
            # Solution: k = (a * v) / (v * v)
            k: np.ndarray = np.sum(a * v, axis=-1) / np.sum(v * v, axis=-1)  # (N, )
            return a - np.repeat(k, 3).reshape(v.shape) * v

        @staticmethod
        def proj_multi_vec_to_a_plane(a_arr: np.ndarray, v: np.ndarray):
            v_arr = np.zeros_like(a_arr)
            v_arr[:, :] = v
            return Common.MathHelper.proj_vec_to_plane(a_arr, v_arr)

        @staticmethod
        def quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """
            Rotation from vector a to vector b
            :param a: (n, 3) vector
            :param b: (n, 3) vector
            :return: (n, 4) quaternion
            """
            cross_res = np.cross(a, b)
            w_ = np.sqrt((a ** 2).sum(axis=-1) * (b ** 2).sum(axis=-1)) + (a * b).sum(axis=-1)
            res_ = np.concatenate([cross_res, w_[..., np.newaxis]], axis=-1)
            return res_ / np.linalg.norm(res_, axis=-1, keepdims=True)

        @staticmethod
        def quat_to_axis_angle(q: np.ndarray, normalize=True, copy=True):
            assert q.shape[-1] == 4
            return Rotation(q.reshape((-1, 4)), normalize=normalize, copy=copy).as_rotvec().reshape(q.shape[:-1] + (3, ))

        @staticmethod
        def quat_from_axis_angle(axis: np.ndarray, angle: Optional[np.ndarray] = None, normalize: bool = False) -> np.ndarray:
            if angle is not None:
                assert axis.shape == angle.shape + (3,)
                if normalize:
                    axis: np.ndarray = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
                angle: np.ndarray = (0.5 * angle)[..., None]
                sin_res: np.ndarray = np.sin(angle)
                cos_res: np.ndarray = np.cos(angle)
                result = np.concatenate([axis * sin_res, cos_res], axis=-1)
                return result
            else:
                return Rotation.from_rotvec(axis.reshape((-1, 3))).as_quat().reshape(axis.shape[:-1] + (4,))

        @staticmethod
        def log_quat(q: np.ndarray) -> np.ndarray:
            """
            log quaternion。
            :param q: (n, 4) quaternion
            :return:
            """
            if q.shape[-1] != 4:
                raise ArithmeticError
            if q.ndim > 1:
                return 0.5 * Rotation(q.reshape(-1, 4), copy=False).as_rotvec().reshape(q.shape[:-1] + (3,))
            else:
                return 0.5 * Rotation(q, copy=False).as_rotvec()

        @staticmethod
        def exp_to_quat(v: np.ndarray) -> np.ndarray:
            """

            :param v:
            :return:
            """
            # Note that q and -q is the same rotation. so result is not unique
            if v.shape[-1] != 3:
                raise ArithmeticError
            if v.ndim > 1:
                return Rotation.from_rotvec(2 * v.reshape(-1, 3)).as_quat().reshape(v.shape[:-1] + (4,))
            else:
                return Rotation.from_rotvec(2 * v).as_quat()

        @staticmethod
        def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
            return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)

        @staticmethod
        def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
            return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)

        @staticmethod
        def facing_decompose(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            return: ry, facing. ry only has y component, and facing only has (x, z) component.
            """
            return Common.MathHelper.y_decompose(q)

        @staticmethod
        def extract_heading_Y_up(q: np.ndarray):
            """ extract the rotation around Y axis from given quaternions

                note the quaterions should be {(x,y,z,w)}
            """
            q = np.asarray(q)
            shape = q.shape
            q = q.reshape(-1, 4)

            v = Rotation(q, normalize=False, copy=False).as_matrix()[:, :, 1]

            # axis=np.cross(v,(0,1,0))
            axis = v[:, (2, 1, 0)]
            axis *= [-1, 0, 1]

            norms = np.linalg.norm(axis, axis=-1)
            scales = np.empty_like(norms)
            small_angle = (norms <= 1e-3)
            large_angle = ~small_angle

            scales[small_angle] = norms[small_angle] + norms[small_angle] ** 3 / 6
            scales[large_angle] = np.arccos(v[large_angle, 1]) / norms[large_angle]

            correct = Rotation.from_rotvec(axis * scales[:, None])

            heading = (correct * Rotation(q, normalize=False, copy=False)).as_quat()
            heading[heading[:, -1] < 0] *= -1

            return heading.reshape(shape)

        @staticmethod
        def decompose_rotation(q: np.ndarray, vb: np.ndarray):
            rot_q = Rotation(q, copy=False)
            va = rot_q.apply(vb)
            va /= np.linalg.norm(va, axis=-1, keepdims=True)

            rot_axis = np.cross(va, vb)
            rot_axis_norm = np.linalg.norm(rot_axis, axis=-1, keepdims=True)
            rot_axis_norm[rot_axis_norm < 1e-14] = 1e-14
            rot_axis /= rot_axis_norm

            rot_angle = np.asarray(-np.arccos(np.clip(va.dot(vb), -1, 1))).reshape(-1)
            # TODO: minus or plus..?

            if rot_axis.ndim > 1:
                rot_angle = rot_angle.reshape(-1, 1)

            ret_result: np.ndarray = (Rotation.from_rotvec(rot_angle * (-rot_axis)) * rot_q).as_quat()
            ret_result: np.ndarray = Common.MathHelper.flip_quat_by_dot(ret_result)
            return ret_result

        @staticmethod
        def axis_decompose(q: np.ndarray, axis: np.ndarray):
            """
            return:
            res: rotation along axis
            r_other:
            """
            assert axis.ndim == 1 and axis.shape[0] == 3
            res = Common.MathHelper.decompose_rotation(q, np.asarray(axis))
            r_other = (Rotation(res, copy=False, normalize=False).inv() * Rotation(q, copy=False, normalize=False)).as_quat()
            r_other = Common.MathHelper.flip_quat_by_dot(r_other)
            res[np.abs(res) < 1e-14] = 0
            r_other[np.abs(r_other) < 1e-14] = 0
            res /= np.linalg.norm(res, axis=-1, keepdims=True)
            r_other /= np.linalg.norm(r_other, axis=-1, keepdims=True)
            return res, r_other

        @staticmethod
        def x_decompose(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return Common.MathHelper.axis_decompose(q, np.array([1.0, 0.0, 0.0]))

        @staticmethod
        def y_decompose(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return Common.MathHelper.axis_decompose(q, np.array([0.0, 1.0, 0.0]))

        @staticmethod
        def z_decompose(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return Common.MathHelper.axis_decompose(q, np.array([0.0, 0.0, 1.0]))

        @staticmethod
        def resample_joint_linear(x: np.ndarray, ratio: int, old_fps: int):
            old_frame: int = x.shape[0]
            new_frame: int = old_frame * ratio
            new_fps: int = old_fps * ratio
            result: np.ndarray = np.zeros((new_frame,) + x.shape[1:])
            nj: int = x.shape[1]
            ticks = np.arange(0, old_frame, dtype=np.float64) / old_fps
            new_ticks = np.arange(0, new_frame, dtype=np.float64) / new_fps
            for index in range(nj):
                j_interp = interp1d(ticks, x[:, index], kind='linear', axis=0, copy=True, bounds_error=False, assume_sorted=True)
                result[:, index] = j_interp(new_ticks)
            return result

        @staticmethod
        def slerp(q0s: np.ndarray, q1s: np.ndarray, t: Union[float, np.ndarray], eps: float = 1e-7):
            q0s = q0s.reshape((1, 4)) if q0s.shape == (4,) else q0s
            q1s = q1s.reshape((1, 4)) if q1s.shape == (4,) else q1s
            assert q0s.shape[-1] == 4 and q0s.ndim == 2 and q0s.shape == q1s.shape
            is_ndarray = isinstance(t, np.ndarray)
            assert not is_ndarray or t.size == q0s.shape[0]
            t = t.reshape((-1, 1)) if is_ndarray else t

            # filp by dot
            q0, q1 = Common.MathHelper.flip_quat_pair_by_dot(q0s, q1s)  # (n, 4), (n, 4)
            if np.allclose(q0, q1):
                return q0

            theta = np.arccos(np.sum(q0 * q1, axis=-1))  # (n,)
            res = Common.MathHelper.unit_quat_arr(q0.shape)  # (n, 4)

            small_flag: np.ndarray = np.abs(theta) < eps  # (small,)
            small_idx = np.argwhere(small_flag).flatten()  # (small,)
            t_small = t[small_idx] if is_ndarray else t  # (small, 1) or float
            res[small_idx] = (1.0 - t_small) * q0[small_idx] + t_small * q1[small_idx]  # (small, 4)
            res[small_idx] /= np.linalg.norm(res[small_idx], axis=-1, keepdims=True)  # (small, 4)

            plain_idx: np.ndarray = np.argwhere(~small_flag).flatten()  # (plain,)
            theta_plain = theta[plain_idx, None]  # (plain, 1)
            inv_sin_theta = 1.0 / np.sin(theta_plain)  # (plain, 1)
            t_plain = t[plain_idx] if is_ndarray else t  # (plain, 1) or float
            res[plain_idx] = (np.sin((1.0 - t_plain) * theta_plain) * inv_sin_theta) * q0[plain_idx] + \
                            (np.sin(t_plain * theta_plain) * inv_sin_theta) * q1[plain_idx]  # (plain, 4)

            res = np.ascontiguousarray(res / np.linalg.norm(res, axis=-1, keepdims=True))
            return res

        @staticmethod
        def average_quat_by_slerp(qs: List[np.ndarray]) -> np.ndarray:
            result: np.ndarray = qs[0].copy()
            len_qs: int = len(qs)
            for i in range(1, len_qs):
                ratio: float = float(i) / (i + 1)
                result: np.ndarray = Common.MathHelper.slerp(result, qs[i], ratio)
            return result

        @staticmethod
        def torch_skew(v):
            '''
            :param v : torch.Tensor [3,1] or [1,3]
            this function will return the skew matrix (cross product matrix) of a vector
            be sure that it has ONLY 3 element
            it can be autograd
            '''
            import torch
            skv = torch.diag(torch.flatten(v)).roll(1, 1).roll(-1, 0)
            return skv - skv.transpose(0, 1)

        @staticmethod
        def cross_mat(v):
            """create cross-product matrix for v

            Args:
                v (torch.Tensor): a vector with shape (..., 3, 1)
            """
            import torch
            mat = torch.stack((
                torch.zeros_like(v[..., 0, :]), -v[..., 2, :], v[..., 1, :],
                v[..., 2, :], torch.zeros_like(v[..., 1, :]), -v[..., 0, :],
                -v[..., 1, :], v[..., 0, :], torch.zeros_like(v[..., 2, :])
            ), dim=-1).view(*v.shape[:-2], 3, 3)

            return mat

        @staticmethod
        def np_skew(v: np.ndarray):
            return np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]
                            ], dtype=np.float64)


    class RotateConvertFast:
        @staticmethod
        def quat_single_to_other_rotate(x: np.ndarray, rotate_type) -> np.ndarray:
            pass

        @staticmethod
        def quat_single_from_other_rotate(x: np.ndarray, rotate_type) -> np.ndarray:
            pass

        @staticmethod
        def quat_to_other_rotate(x: np.ndarray, rotate_type) -> np.ndarray:
            x: np.ndarray = np.ascontiguousarray(x, dtype=np.float64)
            if rotate_type == Common.RotateType.Matrix or rotate_type == Common.RotateType.SVD9d:
                return quat_to_matrix_fast(x)
            elif rotate_type == Common.RotateType.Quaternion:
                return x
            elif rotate_type == Common.RotateType.AxisAngle:
                return quat_to_rotvec_fast(x)[1]
            elif rotate_type == Common.RotateType.Vec6d:
                return quat_to_vec6d_fast(x)
            else:
                raise NotImplementedError

        @staticmethod
        def quat_from_other_rotate(x: np.ndarray, rotate_type) -> np.ndarray:
            q: np.ndarray = np.ascontiguousarray(x, dtype=np.float64)
            if rotate_type == Common.RotateType.Matrix or rotate_type == Common.RotateType.SVD9d:
                raise NotImplementedError
            elif rotate_type == Common.RotateType.Quaternion:
                return q
            elif rotate_type == Common.RotateType.AxisAngle:
                return quat_from_rotvec_fast(q)
            elif rotate_type == Common.RotateType.Vec6d:
                return six_dim_mat_to_quat_fast(q)
            else:
                raise NotImplementedError

    class SmoothOperator:
        class SmoothMode(IntEnum):
            NO = 0  # not using smooth
            GAUSSIAN = 1  # use gaussian smooth
            BUTTER_WORTH = 2  # use butter worth smooth

        class GaussianBase:
            __slots__ = ("width",)

            def __init__(self, width: Optional[int]):
                self.width: Optional[int] = width

        class FilterInfoBase:

            __slots__ = ("order", "wn")

            def __init__(self, order: int, cut_off_freq: float, sample_freq: int):
                self.order = order
                self.wn = self.calc_freq(cut_off_freq, sample_freq)

            @classmethod
            def build_from_dict(cls, info: Optional[Dict[str, Any]], sample_freq: int):
                return cls(info["order"], info["cut_off_freq"], sample_freq) if info is not None else None

            @staticmethod
            def calc_freq(cut_off_freq: float, sample_freq: float) -> float:
                return cut_off_freq / (sample_freq / 2)

        class ButterWorthBase(FilterInfoBase):
            __slots__ = ("order", "wn")

            def __init__(self, order: int, cut_off_freq: float, sample_freq: int):
                super().__init__(order, cut_off_freq, sample_freq)

        @staticmethod
        def smooth_operator(x: np.ndarray, smooth_type) -> np.ndarray:
            """
            The first dimension of x is time
            """
            # print(f"call smoother operator, smooth_type == {type(smooth_type)}")
            if smooth_type is None:
                result = x
            elif isinstance(smooth_type, Common.SmoothOperator.GaussianBase):
                if smooth_type.width is not None:
                    result = filters.gaussian_filter1d(x, smooth_type.width, axis=0, mode='nearest')
                else:
                    result = x
            elif isinstance(smooth_type, Common.SmoothOperator.ButterWorthBase):
                b, a = signal.butter(smooth_type.order, smooth_type.wn)
                result = signal.filtfilt(b, a, x, axis=0)
            else:
                raise NotImplementedError("Only support GaussianBase and ButterWorthBase.")

            return result


class pymotionlib:
    class Utils:
        @staticmethod
        def quat_product(p: np.ndarray, q: np.ndarray, inv_p: bool = False, inv_q: bool = False):
            if p.shape[-1] != 4 or q.shape[-1] != 4:
                raise ValueError('operands should be quaternions')

            if len(p.shape) != len(q.shape):
                if len(p.shape) == 1:
                    p.reshape([1] * (len(q.shape) - 1) + [4])
                elif len(q.shape) == 1:
                    q.reshape([1] * (len(p.shape) - 1) + [4])
                else:
                    raise ValueError('mismatching dimensions')

            is_flat = len(p.shape) == 1
            if is_flat:
                p = p.reshape(1, 4)
                q = q.reshape(1, 4)

            product = np.empty([max(p.shape[i], q.shape[i]) for i in range(len(p.shape) - 1)] + [4],
                            dtype=np.result_type(p.dtype, q.dtype))

            pw = p[..., 3] if not inv_p else -p[..., 3]
            qw = q[..., 3] if not inv_q else -q[..., 3]

            product[..., 3] = pw * qw - np.sum(p[..., :3] * q[..., :3], axis=-1)
            product[..., :3] = (pw[..., None] * q[..., :3] + qw[..., None] * p[..., :3] +
                                np.cross(p[..., :3], q[..., :3]))

            if is_flat:
                product = product.reshape(4)

            return product

        @staticmethod
        def flip_vector(vt: np.ndarray, normal: np.ndarray, inplace: bool):
            vt = np.asarray(vt).reshape(-1, 3)
            normal = np.asarray(normal).reshape(-1, 3)
            if inplace:
                vt -= (2 * np.sum(vt * normal, axis=-1, keepdims=True)) * normal
                return vt
            else:
                return vt - (2 * np.sum(vt * normal, axis=-1, keepdims=True)) * normal

        @staticmethod
        def flip_quaternion(qt: np.ndarray, normal: np.ndarray, inplace: bool):
            qt = np.asarray(qt).reshape(-1, 4)
            normal = np.asarray(normal).reshape(-1, 3)

            if not inplace:
                qt = qt.copy()
            pymotionlib.Utils.flip_vector(qt[:, :3], normal, True)
            qt[:, -1] = -qt[:, -1]
            return qt

        @staticmethod
        def align_angles(a: np.ndarray, degrees: bool, inplace: bool):
            ''' make the angles in the array continuous

                we assume the first dim of a is the time
            '''
            step = 360. if degrees else np.pi * 2

            a = np.asarray(a)
            diff = np.diff(a, axis=0)
            num_steps = np.round(diff / step)
            num_steps = np.cumsum(num_steps, axis=0)
            if not inplace:
                a = a.copy()
            a[1:] -= num_steps * step

            return a

        @staticmethod
        def align_quaternion(qt: np.ndarray, inplace: bool):
            ''' make q_n and q_n+1 in the same semisphere

                the first axis of qt should be the time
            '''
            qt = np.asarray(qt)
            if qt.shape[-1] != 4:
                raise ValueError('qt has to be an array of quaterions')

            if not inplace:
                qt = qt.copy()

            if qt.size == 4:  # do nothing since there is only one quation
                return qt

            sign = np.sum(qt[:-1] * qt[1:], axis=-1)
            sign[sign < 0] = -1
            sign[sign >= 0] = 1
            sign = np.cumprod(sign, axis=0, )

            qt[1:][sign < 0] *= -1

            return qt

        """
        @staticmethod
        def extract_heading_Y_up(q: np.ndarray):
            # extract the rotation around Y axis from given quaternions note the quaterions should be {(x,y,z,w)}
            q = np.asarray(q)
            shape = q.shape
            q = q.reshape(-1, 4)

            v = R(q, normalize=False, copy=False).as_matrix()[:, :, 1]

            # axis=np.cross(v,(0,1,0))
            axis = v[:, (2, 1, 0)]
            axis *= [-1, 0, 1]

            norms = np.linalg.norm(axis, axis=-1)
            scales = np.empty_like(norms)
            small_angle = (norms <= 1e-3)
            large_angle = ~small_angle

            scales[small_angle] = norms[small_angle] + norms[small_angle] ** 3 / 6
            scales[large_angle] = np.arccos(v[large_angle, 1]) / norms[large_angle]

            correct = R.from_rotvec(axis * scales[:, None])

            heading = (correct * R(q, normalize=False, copy=False)).as_quat()
            heading[heading[:, -1] < 0] *= -1

            return heading.reshape(shape)
        """

        @staticmethod
        def extract_heading_frame_Y_up(root_pos, root_rots):
            heading = pymotionlib.Utils.extract_heading_Y_up(root_rots)

            pos = np.copy(root_pos)
            pos[..., 1] = 0

            return pos, heading

        @staticmethod
        def get_joint_color(names, left='r', right='b', otherwise='y'):
            matches = (
                ('l', 'r'),
                ('L', 'R'),
                ('left', 'right'),
                ('Left', 'Right'),
                ('LEFT', 'RIGHT')
            )

            def check(n, i):
                for m in matches:
                    if n[:len(m[i])] == m[i] and m[1 - i] + n[len(m[i]):] in names:
                        return True

                    if n[-len(m[i]):] == m[i] and n[:-len(m[i])] + m[1 - i] in names:
                        return True

                return False

            color = [left if check(n, 0) else right if check(n, 1) else otherwise for n in names]
            return color

        @staticmethod
        def animate_motion_data(data, show_skeleton=True, show_animation=True, interval=1):
            if (not show_skeleton) and (not show_animation):
                return

            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from mpl_toolkits.mplot3d import Axes3D

            parent_idx = np.array(data._skeleton_joint_parents)
            parent_idx[0] = 0

            joint_colors = pymotionlib.utils.get_joint_color(data.joint_names)

            if data.end_sites is not None:
                for i in range(len(joint_colors)):
                    if i in data.end_sites:
                        joint_colors[i] = 'k'

            #############################
            # draw skeleton
            if show_skeleton:
                ref_joint_positions = data.get_reference_pose()
                tmp = ref_joint_positions.reshape(-1, 3)
                bound = np.array([np.min(tmp, axis=0), np.max(tmp, axis=0)])
                bound[1, :] -= bound[0, :]
                bound[1, :] = np.max(bound[1, :])
                bound[1, :] += bound[0, :]

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot('111', projection='3d')

                # ax.set_aspect('equal')
                ax.set_xlabel('x')
                ax.set_ylabel('z')
                ax.set_zlabel('y')

                pos = ref_joint_positions
                strokes = [plt.plot(xs=pos[(i, p), 0], zs=pos[(i, p), 1], ys=-pos[(i, p), 2], c=joint_colors[i], marker='x',
                                    linestyle='solid') for (i, p) in enumerate(parent_idx)]

                ax.auto_scale_xyz(bound[:, 0], -bound[:, 2], bound[:, 1])

            ########################################
            # animate motion
            if show_animation:
                joint_pos = data._joint_position
                tmp = joint_pos[:1].reshape(-1, 3)
                bound = np.array([np.min(tmp, axis=0), np.max(tmp, axis=0)])
                bound[1, :] -= bound[0, :]
                bound[1, :] = np.max(bound[1, :])
                bound[1, :] += bound[0, :]

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot('111', projection='3d')

                # ax.set_aspect('equal')
                ax.set_xlabel('x')
                ax.set_ylabel('z')
                ax.set_zlabel('y')

                pos = joint_pos[0]
                strokes = [plt.plot(xs=pos[(i, p), 0], zs=pos[(i, p), 1], ys=-pos[(i, p), 2], c=joint_colors[i], marker='x',
                                    linestyle='solid') for (i, p) in enumerate(parent_idx)]

                ax.auto_scale_xyz(bound[:, 0], -bound[:, 2], bound[:, 1])

                def update_lines(num):
                    for (i, p) in enumerate(parent_idx):
                        strokes[i][0].set_data(joint_pos[num][(i, p), 0], -joint_pos[num][(i, p), 2])
                        strokes[i][0].set_3d_properties(joint_pos[num][(i, p), 1])
                    plt.title('frame {num}'.format(num=num))

                line_ani = animation.FuncAnimation(
                    fig, update_lines, joint_pos.shape[0],
                    interval=interval, blit=False)

            plt.show()


    class BVHLoader:
        @staticmethod
        def load(fn: str, insert_T_pose: bool = False, ignore_root_offset=True, max_frames=None, ignore_root_xz_pos=False):
            with open(fn, 'r') as f:
                return pymotionlib.BVHLoader.load_from_io(f, fn, insert_T_pose, ignore_root_offset, max_frames, ignore_root_xz_pos)

        @staticmethod
        def load_from_string(
            bvh_str: str,
            insert_T_pose: bool = False,
            ignore_root_offset=True,
            max_frames=None,
            ignore_root_xz_pos=False
        ):
            import io
            return pymotionlib.BVHLoader.load_from_io(io.StringIO(bvh_str), 'str', insert_T_pose, ignore_root_offset, max_frames, ignore_root_xz_pos)

        @staticmethod
        def load_from_io(
            f,
            fn='',
            insert_T_pose: bool = False,
            ignore_root_offset=True,
            max_frames=None,
            ignore_root_xz_pos=False
        ):
            channels = []
            joints = []
            joint_parents = []
            joint_offsets = []
            end_sites = []
            fps = 0

            parent_stack = [None]
            for line in f:
                if 'ROOT' in line or 'JOINT' in line:
                    joints.append(line.split()[-1])
                    joint_parents.append(parent_stack[-1])
                    channels.append(None)
                    joint_offsets.append([0, 0, 0])

                elif 'End Site' in line:
                    end_sites.append(len(joints))

                    joints.append(parent_stack[-1] + '_end')
                    joint_parents.append(parent_stack[-1])
                    channels.append(None)
                    joint_offsets.append([0, 0, 0])

                elif '{' in line:
                    parent_stack.append(joints[-1])

                elif '}' in line:
                    parent_stack.pop()

                elif 'OFFSET' in line:
                    joint_offsets[-1] = [float(x) for x in line.split()[-3:]]

                elif 'CHANNELS' in line:
                    trans_order = []
                    trans_channels = []
                    rot_order = []
                    rot_channels = []
                    for i, token in enumerate(line.split()):
                        if 'position' in token:
                            trans_order.append(token[0])
                            trans_channels.append(i - 2)

                        if 'rotation' in token:
                            rot_order.append(token[0])
                            rot_channels.append(i - 2)

                    channels[-1] = [(''.join(trans_order), trans_channels), (''.join(rot_order), rot_channels)]

                elif 'Frame Time:' in line:
                    _frame_time = float(line.split()[-1])
                    logging.info(f'frame time: {_frame_time}\n')
                    fps = round(1. / _frame_time)
                    break

            values = []
            for line in f:
                tokens = line.split()
                if len(tokens) == 0:
                    break
                values.append([float(x) for x in tokens])
                if max_frames is not None and len(values) >= max_frames:
                    break

            values = np.array(values)
            # values = values.reshape(values.shape[0],-1,3)
            if insert_T_pose:
                values = np.concatenate((np.zeros_like(values[:1]), values), axis=0)

            assert (parent_stack[0] is None)
            data = pymotionlib.MotionData.MotionData()
            data._fps = fps

            data._skeleton_joints = joints
            data._skeleton_joint_parents = [joints.index(n) if n is not None else -1 for n in joint_parents]
            data._skeleton_joint_offsets = np.array(joint_offsets)
            data._end_sites = end_sites

            if ignore_root_offset:
                data._skeleton_joint_offsets[0].fill(0)

            data._num_frames = values.shape[0]
            data._num_joints = len(data._skeleton_joints)

            data._joint_translation = np.zeros((data._num_frames, data._num_joints, 3))
            data._joint_rotation = np.zeros((data._num_frames, data._num_joints, 4))
            data._joint_rotation[:, :, -1] = 1

            value_idx = 0
            for i, ch in enumerate(channels):
                if ch is None:
                    continue

                joint_num_channels = len(ch[0][1]) + len(ch[1][1])
                joint_values = values[:, value_idx:value_idx + joint_num_channels]
                value_idx += joint_num_channels

                if not ch[0][0] == '':
                    data._joint_translation[:, i] = joint_values[:, ch[0][1]]
                    if not ch[0] == 'XYZ':
                        data._joint_translation[:, i] = data._joint_translation[:, i][:, [ord(c) - ord('X') for c in ch[0][0]]]
                    if ignore_root_xz_pos:
                        data._joint_translation[:, i, [0, 2]] = 0.0

                if not ch[1][0] == '':
                    rot = R.from_euler(ch[1][0], joint_values[:, ch[1][1]], degrees=True)
                    data._joint_rotation[:, i] = rot.as_quat()

            logging.info('loaded %d frames @ %d fps from %s\n' % (data._num_frames, data._fps, fn))

            data._joint_position = None
            data._joint_orientation = None
            data.align_joint_rotation_representation()
            data.recompute_joint_global_info()
            data.to_contiguous()

            return data

        @staticmethod
        def save(data, fn: str, fmt: str = '%10.6f', euler_order: str = 'XYZ', translational_joints=False,
                insert_T_pose: bool = False):
            dirname = os.path.dirname(fn)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            with open(fn, 'w') as f:
                return pymotionlib.BVHLoader.save_to_io(data, f, fmt, euler_order, translational_joints, insert_T_pose)

        @staticmethod
        def save_as_string(data, fmt: str = '%10.6f', euler_order: str = 'XYZ', translational_joints=False,
                        insert_T_pose: bool = False):
            import io
            f = io.StringIO()
            pymotionlib.BVHLoader.save_to_io(data, f, fmt, euler_order, translational_joints, insert_T_pose)

            return f.getvalue()

        @staticmethod
        def save_to_io(data, f, fmt: str = '%10.6f', euler_order: str = 'XYZ', translational_joints=False,
                    insert_T_pose: bool = False):
            if not euler_order in ['XYZ', 'XZY', 'YZX', 'YXZ', 'ZYX', 'ZXY']:
                raise ValueError('euler_order ' + euler_order + ' is not supported!')

            # save header
            children = [[] for _ in range(data._num_joints)]
            for i, p in enumerate(data._skeleton_joint_parents[1:]):
                children[p].append(i + 1)

            tab = ' ' * 4
            f.write('HIERARCHY\n')
            f.write('ROOT ' + data._skeleton_joints[0] + '\n')
            f.write('{\n')
            f.write(tab + 'OFFSET ' + ' '.join(fmt % s for s in data._skeleton_joint_offsets[0]) + '\n')
            f.write(tab + 'CHANNELS 6 Xposition Yposition Zposition ' + ' '.join(c + 'rotation' for c in euler_order) + '\n')

            q = [(i, 1) for i in children[0][::-1]]
            last_level = 1
            output_order = [0]
            while len(q) > 0:
                idx, level = q.pop()
                output_order.append(idx)

                while last_level > level:
                    f.write(tab * (last_level - 1) + '}\n')
                    last_level -= 1

                indent = tab * level

                end_site = data._end_sites is not None and idx in data._end_sites
                if end_site:
                    f.write(indent + 'End Site\n')
                else:
                    f.write(indent + 'JOINT ' + data._skeleton_joints[idx] + '\n')

                f.write(indent + '{\n')
                level += 1
                indent += tab
                f.write(indent + 'OFFSET ' + ' '.join(fmt % s for s in data._skeleton_joint_offsets[idx]) + '\n')

                if not end_site:
                    if translational_joints:
                        f.write(indent + 'CHANNELS 6 Xposition Yposition Zposition ' + ' '.join(
                            c + 'rotation' for c in euler_order) + '\n')
                    else:
                        f.write(indent + 'CHANNELS 3 ' + ' '.join(c + 'rotation' for c in euler_order) + '\n')

                    q.extend([(i, level) for i in children[idx][::-1]])

                last_level = level

            while last_level > 0:
                f.write(tab * (last_level - 1) + '}\n')
                last_level -= 1

            f.write('MOTION\n')
            f.write('Frames: %d\n' % data.num_frames)
            f.write('Frame Time: ' + (fmt % (1 / data.fps)) + '\n')

            # prepare channels
            value_idx = 0
            num_channels = 6 + (6 if translational_joints else 3) * (
                    data._num_joints - 1 -
                    (len(data._end_sites) if data._end_sites is not None else 0))
            values = np.zeros((data.num_frames, num_channels))
            for i in output_order:
                if data._end_sites is not None and i in data._end_sites:
                    continue

                if i == 0 or translational_joints:
                    values[:, value_idx:value_idx + 3] = data._joint_translation[:, i]
                    value_idx += 3

                rot = R.from_quat(data._joint_rotation[:, i])
                values[:, value_idx:value_idx + 3] = rot.as_euler(euler_order, degrees=True)
                value_idx += 3

            # write frames
            if insert_T_pose:
                f.write(' '.join([fmt % 0] * num_channels))
                f.write('\n')
            f.write('\n'.join([' '.join(fmt % x for x in line) for line in values]))


    class ExtEndSite:
        # Add by Zhenhua Song
        @staticmethod
        def load_no_end_site(f: IO, ignore_root_offset=True, max_frames=None):
            channels = []
            joints = []
            joint_parents = []
            joint_offsets = []
            end_sites = []
            fps = 0

            parent_stack = [None]
            buf = f.readlines()
            buf_idx = 0
            while buf_idx < len(buf):
                line = buf[buf_idx].lstrip()
                buf_idx += 1
                if 'ROOT' in line or 'JOINT' in line:
                    joints.append(line.split()[-1])
                    joint_parents.append(parent_stack[-1])
                    channels.append(None)
                    joint_offsets.append([0, 0, 0])

                elif 'End Site' in line:
                    while "}" not in buf[buf_idx]:
                        buf_idx += 1
                    buf_idx += 1

                elif '{' in line:
                    parent_stack.append(joints[-1])

                elif '}' in line:
                    parent_stack.pop()

                elif 'OFFSET' in line:
                    joint_offsets[-1] = [float(x) for x in line.split()[-3:]]

                elif 'CHANNELS' in line:
                    trans_order = []
                    trans_channels = []
                    rot_order = []
                    rot_channels = []
                    for i, token in enumerate(line.split()):
                        if 'position' in token:
                            trans_order.append(token[0])
                            trans_channels.append(i - 2)

                        if 'rotation' in token:
                            rot_order.append(token[0])
                            rot_channels.append(i - 2)

                    channels[-1] = [(''.join(trans_order), trans_channels), (''.join(rot_order), rot_channels)]

                elif 'Frame Time:' in line:
                    _frame_time = float(line.split()[-1])
                    print('frame time: ', _frame_time)
                    fps = round(1. / _frame_time)
                    break

            values = []
            while buf_idx < len(buf):
                line = buf[buf_idx]
                buf_idx += 1
                tokens = line.split()
                if len(tokens) == 0:
                    break
                values.append([float(x) for x in tokens])
                if max_frames is not None and len(values) >= max_frames:
                    break

            values = np.array(values)

            assert (parent_stack[0] is None)
            data = pymotionlib.MotionData.MotionData()
            data._fps = fps

            data._skeleton_joints = joints
            data._skeleton_joint_parents = [joints.index(n) if n is not None else -1 for n in joint_parents]
            data._skeleton_joint_offsets = np.array(joint_offsets)
            data._end_sites = end_sites

            if ignore_root_offset:
                data._skeleton_joint_offsets[0].fill(0)

            data._num_frames = values.shape[0]
            data._num_joints = len(data._skeleton_joints)

            data._joint_translation = np.zeros((data._num_frames, data._num_joints, 3))
            data._joint_rotation = np.zeros((data._num_frames, data._num_joints, 4))
            data._joint_rotation[:, :, -1] = 1

            value_idx = 0
            for i, ch in enumerate(channels):
                if ch is None:
                    continue

                joint_num_channels = len(ch[0][1]) + len(ch[1][1])
                joint_values = values[:, value_idx:value_idx + joint_num_channels]
                value_idx += joint_num_channels

                if not ch[0][0] == '':
                    data._joint_translation[:, i] = joint_values[:, ch[0][1]]
                    if not ch[0] == 'XYZ':
                        data._joint_translation[:, i] = data._joint_translation[:, i][:, [ord(c) - ord('X') for c in ch[0][0]]]

                if not ch[1][0] == '':
                    rot = Rotation.from_euler(ch[1][0], joint_values[:, ch[1][1]], degrees=True)
                    data._joint_rotation[:, i] = rot.as_quat()

            print('loaded %d frames @ %d fps' % (data._num_frames, data._fps))

            data._joint_position = None
            data._joint_orientation = None
            data.align_joint_rotation_representation()
            data.recompute_joint_global_info()

            return data


        # assume data doesn't have end site
        # if a joint has no child, there must be a end site
        # the default offset is (0, 0, 0)
        @staticmethod
        def save_ext_end_site(data, f: IO, fmt: str = '%10.6f',
                            euler_order: str = 'XYZ',
                            ext_end_site: Optional[Dict[int, np.ndarray]] = None):
            if data.end_sites:
                raise ValueError("Assume data dose not have end site.")

            if not euler_order in ['XYZ', 'XZY', 'YZX', 'YXZ', 'ZYX', 'ZXY']:
                raise ValueError('euler_order ' + euler_order + ' is not supported!')

            # save header
            children = [[] for _ in range(data._num_joints)]
            for i, p in enumerate(data._skeleton_joint_parents[1:]):
                children[p].append(i + 1)

            tab = ' ' * 4
            f.write('HIERARCHY\nROOT ' + data._skeleton_joints[0] + '\n{\n')
            f.write(tab + 'OFFSET ' + ' '.join(fmt % s for s in data._skeleton_joint_offsets[0]) + '\n')
            f.write(tab + 'CHANNELS 6 Xposition Yposition Zposition ' + ' '.join(c + 'rotation' for c in euler_order) + '\n')

            q = [(i, 1) for i in children[0][::-1]]
            last_level = 1
            output_order = [0]
            while len(q) > 0:
                idx, level = q.pop()
                output_order.append(idx)

                while last_level > level:
                    f.write(tab * (last_level - 1) + '}\n')
                    last_level -= 1

                indent = tab * level
                f.write(indent + 'JOINT ' + data._skeleton_joints[idx] + '\n')
                f.write(indent + '{\n')
                level += 1
                indent += tab
                f.write(indent + 'OFFSET ' + ' '.join(fmt % s for s in data._skeleton_joint_offsets[idx]) + '\n')
                f.write(indent + 'CHANNELS 3 ' + ' '.join(c + 'rotation' for c in euler_order) + '\n')

                if len(children[idx]) == 0:
                    if ext_end_site is None:
                        offset = np.array([0, 0, 0])  # Add default end site: offset (0, 0, 0)
                    else:
                        offset = ext_end_site[idx]
                    f.write(tab * level + "End Site\n")
                    f.write(tab * level + "{\n")
                    f.write(tab * (level + 1) + "OFFSET " + ' '.join(fmt % s for s in offset) + '\n')
                    f.write(tab * level + "}\n")

                q.extend([(i, level) for i in children[idx][::-1]])
                last_level = level

            while last_level > 0:
                f.write(tab * (last_level - 1) + '}\n')
                last_level -= 1

            # Write frames
            f.write('MOTION\n')
            f.write('Frames: %d\n' % data.num_frames)
            f.write('Frame Time: ' + (fmt % (1 / data.fps)) + '\n')

            # prepare channels
            value_idx = 0
            num_channels = 6 + 3 * (data._num_joints - 1)
            values = np.zeros((data.num_frames, num_channels))
            for i in output_order:
                if data._end_sites is not None and i in data._end_sites:
                    continue

                if i == 0:
                    values[:, value_idx:value_idx + 3] = data._joint_translation[:, i]
                    value_idx += 3

                rot = Rotation.from_quat(data._joint_rotation[:, i])
                values[:, value_idx:value_idx + 3] = rot.as_euler(euler_order, degrees=True)
                value_idx += 3

            f.write('\n'.join([' '.join(fmt % x for x in line) for line in values]))


    class MotionHelper:
        @staticmethod
        def calc_children(data):
            children = [[] for _ in range(data._num_joints)]
            for i, p in enumerate(data._skeleton_joint_parents[1:]):
                children[p].append(i + 1)
            return children

        @staticmethod
        def calc_name_idx(data) -> Dict[str, int]:
            return dict(zip(data.joint_names, range(len(data.joint_names))))

        @staticmethod
        def adjust_root_height(data, dh: Optional[float] = None):
            if dh is None:
                min_y_pos = np.min(data._joint_position[:, :, 1], axis=1)
                min_y_pos[min_y_pos > 0] = 0
                dy = np.min(min_y_pos)
            else:
                dy = dh
            data._joint_position[:, :, 1] -= dy
            data._joint_translation[:, 0, 1] -= dy

    class MotionData:
        class MotionData:

            __slots__ = (
                "_skeleton_joints",
                "_skeleton_joint_parents",
                "_skeleton_joint_offsets",
                "_end_sites",
                "_num_joints",
                "_num_frames",
                "_fps",
                "_joint_rotation",
                "_joint_translation",
                "_joint_position",
                "_joint_orientation"
            )

            def __init__(self) -> None:
                # skeleton
                self._skeleton_joints: Optional[List[str]] = None  # name of each joint
                self._skeleton_joint_parents: Optional[List[int]] = None
                self._skeleton_joint_offsets: Optional[np.ndarray] = None  # Joint OFFSET in BVH file
                self._end_sites: Optional[List[int]] = None
                self._num_joints = 0

                # animation
                self._num_frames = 0
                self._fps = 0

                self._joint_rotation: Optional[np.ndarray] = None  # joint local rotation
                self._joint_translation: Optional[np.ndarray] = None  # [:, 0, :] is root position. other is zero.

                # pre-computed global information
                self._joint_position: Optional[np.ndarray] = None
                self._joint_orientation: Optional[np.ndarray] = None

            @property
            def joint_rotation(self) -> Optional[np.ndarray]:
                return self._joint_rotation

            @property
            def joint_translation(self) -> Optional[np.ndarray]:
                return self._joint_translation

            @property
            def num_frames(self) -> int:
                return self._num_frames

            @property
            def num_joints(self) -> int:
                return self._num_joints

            @property
            def joint_position(self) -> Optional[np.ndarray]:
                return self._joint_position

            @property
            def joint_orientation(self) -> Optional[np.ndarray]:
                return self._joint_orientation

            @property
            def joint_parents_idx(self):
                return self._skeleton_joint_parents

            @property
            def joint_names(self) -> Optional[List[str]]:
                return self._skeleton_joints

            @property
            def end_sites(self) -> Optional[List[int]]:
                return self._end_sites

            # Add by Zhenhua Song
            def get_end_flags(self) -> np.ndarray:
                flags = [0] * self._num_joints
                if self._end_sites:
                    for end_idx in self._end_sites:
                        flags[end_idx] = 1
                return np.array(flags)

            @property
            def joint_offsets(self):
                return self._skeleton_joint_offsets

            @property
            def fps(self):
                return self._fps

            # Add by Zhenhua Song
            def remove_end_sites(self, copy: bool = True):
                ret = self.sub_sequence(copy=copy)
                if not ret._end_sites:
                    return ret

                # modify attr index
                joint_idx: np.ndarray = np.arange(0, ret._num_joints, dtype=np.uint64)
                joint_idx: np.ndarray = np.delete(joint_idx, np.array(ret._end_sites, dtype=np.uint64))
                if ret._joint_translation is not None:
                    ret._joint_translation = ret._joint_translation[:, joint_idx, :]
                if ret._joint_rotation is not None:
                    ret._joint_rotation = ret._joint_rotation[:, joint_idx, :]
                if ret._joint_position is not None:
                    ret._joint_position = ret._joint_position[:, joint_idx, :]
                if ret._joint_orientation is not None:
                    ret._joint_orientation = ret._joint_orientation[:, joint_idx, :]

                # modify parent index
                for i in range(len(ret._end_sites)):
                    end_idx = ret._end_sites[i] - i
                    before = ret._skeleton_joint_parents[:end_idx]
                    after = ret._skeleton_joint_parents[end_idx + 1:]
                    for j in range(len(after)):
                        if after[j] > end_idx:
                            after[j] -= 1
                    ret._skeleton_joint_parents = before + after

                # modify other attributes
                ret._skeleton_joints = np.array(ret._skeleton_joints)[joint_idx].tolist()
                ret._skeleton_joint_offsets = np.array(ret._skeleton_joint_offsets)[joint_idx]
                ret._num_joints -= len(ret._end_sites)
                ret._end_sites.clear()

                return ret.to_contiguous()

            # Add by Zhenhua Song
            def set_anim_attrs(self, num_frames, fps):
                self._num_frames = num_frames
                self._fps = fps
                self._joint_translation = np.zeros((num_frames, self._num_joints, 3))

            # Add by Zhenhua Song
            def to_contiguous(self):
                if self._joint_rotation is not None:
                    self._joint_rotation = np.ascontiguousarray(self._joint_rotation)
                if self._joint_translation is not None:
                    self._joint_translation = np.ascontiguousarray(self._joint_translation)
                if self._joint_orientation is not None:
                    self._joint_orientation = np.ascontiguousarray(self._joint_orientation)
                if self._joint_position is not None:
                    self._joint_position = np.ascontiguousarray(self._joint_position)

                return self

            def align_joint_rotation_representation(self):
                """ make sure that the quaternions are aligned
                """
                if self._joint_rotation is not None:
                    pymotionlib.Utils.align_quaternion(self._joint_rotation, True)

                return self

            def reset_global_info(self):
                self._joint_position: Optional[np.ndarray] = None
                self._joint_orientation: Optional[np.ndarray] = None

            def compute_joint_global_info(self, joint_translation: np.ndarray, joint_rotation: np.ndarray,
                                        joint_position: np.ndarray = None, joint_orientation: np.ndarray = None):
                """ compute global information based on given local information
                """

                joint_translation = np.asarray(joint_translation).reshape((-1, self._num_joints, 3))
                joint_rotation = np.asarray(joint_rotation).reshape((-1, self._num_joints, 4))

                num_frames, num_joints = joint_rotation.shape[:2]
                if joint_position is None:
                    joint_position = np.zeros((num_frames, num_joints, 3))
                else:
                    joint_position.fill(0)
                    joint_position = joint_position.reshape((num_frames, num_joints, 3))

                if joint_orientation is None:
                    joint_orientation = np.zeros((num_frames, num_joints, 4))
                else:
                    joint_orientation.fill(0)
                    joint_orientation = joint_orientation.reshape((num_frames, num_joints, 4))

                for i, pi in enumerate(self._skeleton_joint_parents):
                    joint_position[:, i, :] = joint_translation[:, i, :] + self._skeleton_joint_offsets[i, :]

                    joint_orientation[:, i, :] = joint_rotation[:, i, :]

                    if pi < 0:
                        assert (i == 0)
                        continue

                    parent_orient = R(joint_orientation[:, pi, :], normalize=False, copy=False)
                    joint_position[:, i, :] = parent_orient.apply(joint_position[:, i, :]) + joint_position[:, pi, :]
                    joint_orientation[:, i, :] = (
                                parent_orient * R(joint_orientation[:, i, :], normalize=False, copy=False)).as_quat()
                    joint_orientation[:, i, :] /= np.linalg.norm(joint_orientation[:, i, :], axis=-1, keepdims=True)

                return joint_position, joint_orientation

            def recompute_joint_global_info(self):
                #########
                # now pre-compute joint global positions and orientations
                self._joint_position, self._joint_orientation = self.compute_joint_global_info(
                    self._joint_translation, self._joint_rotation, self._joint_position, self._joint_orientation)

                pymotionlib.Utils.align_quaternion(self._joint_orientation, True)

                return self

            def compute_joint_local_info(self, joint_position: np.ndarray, joint_orientation: np.ndarray,
                                        joint_translation: np.ndarray = None, joint_rotation: np.ndarray = None):
                """ compute local information based on given global information
                """

                joint_position = np.asarray(joint_position).reshape((-1, self._num_joints, 3))
                joint_orientation = np.asarray(joint_orientation).reshape((-1, self._num_joints, 4))

                num_frames, num_joints = joint_position.shape[:2]
                if joint_translation is None:
                    joint_translation = np.zeros((num_frames, num_joints, 3))
                else:
                    joint_translation.fill(0)
                    joint_translation = joint_translation.reshape((num_frames, num_joints, 3))

                if joint_rotation is None:
                    joint_rotation = np.zeros((num_frames, num_joints, 4))
                else:
                    joint_rotation.fill(0)
                    joint_rotation = joint_rotation.reshape((num_frames, num_joints, 4))

                joint_translation[:, 0] = joint_position[:, 0]
                joint_translation[:, 1:] = joint_position[:, 1:] - joint_position[:, self._skeleton_joint_parents[1:]]
                joint_translation[:, 1:] = R(joint_orientation[:, self._skeleton_joint_parents[1:]].ravel().reshape(-1, 4),
                                            normalize=False, copy=False).apply(
                    joint_translation[:, 1:].reshape(-1, 3), inverse=True
                ).reshape((num_frames, num_joints - 1, 3))

                joint_translation[:, 1:] -= self._skeleton_joint_offsets[1:]

                joint_rotation[:, 0] = joint_orientation[:, 0]
                joint_rotation[:, 1:] = pymotionlib.Utils.quat_product(
                    joint_orientation[:, self._skeleton_joint_parents[1:]],
                    joint_orientation[:, 1:],
                    inv_p=True)

                return joint_translation, joint_rotation

            def resample(self, new_fps: int):
                if new_fps == self.fps:
                    return self

                if self.num_frames == 1:
                    self._fps = new_fps
                    return self

                from scipy.spatial.transform import Rotation, Slerp
                from scipy.interpolate import interp1d

                length = (self.num_frames - 1) / self.fps
                new_num_frames = int(np.floor(length * new_fps)) + 1

                # if comm_rank == 0:
                #    print('fps: %d -> %d' % (self.fps, new_fps))
                #    print('num frames: %d -> %d' % (self.num_frames, new_num_frames), flush=True)

                ticks = np.arange(0, self.num_frames, dtype=np.float64) / self.fps
                new_ticks = np.arange(0, new_num_frames, dtype=np.float64) / new_fps

                # deal with root position with linear interp
                joint_trans_interp = interp1d(ticks, self._joint_translation, kind='linear', axis=0, copy=False,
                                            bounds_error=True, assume_sorted=True)
                self._joint_translation = joint_trans_interp(new_ticks)

                # handle joint rotations using slerp
                cur_joint_rots = self._joint_rotation
                num_joints = self.num_joints

                self._joint_rotation = np.zeros((new_num_frames, num_joints, 4))
                for i in range(num_joints):
                    rotations = Rotation.from_quat(cur_joint_rots[:, i])
                    self._joint_rotation[:, i] = Slerp(ticks, rotations)(new_ticks).as_quat()

                self._num_frames = new_num_frames
                self._fps = new_fps

                self._joint_position = None
                self._joint_orientation = None

                pymotionlib.Utils.align_quaternion(self._joint_rotation, True)
                self.recompute_joint_global_info()

                # print("After bvh resample")
                return self

            def sub_sequence(self, start: Optional[int] = None,
                            end: Optional[int] = None,
                            skip: Optional[int] = None, copy: bool = True):
                sub = pymotionlib.MotionData.MotionData()

                sub._skeleton_joints = self._skeleton_joints
                sub._skeleton_joint_parents = self._skeleton_joint_parents
                sub._skeleton_joint_offsets = self._skeleton_joint_offsets
                sub._num_joints = self._num_joints

                sub._end_sites = self._end_sites
                sub._fps = self._fps

                key = slice(start, end, skip)

                sub._joint_rotation = self._joint_rotation[key] if self._joint_rotation is not None else None
                sub._joint_translation = self._joint_translation[key] if self._joint_translation is not None else None

                sub._joint_position = self._joint_position[key] if self._joint_position is not None else None
                sub._joint_orientation = self._joint_orientation[key] if self._joint_orientation is not None else None

                sub._num_frames = sub._joint_rotation.shape[0] if self._joint_rotation is not None else 0

                if copy:
                    import copy
                    sub._skeleton_joints = copy.deepcopy(sub._skeleton_joints)
                    sub._skeleton_joint_parents = copy.deepcopy(sub._skeleton_joint_parents)
                    sub._skeleton_joint_offsets = copy.deepcopy(sub._skeleton_joint_offsets)
                    sub._num_joints = sub._num_joints

                    sub._end_sites = copy.deepcopy(sub._end_sites)

                    sub._joint_rotation = sub._joint_rotation.copy() if sub._joint_rotation is not None else None
                    sub._joint_translation = sub._joint_translation.copy() if sub._joint_translation is not None else None

                    sub._joint_position = sub._joint_position.copy() if sub._joint_position is not None else None
                    sub._joint_orientation = sub._joint_orientation.copy() if sub._joint_orientation is not None else None

                return sub

            # Add by Zhenhua Song
            def get_t_pose(self):
                ret = self.sub_sequence(0, 1)
                ret._joint_translation.fill(0)
                ret._joint_rotation.fill(0)
                ret._joint_rotation[..., 3] = 1
                ret.recompute_joint_global_info()
                return ret

            def get_hierarchy(self, copy: bool = False):
                """
                Get bvh hierarchy
                """
                return self.sub_sequence(self._num_frames, self._num_frames, copy=copy)

            def append_trans_rotation(self, trans: np.ndarray, rotation: np.ndarray):
                self._num_frames += trans.shape[0]
                if self._joint_rotation is not None:
                    self._joint_rotation = np.concatenate([self._joint_rotation, rotation], axis=0)
                else:
                    self._joint_rotation = rotation.copy()

                if self._joint_translation is not None:
                    self._joint_translation = np.concatenate([self._joint_translation, trans], axis=0)
                else:
                    self._joint_translation = trans.copy()

            def append(self, other_):
                import operator
                other = other_
                assert self.fps == other.fps
                assert operator.eq(self.joint_names, other.joint_names)
                assert operator.eq(self.joint_parents_idx, other.joint_parents_idx)
                assert operator.eq(self.end_sites, other.end_sites)
                self._num_frames += other._num_frames

                if self.joint_rotation is not None:
                    self._joint_rotation = np.concatenate([self._joint_rotation, other._joint_rotation], axis=0)
                else:
                    self._joint_rotation = other._joint_rotation.copy()

                if self._joint_translation is not None:
                    self._joint_translation = np.concatenate([self._joint_translation, other._joint_translation], axis=0)
                else:
                    self._joint_translation = other._joint_translation.copy()

                if self._joint_position is not None:
                    self._joint_position = np.concatenate([self._joint_position, other._joint_position], axis=0)
                else:
                    self._joint_position = other._joint_position.copy()

                if self._joint_orientation is not None:
                    self._joint_orientation = np.concatenate([self._joint_orientation, other._joint_orientation], axis=0)
                else:
                    self._joint_translation = other._joint_translation.copy()

                return self

            def scale(self, factor: float):
                self._skeleton_joint_offsets *= factor

                if self._joint_translation is not None:
                    self._joint_translation *= factor

                if self._joint_position is not None:
                    self._joint_position *= factor

                return self

            def compute_linear_velocity(self, forward: bool = False):
                """ compute linear velocities of every joint using finite difference

                    the velocities are in the world coordinates

                    return: an array of size (num_frame, num_joint, 3),
                        for forward/backward difference, the last/first frame is the
                        frame next to it
                """
                if self._joint_position is None:
                    self.recompute_joint_global_info()
                # Modified by Zhenhua Song
                # v = np.empty_like(self._joint_position)
                # frag = v[:-1] if forward else v[1:]
                # frag[:] = np.diff(self._joint_position, axis=0) * self._fps

                v: np.ndarray = np.zeros_like(self._joint_position)
                frag: np.ndarray = np.diff(self._joint_position, axis=0) * self._fps
                if forward:
                    v[:-1] = frag  # v[t] = x[t + 1] - x[t]. e.g. v[0] = x[1] - x[0], v[1] = x[2] - x[1]
                else:
                    v[1:] = frag  # v[t] = x[t] - x[t - 1]. e.g. v[1] = x[1] - x[0]

                v[-1 if forward else 0] = v[-2 if forward else 1]
                return v

            def compute_angular_velocity(self, forward: bool = False):
                """ compute angular velocities of every joint using finite difference

                    the velocities are in the world coordinates

                    forward: if True, we compute w_n = 2 (q_n+1 - q_n) * q_n.inv() ,
                        otherwise, we compute w_n = 2 (q_n - q_n-1) * q_n-1.inv()

                    return: an array of size (num_frame, num_joint, 3),
                        for forward/backward difference, the last/first frame is the
                        frame next to it
                """
                if self._joint_orientation is None:
                    self.recompute_joint_global_info()

                qd = np.diff(self._joint_orientation, axis=0) * self._fps

                # note that we cannot use R(q).inv() here, because scipy implement inv() as
                #  (x,y,z,w).inv() = (x,y,z,-w)
                # which is not the conjugate!
                
                # modified by heyuan Yao 
                q = self._joint_orientation[:-1] #if forward else self._joint_orientation[1:]
                q_conj = q.copy().reshape(-1, 4)
                q_conj[:, :3] *= -1
                qw = pymotionlib.Utils.quat_product(qd.reshape(-1, 4), q_conj)

                # Modify By Zhenhua Song
                # w = np.empty((self._num_frames, self._num_joints, 3))
                # frag = w[:-1] if forward else w[1:]
                # frag[:] = qw[:, :3].reshape(self._num_frames - 1, self._num_joints, 3)
                # frag[:] *= 2
                w = np.zeros((self._num_frames, self._num_joints, 3))
                frag = 2 * qw[:, :3].reshape(self._num_frames - 1, self._num_joints, 3)
                if forward:
                    w[:-1] = frag
                else:
                    w[1:] = frag

                w[-1 if forward else 0] = w[-2 if forward else 1]
                return w

            def compute_translational_speed(self, forward: bool):
                """ compute the `local` translational velocities of every joint using finite difference

                    note that different from `compute_linear_velocity`, this is the relative
                    speed of joints wrt. their parents, and the values are represented in the
                    parents' local coordinates

                    return: an array of size (num_frame, num_joint, 3),
                        for forward/backward difference, the last/first frame is the
                        frame next to it
                """
                # Modify by Zhenhua Song
                # v = np.empty_like(self._joint_translation)
                # frag = v[:-1] if forward else v[1:]
                # frag[:] = np.diff(self._joint_translation, axis=0) * self._fps
                v = np.zeros_like(self._joint_translation)
                frag = np.diff(self._joint_translation, axis=0) * self._fps
                if forward:
                    v[:-1] = frag
                else:
                    v[1:] = frag
                v[-1 if forward else 0] = v[-2 if forward else 1]
                return v

            def compute_rotational_speed(self, forward: bool):
                """ compute the `local` rotational speed of every joint using finite difference

                    note that different from `compute_angular_velocity`, this is the relative
                    speed of joints wrt. their parents, and the values are represented in the
                    parents' local coordinates

                    forward: if True, we compute w_n = 2 (q_n+1 - q_n) * q_n.inv() ,
                        otherwise, we compute w_n = 2 (q_n - q_n-1) * q_n.inv()

                    return: an array of size (num_frame, num_joint, 3),
                        for forward/backward difference, the last/first frame is the
                        frame next to it
                """
                qd = np.diff(self._joint_rotation, axis=0) * self._fps

                # note that we cannot use R(q).inv() here, because scipy implement inv() as
                #  (x,y,z,w).inv() = (x,y,z,-w)
                # which is not the conjugate!
                # modified by heyuan Yao 
                q = self._joint_rotation[:-1]
                q_conj = q.copy().reshape(-1, 4)
                q_conj[:, :3] *= -1
                qw = pymotionlib.Utils.quat_product(qd.reshape(-1, 4), q_conj)

                w = np.zeros((self._num_frames, self._num_joints, 3))
                # frag = w[:-1] if forward else w[1:]
                # frag[:] = qw[:, :3].reshape(self._num_frames - 1, self._num_joints, 3)
                # frag[:] *= 2

                # Modify By Zhenhua Song
                frag = 2 * qw[:, :3].reshape(self._num_frames - 1, self._num_joints, 3)
                if forward:
                    w[:-1] = frag
                else:
                    w[1:] = frag

                w[-1 if forward else 0] = w[-2 if forward else 1].copy()
                return w

            def reconfig_reference_pose(self,
                                        rotations: Union[List[np.ndarray], np.ndarray, Dict[str, np.ndarray]],
                                        treat_as_global_orientations: bool,
                                        treat_as_reverse_rotation: bool
                                        ):
                """ reconfigurate the reference pose (T pose) of this bvh object
                Parameters
                -------
                rotations: rotations on the current T pose

                treat_as_global_orientations: if true, the input rotations will be treat as
                    target orientations of the bones

                treat_as_reverse_rotation: if true, the input rotations are considered as those
                    rotating the target pose to the current pose
                """

                if isinstance(rotations, list):
                    rotations += [np.array([0, 0, 0, 1.0]) for i in range(self.num_joints - len(rotations))]
                    rotations = np.array(rotations)
                elif isinstance(rotations, np.ndarray):
                    rotations = np.concatenate((rotations, np.tile([0, 0, 0, 1.0], (self.num_joints - len(rotations), 1))),
                                            axis=0)
                elif isinstance(rotations, dict):
                    rotations_ = np.array([[0, 0, 0, 1.0] for i in range(self.num_joints)])
                    for jnt, rot in rotations.items():
                        idx = self.joint_names.index(jnt)
                        rotations_[idx, :] = rot
                    rotations = rotations_
                else:
                    raise ValueError('unsupported type: rotations (' + type(rotations) + ')')

                rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)

                if not treat_as_global_orientations:
                    for i, p in enumerate(self.joint_parents_idx[1:]):
                        idx = i + 1
                        r = R(rotations[p])
                        rotations[idx, :] = (r * R(rotations[idx])).as_quat()
                        rotations[idx, :] /= np.sqrt(np.sum(rotations[idx, :] ** 2))

                if treat_as_reverse_rotation:
                    rotations[:, -1] = -rotations[:, -1]

                rotations = R(rotations)

                for i, p in enumerate(self.joint_parents_idx):
                    new_rot = R.from_quat(self.joint_rotation[:, i]) * rotations[i].inv()
                    if p >= 0:
                        new_rot = rotations[p] * new_rot
                        self._skeleton_joint_offsets[i] = rotations[p].apply(self._skeleton_joint_offsets[i])

                        self._joint_translation[:, i] = rotations[p].apply(self._joint_translation[:, i])

                    self._joint_rotation[:, i] = new_rot.as_quat()

                self._joint_rotation /= np.linalg.norm(self._joint_rotation, axis=-1, keepdims=True)

                pymotionlib.Utils.align_quaternion(self._joint_rotation, True)
                self.recompute_joint_global_info()

                return self

            def get_mirror_joint_indices(self):
                indices = list(range(self._num_joints))

                def index(name):
                    try:
                        return self._skeleton_joints.index(name)
                    except ValueError:
                        return -1

                for i, n in enumerate(self._skeleton_joints):
                    # rule 1: left->right
                    idx = -1
                    if n.find('left') == 0:
                        idx = index('right' + n[4:])
                    elif n.find('Left') == 0:
                        idx = index('Right' + n[4:])
                    elif n.find('LEFT') == 0:
                        idx = index('RIGHT' + n[4:])
                    elif n.find('right') == 0:
                        idx = index('left' + n[5:])
                    elif n.find('Right') == 0:
                        idx = index('Left' + n[5:])
                    elif n.find('RIGHT') == 0:
                        idx = index('LEFT' + n[5:])
                    elif n.find('L') == 0:
                        idx = index('R' + n[1:])
                    elif n.find('l') == 0:
                        idx = index('r' + n[1:])
                    elif n.find('R') == 0:
                        idx = index('L' + n[1:])
                    elif n.find('r') == 0:
                        idx = index('l' + n[1:])

                    indices[i] = idx if idx >= 0 else i

                return indices

            def symmetrize_skeleton(self, plane_of_symmetry_normal: Union[List[float], np.ndarray],
                                    mirror_joint_indices: Union[None, List[int]]):
                """ fix skeleton joint offsets to make the skeleton symmetric

                Parameters
                ----------
                plane_of_symmetry_normal : the normal of the plan of symmetry of the skeleton
                    note that the

                mirror_joint_indices: should be the index of the mirror joint of a joint
                            if not provided, get_mirror_joint_indices() will be called to get a best estimation

                """
                if mirror_joint_indices is None:
                    mirror_joint_indices = self.get_mirror_joint_indices()

                mirror_offsets = pymotionlib.Utils.flip_vector(self._skeleton_joint_offsets, plane_of_symmetry_normal, inplace=False)
                self._skeleton_joint_offsets += mirror_offsets[mirror_joint_indices]
                self._skeleton_joint_offsets /= 2

                self.recompute_joint_global_info()

                return self

            def flip(self, plane_of_symmetry_normal: Union[List[float], np.ndarray],
                    mirror_joint_indices: Union[None, List[int]] = None):
                """ flip the animation wrt the plane of symmetry while assuming the plane passes the origin point

                Note that if the character is not symmetric or if a wrong normal vector is given, the result will not look good

                Parameters
                ----------
                plane_of_symmetry_normal : the normal of the plan of symmetry of the skeleton
                    note that the

                mirror_joint_indices: should be the index of the mirror joint of a joint
                            if not provided, get_mirror_joint_indices() will be called to get a best estimation


                Returns
                -------
                None
                """
                pymotionlib.Utils.flip_quaternion(self._joint_rotation.reshape(-1, 4), plane_of_symmetry_normal, inplace=True)
                pymotionlib.Utils.flip_vector(self._joint_translation.reshape(-1, 3), plane_of_symmetry_normal, inplace=True)

                if mirror_joint_indices is None:
                    mirror_joint_indices = self.get_mirror_joint_indices()

                self._joint_rotation[:] = self._joint_rotation[:, mirror_joint_indices]
                self._joint_translation[:] = self._joint_translation[:, mirror_joint_indices]

                pymotionlib.Utils.align_quaternion(self._joint_rotation, True)
                self.recompute_joint_global_info()

                return self

            def get_reference_pose(self):
                pos = self.joint_offsets.copy()
                for i, p in enumerate(self.joint_parents_idx[1:]):
                    pos[i + 1] += pos[p]

                return pos

            def retarget(self, joint_map: Dict[str, Union[str, List[str]]]):
                """ create a new skeleton based on the joint map and retarget the motion to it

                the hierarchy of current skeleton will be maintained.

                """

                joint_map_inv = [None] * self._num_joints
                try:
                    for k, v in joint_map.items():
                        if isinstance(v, str):
                            joint_map_inv[self._skeleton_joints.index(v)] = k
                        else:
                            for v_ in v:
                                joint_map_inv[self._skeleton_joints.index(v_)] = k

                except ValueError:
                    print('cannot find joint', v)
                    raise

                if joint_map_inv[0] is None:
                    print('root joint is not specified')
                    raise ValueError('root joint is not specified')

                ref_pose = self.get_reference_pose()

                data = pymotionlib.Utils.MotionData()
                data._skeleton_joints = [joint_map_inv[0]]
                data._skeleton_joint_parents = [-1]
                data._skeleton_joint_offsets = [ref_pose[0]]

                for i, n in enumerate(joint_map_inv[1:]):
                    if n is None:
                        continue

                    if n in data._skeleton_joints:
                        continue

                    idx = i + 1

                    data._skeleton_joints.append(n)
                    p = self._skeleton_joint_parents[idx]
                    while p >= 0:
                        if joint_map_inv[p] is not None:
                            break
                        p = self._skeleton_joint_parents[p]

                    if p < 0:
                        print('cannot find the parent joint for', n)
                        raise ValueError('cannot find the parent joint for ' + str(n))

                    while (self._skeleton_joint_parents[p] >= 0 and
                        joint_map_inv[self._skeleton_joint_parents[p]] == joint_map_inv[p]):
                        p = self._skeleton_joint_parents[p]

                    data._skeleton_joint_parents.append(data._skeleton_joints.index(joint_map_inv[p]))
                    data._skeleton_joint_offsets.append(ref_pose[idx] - ref_pose[p])

                data._num_joints = len(joint_map)
                data._skeleton_joint_offsets = np.asarray(data._skeleton_joint_offsets)

                # now retarget the motion by copying the data
                data._num_frames = self._num_frames
                data._fps = self._fps

                data._joint_rotation = np.zeros((data._num_frames, data._num_joints, 4))
                data._joint_rotation.reshape(-1, 4)[:, -1] = 1
                data._joint_translation = np.zeros((data._num_frames, data._num_joints, 3))

                for i, n in enumerate(joint_map_inv):
                    if n is None:
                        continue

                    idx = data._skeleton_joints.index(n)
                    data._joint_rotation[:, idx] = (R.from_quat(data._joint_rotation[:, idx]) *
                                                    R.from_quat(self._joint_rotation[:, i])).as_quat()
                    data._joint_rotation[:, idx] /= np.linalg.norm(data._joint_rotation[:, idx], axis=-1, keepdims=True)

                    data._joint_translation[:, idx] += self._joint_translation[:, i]

                pymotionlib.Utils.align_quaternion(data._joint_rotation, True)
                data.recompute_joint_global_info()

                return data

            def remore_reference_nodes(self, new_root):
                """ create a new skeleton with the root joint as specified

                    some software may export motions with 'reference node', this function will remove those node and bake the
                    corresponding transformations into the new root

                    note that we only allows a single root joint, so that the siblings of the new_root will be removed
                """
                try:
                    new_root_idx = self.joint_names.index(new_root)
                except ValueError:
                    raise ValueError('cannot find joint ' + new_root)

                data = pymotionlib.Utils.MotionData()

                keep_joints = np.zeros(self.num_joints, dtype=bool)
                keep_joints[new_root_idx] = True
                for i in range(new_root_idx + 1, self.num_joints):
                    keep_joints[i] = keep_joints[self.joint_parents_idx[i]]

                new_joint_indices = np.cumsum(keep_joints.astype(int)) - 1

                # skeleton
                data._skeleton_joints = list(np.asarray(self.joint_names)[keep_joints])
                data._skeleton_joint_parents = list(new_joint_indices[np.asarray(self.joint_parents_idx)[keep_joints]])
                data._skeleton_joint_offsets = np.asarray(self.joint_offsets)[keep_joints]
                data._end_sites = None if self.end_sites is None else list(
                    new_joint_indices[list(n for n in self._end_sites if keep_joints[n])])
                data._num_joints = len(data._skeleton_joints)

                # animation
                data._num_frames = self._num_frames
                data._fps = self._fps

                data._joint_rotation = self._joint_rotation[:, keep_joints, :]
                data._joint_translation = self._joint_translation[:, keep_joints, :]

                data._joint_rotation[:, 0, :] = self.joint_orientation[:, new_root_idx, :]
                data._joint_translation[:, 0, :] = self.joint_position[:, new_root_idx, :]

                data.recompute_joint_global_info()

                return data

            # Add by Zhenhua Song
            def z_up_to_y_up(self):
                # 1. change root quaternion
                delta_rot: R = R.from_rotvec(np.array([-0.5 * np.pi, 0.0, 0.0]))
                self._joint_rotation[:, 0, :] = (delta_rot * R(self._joint_rotation[:, 0, :])).as_quat()

                # 2. change root position
                root_pos_y: np.ndarray = self._joint_translation[:, 0, 1].copy()
                root_pos_z: np.ndarray = self._joint_translation[:, 0, 2].copy()
                self._joint_translation[:, 0, 1] = root_pos_z
                self._joint_translation[:, 0, 2] = -root_pos_y

                # 3. we should NOT change local offset..
                # off_y: np.ndarray = self._skeleton_joint_offsets[:, 1].copy()
                # off_z: np.ndarray = self._skeleton_joint_offsets[:, 2].copy()
                # self._skeleton_joint_offsets[:, 1] = off_z
                # self._skeleton_joint_offsets[:, 2] = -off_y

                self.recompute_joint_global_info()

            def re_root(self, new_root):
                """ change the root to another joint

                    the joints will be reordered to ensure that a joint always behind its parent
                """
                raise NotImplementedError

            # Add by Zhenhua Song
            def remove_root_pos(self):
                """
                Note: this method is in place
                """
                self._joint_translation[:, :, :] = 0
                self._joint_position[:, :, :] -= self._joint_position[:, :, 0:1]

            # Add by Zhenhua Song
            def to_facing_coordinate(self):
                """
                Note: this method is in place
                """
                self._joint_translation[:, 0, :] = 0
                self._joint_rotation[:, 0, :] = Common.MathHelper.y_decompose(self._joint_rotation[:, 0, :])[1]
                # assert np.all(np.abs(self._joint_rotation[:, 0, 1] < 1e-10))
                self._joint_orientation = None
                self._joint_position = None
                self.recompute_joint_global_info()
                return self

            # Add by Zhenhua Song
            def to_local_coordinate(self):
                """
                Note: this method is in place operation
                """
                self._joint_translation[:, 0, :] = 0
                self._joint_rotation[:, 0, :] = Common.MathHelper.unit_quat_arr((self._num_frames, 4))
                self._joint_orientation = None
                self._joint_position = None
                self.recompute_joint_global_info()
                return self

            # Add by Zhenhua Song
            def get_adj_matrix(self) -> np.ndarray:
                num_joints = len(self.joint_parents_idx)
                result: np.ndarray = np.zeros(num_joints, dtype=np.int32)
                for idx, parent_idx in enumerate(self.joint_parents_idx):
                    if parent_idx == -1:
                        result[idx, parent_idx] = 1
                        result[parent_idx, idx] = 1
                return result

            # Add by Zhenhua Song
            def get_neighbours(self) -> List[List[int]]:
                num_joints: int = len(self.joint_parents_idx)
                result: List[List[int]] = [[] for _ in range(num_joints)]
                for idx, parent_idx in enumerate(self.joint_parents_idx):
                    if parent_idx == -1:
                        continue
                    result[idx].append(parent_idx)
                    result[parent_idx].append(idx)
                for idx in range(num_joints):
                    result[idx].sort()
                return result


class ODESim:

    class BodyInfo:
        @staticmethod
        def my_concatenate(tup: Iterable[np.ndarray], axis=0):
            a, b = tup
            if np.size(b) == 0 or b is None:
                return a
            if np.size(a) == 0 or a is None:
                return b
            return np.concatenate([a,b], axis=axis)

        class BodyInfo:

            __slots__ = (
                "world", "space", "bodies",
                "body_c_id", "parent", "children",
                "mass_val", "sum_mass", "root_body_id",
                "initial_inertia", "visualize_color"
            )

            def __init__(self, world: ode.World, space: ode.SpaceBase):
                self.world: ode.World = world  # world character belongs to
                self.space: ode.SpaceBase = space  # used for collision detection

                self.bodies: List[ode.Body] = []  # load from file
                self.body_c_id: Optional[np.ndarray] = None  # pointer for bodies. dtype == np.uint64. calc in init_after_load()

                self.parent: List[int] = []  # parent body's index. load from file in ODECharacterInit.py
                self.children: List[List[int]] = []  # children body's index.

                self.mass_val: Optional[np.ndarray] = None  # calc in init_after_load()
                self.sum_mass: float = 0.0  # calc in init_after_load()
                self.initial_inertia: Optional[np.ndarray] = None  # calc in init_after_load()

                # The default index of root body is 0...
                self.root_body_id: int = 0

                self.visualize_color: Optional[List] = None

            def get_subset(self, remain_body_index: List[int] = None):
                result = ODESim.BodyInfo.BodyInfo(self.world, self.space)
                result.bodies = [self.bodies[index] for index in remain_body_index]
                if self.body_c_id is not None:
                    result.body_c_id = np.ascontiguousarray(self.body_c_id[remain_body_index])
                # actually, we should not compute com by subset..
                result.root_body_id = self.root_body_id
                result.visualize_color = self.visualize_color
                return result

            @property
            def body0(self) -> Optional[ode.Body]:
                """
                Get the 0-th body of the character
                """
                return self.bodies[0] if len(self.bodies) > 0 else None

            @property
            def body1(self) -> Optional[ode.Body]:
                """
                Get the 1-th body of the character
                """
                return self.bodies[1] if len(self.bodies) > 1 else None

            @property
            def root_body(self) -> ode.Body:
                return self.bodies[self.root_body_id]

            def get_name_list(self) -> List[str]:
                """
                get names for all the bodies
                """
                return [body.name for body in self.bodies]

            def calc_body_c_id(self) -> np.ndarray:
                """
                get pointer for all of bodies. shape == (num_body,).  dtype == np.uint64
                """
                self.body_c_id = self.world.bodyListToNumpy(self.bodies)
                return self.body_c_id

            def init_after_load(self, ignore_parent_collision: bool = True,
                                ignore_grandpa_collision: bool = True):
                self.calc_geom_ignore_id(ignore_parent_collision, ignore_grandpa_collision)
                self.calc_body_c_id()
                self.mass_val: Optional[np.ndarray] = np.array([i.mass_val for i in self.bodies])
                self.sum_mass: Optional[np.ndarray] = np.sum(self.mass_val).item()
                self.initial_inertia: Optional[np.ndarray] = self.calc_body_init_inertia()
                self.visualize_color: Optional[List] = [None] * len(self.bodies)

                # here we can compute childrens..
                self.children = [[] for _ in range(len(self.bodies))]
                for i, p in enumerate(self.parent):
                    if p == -1:
                        continue
                    self.children[p].append(i)
                # print(self.children)

            def calc_body_init_inertia(self) -> np.ndarray:
                """
                Compute the initial inertia for all of bodies
                """
                inertia: np.ndarray = np.zeros((len(self), 3, 3), dtype=np.float64)
                for idx, body in enumerate(self.bodies):
                    inertia[idx, :, :] = body.init_inertia.reshape((3, 3))
                return np.ascontiguousarray(inertia)

            def calc_body_init_inertia_inv(self) -> np.ndarray:
                """
                Compute the inverse of initial inertia for all of bodies
                """
                inertia_inv: np.ndarray = np.zeros((len(self), 3, 3))
                for idx, body in enumerate(self.bodies):
                    inertia_inv[idx, :, :] = body.init_inertia_inv.reshape((3, 3))
                return np.ascontiguousarray(inertia_inv)

            def __len__(self) -> int:
                """
                length of self.bodies
                """
                return len(self.bodies)

            def get_body_contact_mu(self) -> np.ndarray:
                res = np.zeros(len(self), dtype=np.float64)
                for idx, body in enumerate(self.bodies):
                    res[idx] = list(body.geom_iter())[0].friction
                return res

            # Get Relative Position of parent body in global coordinate
            def get_relative_global_pos(self) -> np.ndarray:
                assert self.bodies

                pos_res = np.zeros((len(self), 3))
                pos_res[0, :] = self.bodies[0].PositionNumpy
                # parent of root body is -1

                for idx in range(1, len(self)):
                    body = self.bodies[idx]
                    pa_body = self.bodies[self.parent[idx]]
                    pos_res[idx, :] = body.PositionNumpy - pa_body.PositionNumpy

                return pos_res

            # Get Relative Position of parent body in parent's coordinate
            # Maybe should get position in joint's cooridinate
            # def get_relative_local_pos(self):
            #    glo_pos = self.get_relative_global_pos()

            def get_body_pos_at(self, index: int) -> np.ndarray:
                """
                Get position of index-th body. shape == (3,).  dtype == np.float64
                """
                return self.bodies[index].PositionNumpy

            def get_body_velo_at(self, index: int) -> np.ndarray:
                """
                get linear velocity of index-th body. shape == (3,).  dtype == np.float64
                """
                return self.bodies[index].LinearVelNumpy

            def get_body_quat_at(self, index: int) -> np.ndarray:
                """
                get quaternion of index-th body. shape == (3,).  dtype == np.float64
                """
                return self.bodies[index].getQuaternionScipy()

            def get_body_rot_mat_at(self, index: int) -> np.ndarray:
                """
                get rotation matrix of index-th body. shape == (9,).  dtype == np.float64
                """
                return self.bodies[index].getRotationNumpy()

            def get_body_angvel_at(self, index: int) -> np.ndarray:
                """
                get angular velocity of index-th body. shape == (3,).  dtype == np.float64
                """
                return self.bodies[index].getAngularVelNumpy()

            def get_body_pos(self) -> np.ndarray:
                """
                Get all body's position
                return np.ndarray in shape (num body, 3)
                """
                assert self.body_c_id.dtype == np.uint64
                return self.world.getBodyPos(self.body_c_id).reshape((-1, 3))

            def get_body_velo(self) -> np.ndarray:
                """
                Get all body's linear velocity
                return np.ndarray in shape (num body, 3)
                """
                assert self.body_c_id.dtype == np.uint64
                return self.world.getBodyLinVel(self.body_c_id).reshape((-1, 3))

            def get_body_ang_velo(self) -> np.ndarray:
                """
                get all body's angular velocity
                in shape (num body, 3)
                """
                assert self.body_c_id.dtype == np.uint64
                return self.world.getBodyAngVel(self.body_c_id).reshape((-1, 3))

            # get all body's quaternion
            def get_body_quat(self) -> np.ndarray:
                assert self.body_c_id.dtype == np.uint64
                return self.world.getBodyQuatScipy(self.body_c_id).reshape((-1, 4))

            # get all body's rotation
            def get_body_rot(self) -> np.ndarray:
                assert self.body_c_id.dtype == np.uint64
                return self.world.getBodyRot(self.body_c_id).reshape((-1, 3, 3))

            # get all body's force
            def get_body_force(self) -> np.ndarray:
                assert self.body_c_id.dtype == np.uint64
                return self.world.getBodyForce(self.body_c_id).reshape((-1, 3))

            # get all body's torque
            def get_body_torque(self) -> np.ndarray:
                assert self.body_c_id.dtype == np.uint64
                return self.world.getBodyTorque(self.body_c_id).reshape((-1, 3))

            def get_body_inertia(self) -> np.ndarray:
                assert self.body_c_id.dtype == np.uint64
                return self.world.getBodyInertia(self.body_c_id).reshape((-1, 3, 3))

            def set_body_pos(self, pos: np.ndarray):
                assert pos.size == self.body_c_id.size * 3
                self.world.loadBodyPos(self.body_c_id, pos.flatten().astype(np.float64))

            def set_body_velo(self, velo: np.ndarray):
                assert velo.size == self.body_c_id.size * 3
                self.world.loadBodyLinVel(self.body_c_id, velo.flatten().astype(np.float64))

            def set_body_quat(self, quat: np.ndarray):
                self.world.loadBodyQuat(self.body_c_id, quat.flatten())

            def set_body_quat_rot(self, quat: np.ndarray, rot: np.ndarray):
                assert quat.size == self.body_c_id.size * 4
                assert rot.size == self.body_c_id.size * 9
                q = quat.flatten().astype(np.float64)
                r = rot.flatten().astype(np.float64)
                self.world.loadBodyQuatAndRotNoNorm(self.body_c_id, q, r)

            def set_body_ang_velo(self, omega: np.ndarray):
                """
                set body angular velocity
                """
                assert omega.size == 3 * self.body_c_id.size
                self.world.loadBodyAngVel(self.body_c_id, omega.flatten().astype(np.float64))

            def add_body_force(self, force: np.ndarray):
                assert force.size == self.body_c_id.size * 3
                self.world.addBodyForce(self.body_c_id, force.flatten())

            def add_body_torque(self, torque: np.ndarray):
                assert torque.size == self.body_c_id.size * 3
                self.world.addBodyTorque(self.body_c_id, torque.flatten())

            # calc center of mass in world coordinate
            def calc_center_of_mass(self) -> np.ndarray:
                return self.world.compute_body_com(self.body_c_id)

            # calc CoM by BodyInfoState
            def calc_com_by_body_state(self, state) -> np.ndarray:
                pos: np.ndarray = state.pos.reshape((-1, 3))
                return np.matmul(self.mass_val[np.newaxis, :], pos).reshape(3) / self.sum_mass

            def calc_com_and_facing_com_by_body_state(self, state) -> Tuple[np.ndarray, np.ndarray]:
                """
                return: com, facing com
                """
                com: np.ndarray = self.calc_com_by_body_state(state)
                qy, qxz = Common.MathHelper.facing_decompose(state.quat.reshape((-1, 4))[self.root_body_id])
                root_pos = Common.MathHelper.vec_axis_to_zero(state.pos.reshape((-1, 3))[self.root_body_id], 1)
                facing_com: np.ndarray = Rotation(qy, copy=False).inv().apply(com - root_pos)
                return com, facing_com

            def calc_facing_com_by_body_state(self, state) -> np.ndarray:
                """
                return: np.ndarray in shape (3,)
                TODO: check with ODE Character
                """
                _, facing_com = self.calc_com_and_facing_com_by_body_state(state)
                return facing_com

            def calc_velo_com(self) -> np.ndarray:
                """
                Calc Velocity of Center of Mass in World Coordinate
                """
                return np.matmul(self.mass_val[np.newaxis, :], self.get_body_velo()).reshape(-1) / self.sum_mass

            # Divide Rotation of Root into Ry * Rxz. Return Ry.
            def calc_facing_quat(self) -> np.ndarray:
                """
                return: in shape (4,)
                """
                res, _ = Common.MathHelper.facing_decompose(self.bodies[self.root_body_id].getQuaternionScipy())
                return res

            # Calc CoM Momentum
            def calc_sum_mass_pos(self) -> np.ndarray:
                return np.matmul(self.mass_val[np.newaxis, :], self.get_body_pos()).reshape(-1)

            # Calc momentum velocity in world coordinate
            def calc_momentum(self) -> np.ndarray:
                return np.matmul(self.mass_val[np.newaxis, :], self.get_body_velo()).reshape(-1)

            # Calc angular momentum in world coordinate
            def calc_angular_momentum_slow(self) -> Tuple[np.ndarray, ode.Inertia]:
                """
                angular momentum
                """
                com = self.calc_center_of_mass()
                ang_momentum = np.zeros(3)
                tot_inertia: ode.Inertia = ode.Inertia()
                for idx, body in enumerate(self.bodies):
                    m: ode.Mass = body.getMass()
                    inertia: np.ndarray = m.getINumpy().reshape((3, 3))  # inertia in body local coordinate
                    rot: np.ndarray = body.getRotationNumpy().reshape((3, 3))  # body's global rotation
                    inertia: np.ndarray = rot @ inertia @ rot.T  # inertia in world coordinate
                    cci: np.ndarray = body.PositionNumpy - com  # from CoM to body's position in world coordinate

                    ang_momentum += inertia @ body.getAngularVelNumpy() + m.mass * np.cross(cci, body.LinearVelNumpy)

                    inertia_res = ode.Inertia()
                    inertia_res.setMassAndInertia(m.mass, inertia)
                    inertia_res.TransInertiaNumpy(cci)
                    tot_inertia.add(inertia_res)

                return ang_momentum, tot_inertia

            def calc_angular_momentum(self, com: Optional[np.ndarray] = None) -> np.ndarray:
                inertia: np.ndarray = self.initial_inertia
                dcm: np.ndarray = self.get_body_rot()
                inertia: np.ndarray = dcm @ inertia @ dcm.transpose((0, 2, 1))
                ang_vel: np.ndarray = self.get_body_ang_velo()
                angular_momentum: np.ndarray = (inertia @ (ang_vel[..., None])).reshape(ang_vel.shape)

                if com is None:
                    com: np.ndarray = self.calc_center_of_mass()
                com_to_body: np.ndarray = self.get_body_pos() - com[None, :]

                linear_momentum: np.ndarray = self.mass_val[:, None] * self.get_body_velo()
                angular_momentum: np.ndarray = angular_momentum + np.cross(com_to_body, linear_momentum, axis=-1)
                angular_momentum: np.ndarray = np.sum(angular_momentum, axis=0)  # (frame, 3)

                return angular_momentum

            def get_geom_rot(self):
                return [[geom.QuaternionScipy for geom in body.geom_iter()] for body in self.bodies]

            def get_geom_pos(self):
                return [[geom.PositionNumpy for geom in body.geom_iter()] for body in self.bodies]

            def calc_geom_ignore_id(self, ignore_parent_collision: bool = True,
                                    ignore_grandpa_collision: bool = True):
                """
                Calc ignore id of each geoms in character. ignore collision detection between body and its parent & grandparent
                :return:
                """
                if not ignore_parent_collision and not ignore_grandpa_collision:
                    return

                for idx, body in enumerate(self.bodies):
                    ignore_geom_id = []
                    if self.parent[idx] != -1:
                        pa_body_idx = self.parent[idx]
                        for pa_geom in self.bodies[pa_body_idx].geom_iter():
                            if ignore_parent_collision:
                                ignore_geom_id.append(pa_geom.get_gid())

                            if ignore_grandpa_collision:
                                grandpa_idx = self.parent[pa_body_idx]
                                if grandpa_idx != -1:
                                    for grandpa_geom in self.bodies[grandpa_idx].geom_iter():
                                        ignore_geom_id.append(grandpa_geom.get_gid())

                    if len(ignore_geom_id) > 0:
                        for geom in body.geom_iter():
                            geom.extend_ignore_geom_id(ignore_geom_id)

            # Get AABB bounding box of bodies and geoms.
            def get_aabb(self) -> np.ndarray:
                """
                Get AABB bounding box of bodies and geoms.
                """
                return self.space.get_bodies_aabb(self.body_c_id)

            def clear(self):
                for body in self.bodies:
                    for geom in body.geom_iter():
                        geom.destroy_immediate()  # destroy all of geometries
                    body.destroy_immediate()

                self.bodies.clear()
                self.body_c_id: Optional[np.ndarray] = None

                self.parent.clear()

                self.mass_val: Optional[np.ndarray] = None
                self.sum_mass = 0.0

                self.root_body_id: int = 0
                self.visualize_color: Optional[List] = None
                return self

            def get_mirror_index(self) -> List[int]:
                body_names = self.get_name_list()
                return Common.Helper.mirror_name_list(body_names)

    class BodyInfoState:
        class BodyInfoState:
            """
            save the state for rigid bodies
            """
            __slots__ = ("pos", "rot", "quat", "linear_vel", "angular_vel", "pd_target", "force", "torque") # , "contact_info")

            def __init__(self):
                self.pos: Optional[np.ndarray] = None
                self.rot: Optional[np.ndarray] = None
                self.quat: Optional[np.ndarray] = None
                self.linear_vel: Optional[np.ndarray] = None
                self.angular_vel: Optional[np.ndarray] = None

                self.pd_target: Optional[np.ndarray] = None  # Target pose for pd controller
                self.force: Optional[np.ndarray] = None
                self.torque: Optional[np.ndarray] = None

                # self.contact_info = None

            # reshape the rigid body
            def reshape(self):
                if self.pos is not None:
                    self.pos = self.pos.reshape((-1, 3))
                if self.rot is not None:
                    self.rot = self.rot.reshape((-1, 3, 3))
                if self.quat is not None:
                    self.quat = self.quat.reshape((-1, 4))
                if self.linear_vel is not None:
                    self.linear_vel = self.linear_vel.reshape((-1, 3))
                if self.angular_vel is not None:
                    self.angular_vel = self.angular_vel.reshape((-1, 3))
                if self.pd_target is not None:
                    self.pd_target = self.pd_target.reshape((-1, 4))
                if self.force is not None:
                    self.force = self.force.reshape((-1, 3))
                if self.torque is not None:
                    self.torque = self.torque.reshape((-1, 3))
                return self

            def set_value(
                self,
                pos: np.ndarray,
                rot: np.ndarray,
                quat: np.ndarray,
                linvel: np.ndarray,
                angvel: np.ndarray,
                pd_target: Optional[np.ndarray],
                ):
                self.pos = pos.reshape(-1).astype(np.float64)
                self.rot = rot.reshape(-1).astype(np.float64)
                self.quat = quat.reshape(-1).astype(np.float64)
                self.linear_vel = linvel.reshape(-1).astype(np.float64)
                self.angular_vel = angvel.reshape(-1).astype(np.float64)
                if pd_target is not None:
                    self.pd_target = pd_target.astype(np.float64)

                self.to_continuous()
                return self

            def __del__(self):
                del self.pos
                del self.rot
                del self.quat
                del self.linear_vel
                del self.angular_vel
                if self.pd_target is not None:
                    del self.pd_target
                if self.force is not None:
                    del self.force

            def check_failed(self) -> bool:
                if np.any(np.abs(self.pos) > 10000):
                    return True
                if np.any(np.abs(self.rot) > 10000):
                    return True
                if np.any(np.abs(self.quat) > 10000):
                    return True
                if np.any(np.abs(self.linear_vel) > 10000):
                    return True
                if np.any(np.abs(self.angular_vel) > 10000):
                    return True
                return False

            def clear(self):
                self.pos: Optional[np.ndarray] = None
                self.rot: Optional[np.ndarray] = None
                self.quat: Optional[np.ndarray] = None
                self.linear_vel: Optional[np.ndarray] = None
                self.angular_vel: Optional[np.ndarray] = None

                self.pd_target: Optional[np.ndarray] = None
                self.force: Optional[np.ndarray] = None
                self.torque: Optional[np.ndarray] = None

                # self.contact_info = None

            def is_empty(self):
                return self.pos is None and self.rot is None \
                    and self.quat is None and self.linear_vel is None and self.angular_vel is None

            def __len__(self):
                return self.pos.shape[0] // 3 if self.pos is not None else 0

            def calc_delta(self, o):
                d_pos: np.ndarray = np.max(np.abs(self.pos - o.pos))
                d_rot: np.ndarray = np.max(np.abs(self.rot - o.rot))
                d_quat_ode: np.ndarray = np.max(np.abs(self.quat - o.quat))
                d_lin_vel: np.ndarray = np.max(np.abs(self.linear_vel - o.linear_vel))
                d_ang_vel: np.ndarray = np.max(np.abs(self.angular_vel - o.angular_vel))

                return d_pos, d_rot, d_quat_ode, d_lin_vel, d_ang_vel

            def check_delta(self, o):
                d_pos, d_rot, d_quat_ode, d_lin_vel, d_ang_vel = self.calc_delta(o)

                try:
                    assert d_pos == 0
                    assert d_rot == 0
                    assert d_quat_ode == 0
                    assert d_lin_vel == 0
                    assert d_ang_vel == 0
                except AssertionError:
                    bug_info = (
                        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        f"dpos = {d_pos}, "
                        f"drot = {d_rot}, "
                        f"d_quat_ode = {d_quat_ode}, "
                        f"d_lin_vel = {d_lin_vel}, "
                        f"d_ang_vel = {d_ang_vel}"
                    )
                    import logging
                    logging.info(bug_info)
                    print(bug_info)

            def save(self, world: ode.World, body_c_id: np.ndarray):
                _, self.pos, self.quat, self.rot, self.linear_vel, self.angular_vel = world.getBodyInfos(body_c_id)

            def load(self, world: ode.World, body_c_id: np.ndarray):
                world.loadBodyInfos(body_c_id, self.pos, self.quat, self.rot, self.linear_vel, self.angular_vel,
                                    self.force, self.torque)

            def copy(self):
                res = ODESim.BodyInfoState.BodyInfoState()
                if self.pos is not None:
                    res.pos = self.pos.copy()
                if self.rot is not None:
                    res.rot = self.rot.copy()
                if self.quat is not None:
                    res.quat = self.quat.copy()
                if self.linear_vel is not None:
                    res.linear_vel = self.linear_vel.copy()
                if self.angular_vel is not None:
                    res.angular_vel = self.angular_vel.copy()
                if self.pd_target is not None:
                    res.pd_target = self.pd_target.copy()
                if self.force is not None:
                    res.force = self.force.copy()
                if self.torque is not None:
                    res.torque = self.torque.copy()
                return res

            def to_continuous(self):
                self.pos = np.ascontiguousarray(self.pos)
                self.rot = np.ascontiguousarray(self.rot)
                self.quat = np.ascontiguousarray(self.quat)
                self.linear_vel = np.ascontiguousarray(self.linear_vel)
                self.angular_vel = np.ascontiguousarray(self.angular_vel)

                if self.pd_target is not None:
                    self.pd_target = np.ascontiguousarray(self.pd_target)

                if self.force is not None:
                    self.force = np.ascontiguousarray(self.force)

                if self.torque is not None:
                    self.torque = np.ascontiguousarray(self.torque)

                # if self.contact_info is not None:
                #    pass

                return self

            def cat_to_ndarray(self) -> np.ndarray:
                return np.concatenate([self.pos.reshape(-1), self.rot.reshape(-1), self.quat.reshape(-1), self.linear_vel.reshape(-1), self.angular_vel.reshape(-1)])

    class EndJointInfo:
        class EndJointInfo:
            def __init__(self, world: ode.World):
                self.world: ode.World = world
                self.name: List[str] = []
                self.init_global_pos: Optional[np.ndarray] = None
                self.pa_body_id: Optional[np.ndarray] = None  # parent body id. (head, l-r hand, l-r foot)
                self.pa_body_c_id: Optional[np.ndarray] = None

                # Position of joint relative to its parent body in world(global) coordinate
                self.jtob_init_global_pos: Optional[np.ndarray] = None

                # Position of joint relative to its parent body in body coordinate
                self.jtob_init_local_pos: Optional[np.ndarray] = None

                # Parent Joint ID
                self.pa_joint_id: Optional[np.ndarray] = None
                # Parent Joint C ID
                self.pa_joint_c_id: Optional[np.ndarray] = None

                # Positions relative to parent joint in world (global) coordinate
                self.jtoj_init_global_pos: Optional[np.ndarray] = None

                # Position relative to parent joint in pajoint frame
                self.jtoj_init_local_pos: Optional[np.ndarray] = None

                self.weights: Optional[np.ndarray] = None  # weight for calc loss

            def __len__(self) -> int:
                return len(self.name)

            def resize(self):
                self.init_global_pos = np.zeros((len(self), 3))
                self.jtob_init_global_pos = np.zeros((len(self), 3))
                self.jtob_init_local_pos = np.zeros((len(self), 3))
                self.weights = np.ones(len(self))

                return self

            def get_global_pos(self) -> np.ndarray:
                """
                Get End Joint's Global Position
                """
                body_quat: np.ndarray = self.world.getBodyQuatScipy(self.pa_body_c_id).reshape((-1, 4))
                body_pos: np.ndarray = self.world.getBodyPos(self.pa_body_c_id).reshape((-1, 3))
                return quat_apply_forward_fast(body_quat, self.jtob_init_local_pos) + body_pos

            def clear(self):
                self.name.clear()
                self.init_global_pos: Optional[np.ndarray] = None
                self.pa_body_id: Optional[np.ndarray] = None
                self.pa_body_c_id: Optional[np.ndarray] = None
                self.jtob_init_global_pos: Optional[np.ndarray] = None
                self.jtob_init_local_pos: Optional[np.ndarray] = None
                self.pa_joint_id: Optional[np.ndarray] = None
                self.pa_joint_c_id: Optional[np.ndarray] = None
                self.jtoj_init_global_pos: Optional[np.ndarray] = None
                self.jtoj_init_local_pos: Optional[np.ndarray] = None
                self.weights: Optional[np.ndarray] = None

                return self

    class Environment:
        class Environment:
            """
            static geometry in environment
            """
            def __init__(self, world: ode.World, space: ode.SpaceBase):
                self.world: ode.World = world
                self.space: ode.SpaceBase = space
                self.floor: Union[ode.GeomPlane, ode.GeomBox, None] = None
                self.geoms: List[ode.GeomObject] = []
                # Should save position and rotation of non-placeable Geoms (Geom without Body)

            def __len__(self) -> int:
                if self.geoms is not None:
                    return len(self.geoms)
                else:
                    return 0

            def set_space(self, space: Optional[ode.SpaceBase]) -> ode.SpaceBase:
                if self.geoms is not None:
                    for geom in self.geoms:
                        geom.space = space
                self.space = space
                return self.space

            def enable(self):
                if self.geoms is None:
                    return
                for geom in self.geoms:
                    geom.enable()
                return self

            def disable(self):
                if self.geoms is None:
                    return
                for geom in self.geoms:
                    geom.disable()
                return self

            def get_floor_in_list(self) -> Optional[ode.GeomObject]:
                """
                floor will be GeomBox or GeomPlane type..
                """
                if self.geoms is None:
                    return None
                self.floor: Union[ode.GeomPlane, ode.GeomBox] = None
                for geom in self.geoms:
                    if isinstance(geom, ode.GeomPlane):
                        self.floor = geom
                        break
                    elif isinstance(geom, ode.GeomBox):
                        if np.mean(geom.LengthNumpy) > 50:
                            self.floor = geom
                            break
                return self.floor

            def create_floor(self, friction=0.8) -> ode.GeomPlane:
                self.floor = ode.GeomPlane(self.space, (0, 1, 0), 0)
                self.floor.name = "Floor"
                self.floor.character_id = -1
                self.floor.friction = friction
                self.geoms.append(self.floor)
                # Maybe there is only one GeomPlane in the Environment...
                # Modify ode source code, to get plane's pos...
                return self.floor

            def clear(self):
                self.floor: Union[ode.GeomPlane, ode.GeomBox] = None
                self.geoms.clear()
                return self

    class ExtJointList:
        class ExtJointInfo:
            def __init__(self, character0_id: int = 0, body0_id: int = 0,
                        character1_id: int = 0, body1_id: int = 0):
                self.character0_id: int = character0_id
                self.body0_id: int = body0_id

                self.character1_id: int = character1_id
                self.body1_id: int = body1_id


    class JointInfo:
        @staticmethod
        def my_concatenate(tup, axis=0) -> np.ndarray:
            a, b = tup
            if np.size(b) == 0 or b is None:
                return a
            if np.size(a) == 0 or a is None:
                return b
            return np.concatenate([a, b], axis=axis)


        class JointInfosBase:
            def __init__(self, world: ode.World):
                self.world: ode.World = world  #
                self.joints: List[Union[ode.Joint, ode.BallJointAmotor, ode.BallJoint, ode.HingeJoint]] = []
                self.joint_c_id: Optional[np.ndarray] = None  # address(id) of joints in C code. calc in JointInfoInit.py

                self.hinge_c_id: Optional[np.ndarray] = None  # address(id) of joints in C code. calc in JointInfoInit.py
                self.hinge_lo: Optional[np.ndarray] = None  # lo of hinge angle limit
                self.hinge_hi: Optional[np.ndarray] = None  # hi of hinge angle limit

                self.weights: Optional[np.ndarray] = None  # weight of each joint for computing loss. load from file.
                self.euler_axis_local: Optional[np.ndarray] = None  # local joint euler axis. load from file.

                self.torque_limit: Optional[np.ndarray] = None  # max torque add on the body. load from file.
                self.kps: Optional[np.ndarray] = None  # Kp parameter of each joint. load from file.
                self.kds: Optional[np.ndarray] = None  # Kd parameter of each joint. load from file.

            def __add__(self, other):

                self.joints += other.joints
                self.joint_c_id = ODESim.JointInfo.my_concatenate([self.joint_c_id, other.joint_c_id], axis=0)

                self.hinge_c_id = ODESim.JointInfo.my_concatenate([self.hinge_c_id, other.hinge_c_id], axis=0)
                self.hinge_lo = ODESim.JointInfo.my_concatenate([self.hinge_lo, other.hinge_lo], axis=0)
                self.hinge_hi = ODESim.JointInfo.my_concatenate([self.hinge_hi, other.hinge_hi], axis=0)
                self.weights = ODESim.JointInfo.my_concatenate([self.weights, other.weights], axis=0)
                self.euler_axis_local = ODESim.JointInfo.my_concatenate([self.euler_axis_local, other.euler_axis_local], axis=0)

                self.torque_limit = ODESim.JointInfo.my_concatenate([self.torque_limit, other.torque_limit], axis=0)
                self.kps = ODESim.JointInfo.my_concatenate([self.kps, other.kps], axis=0)
                self.kds = ODESim.JointInfo.my_concatenate([self.kds, other.kds], axis=0)

                return self

            def __len__(self) -> int:
                return len(self.joints)

            def joint_names(self) -> List[str]:
                """
                return: each joints' name
                """
                return [i.name for i in self.joints]

            def ball_id(self) -> List[int]:
                """
                all ball joints' index
                """
                return [idx for idx, joint in enumerate(self.joints) if issubclass(type(joint), ode.BallJointBase)]

            def ball_joints(self) -> List[Union[ode.BallJointAmotor, ode.BallJoint]]:
                return [joint for joint in self.joints if issubclass(type(joint), ode.BallJointBase)]

            def hinge_id(self) -> List[int]:
                """
                All Hinge Joints' index
                """
                return [idx for idx, joint in enumerate(self.joints) if type(joint) == ode.HingeJoint]

            def hinge_joints(self) -> List[ode.HingeJoint]:
                """
                All Hinge Joints
                """
                return [joint for joint in self.joints if type(joint) == ode.HingeJoint]

            def hinge_lo_hi(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                return self.hinge_lo, self.hinge_hi

            def has_hinge(self) -> bool:
                return self.hinge_c_id is not None and self.hinge_c_id.size > 0

            def clear(self):
                for joint in self.joints:
                    joint.destroy_immediate()

                self.joints.clear()
                self.joint_c_id: Optional[np.ndarray] = None
                self.hinge_c_id: Optional[np.ndarray] = None
                self.hinge_lo: Optional[np.ndarray] = None
                self.hinge_hi: Optional[np.ndarray] = None

                self.weights: Optional[np.ndarray] = None
                self.euler_axis_local = None

                self.kps: Optional[np.ndarray] = None
                self.kds: Optional[np.ndarray] = None
                self.torque_limit: Optional[np.ndarray] = None

                return self

        class JointInfos(JointInfosBase):
            def __init__(self, world: ode.World):
                super().__init__(world)
                self.pa_joint_id: List[int] = []  # id of joint's parent joint
                self.sample_win: Optional[np.ndarray] = None  # Sample Window for Samcon. load from file.
                self.sample_mask: Optional[np.ndarray] = None

                self.parent_body_index: Optional[np.ndarray] = None  # TODO: concat it from __add__
                self.child_body_index: Optional[np.ndarray] = None
                self.parent_body_c_id: Optional[np.ndarray] = None  # address(id) of parent body of each joint in C code
                self.child_body_c_id: Optional[np.ndarray] = None  # address(id) of child_body of each joint in C code

                # load from file
                self.has_root: bool = False  # whether character has root joint.
                self.root_idx: Optional[int] = None  # index of root joint. For convenience, it should be None or 0.

            def get_subset_by_name(self, remain_joint_names: List[str]):
                curr_name: List[str] = self.joint_names()
                curr_name_dict = {node: index for index, node in enumerate(curr_name)}
                remain_list: List[int] = [curr_name_dict[name] for name in remain_joint_names]
                remain_list.sort()
                self.get_subset(remain_list)

            def get_subset(self, remain_joint_index: List[int]):
                result = ODESim.JointInfo.JointInfos(self.world)
                result.world = self.world
                result.joints = [self.joints[index] for index in remain_joint_index]
                if self.joint_c_id is not None:
                    result.joint_c_id = np.ascontiguousarray(self.joint_c_id[remain_joint_index])
                remain_hinge_index = [index for index in remain_joint_index if isinstance(self.joints[index], ode.HingeJoint)]
                if len(remain_hinge_index) > 0:
                    remain_hinge_joints = [self.joints[index] for index in remain_hinge_index]
                    if self.hinge_c_id is not None:
                        result.hinge_c_id = self.world.jointListToNumpy(remain_hinge_joints)
                    if self.hinge_lo is not None:
                        result.hinge_lo = self.get_hinge_lo(remain_hinge_joints)
                    if self.hinge_hi is not None:
                        result.hinge_hi = self.get_hinge_hi(remain_hinge_joints)

                if self.weights is not None:
                    result.weights = np.ascontiguousarray(self.weights[remain_joint_index])
                if self.euler_axis_local is not None:
                    result.euler_axis_local = np.ascontiguousarray(self.euler_axis_local[remain_joint_index])
                if self.torque_limit is not None:
                    result.torque_limit = np.ascontiguousarray(self.torque_limit[remain_joint_index])
                if self.kps is not None:
                    result.kps = np.ascontiguousarray(self.kps[remain_joint_index])
                if self.kds is not None:
                    result.kds = np.ascontiguousarray(self.kds[remain_joint_index])

                # assume the remaining joints are continuous.
                result.pa_joint_id = []  # TODO: id of joint's parent joint
                if self.sample_win is not None:
                    result.sample_win = np.ascontiguousarray(self.sample_win[remain_joint_index])
                if self.sample_mask is not None:
                    result.sample_mask = np.ascontiguousarray(self.sample_mask[remain_joint_index])

                if self.parent_body_index is not None:  # TODO
                    pass
                if self.child_body_index is not None:
                    pass

                if self.parent_body_c_id is not None:
                    result.parent_body_c_id = np.ascontiguousarray(self.parent_body_c_id[remain_joint_index])
                if self.child_body_c_id is not None:
                    result.child_body_c_id = np.ascontiguousarray(self.child_body_c_id[remain_joint_index])

                result.has_root = self.has_root  # assume root is not modified..
                result.root_idx = self.root_idx
                return result

            @property
            def root_joint(self) -> Optional[ode.Joint]:
                return self.joints[self.root_idx] if self.has_root else None

            def resize_euler_axis_local(self):
                self.euler_axis_local = np.tile(np.eye(3), len(self)).reshape((-1, 3, 3))

            @staticmethod
            def body_rotvec(body: ode.Body) -> np.ndarray:
                """
                Get Body's Rot Vector in world coordinate
                """
                return Rotation.from_matrix(np.array(body.getRotation()).reshape((3, 3))).as_rotvec()

            def disable_root(self) -> None:
                """
                Disable root joint if exists
                """
                if not self.has_root:
                    return
                self.joints[self.root_idx].disable()  # joint->flags |= dJOINT_DISABLED;

            def enable_root(self) -> None:
                """
                enable root joint if exists
                """
                if not self.has_root:
                    return
                self.joints[self.root_idx].enable()

            def parent_qs(self) -> np.ndarray:
                """
                Get parent bodies' quaternion in global coordinate
                """
                res: np.ndarray = self.world.getBodyQuatScipy(self.parent_body_c_id).reshape((-1, 4))  # (num joint, 4)
                # if simulation fails, quaternion result will be very strange.
                # for example, rotations will be zero...
                if self.has_root:
                    res[self.root_idx] = Common.MathHelper.unit_quat()
                res /= np.linalg.norm(res, axis=-1, keepdims=True)
                return res

            def child_qs(self) -> np.ndarray:
                """
                Get Child bodies' quaternion in global coordinate
                """
                res: np.ndarray = self.world.getBodyQuatScipy(self.child_body_c_id).reshape((-1, 4))  # (num joint, 4)
                res /= np.linalg.norm(res, axis=-1, keepdims=True)
                return res

            # return q1s, q2s, local_qs, q1s_inv
            def get_parent_child_qs_old(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                """
                Note: implement of scipy is slow..use cython version instead.
                return:
                parent bodies' quaternion in global coordinate,
                child bodies' quaternion in global coordinate,
                joint's quaternion in parent's local coordinate.
                inverse of parent bodies' quaternion in global coordinate
                """
                raise ValueError("This is slow. Please use cython version instead.")
                parent_qs: np.ndarray = self.parent_qs()  # parent bodies' quaternion in global coordinate
                child_qs: np.ndarray = self.child_qs()  # child bodies' quaternion in global coordinate

                parent_qs_inv: Rotation = Rotation(parent_qs, False, False).inv()  # inv parent bodies' quaternion in global
                local_qs: np.ndarray = (parent_qs_inv * Rotation(child_qs, False, False)).as_quat()  # joints' local quaternion

                # Test with C++ version
                # print(self.joint_c_id)
                c_parent_qs, c_child_qs, c_local_qs, c_parent_qs_inv = self.world.get_all_joint_local_angle(self.joint_c_id)
                print(np.max(np.abs(parent_qs - c_parent_qs)))
                print(np.max(np.abs(child_qs - c_child_qs)))
                print(np.max(np.abs(MathHelper.flip_quat_by_w(local_qs) - MathHelper.flip_quat_by_w(c_local_qs))))
                print(np.max(np.abs(MathHelper.flip_quat_by_w(parent_qs_inv.as_quat()) - MathHelper.flip_quat_by_w(c_parent_qs_inv))))
                return parent_qs, child_qs, local_qs, parent_qs_inv.as_quat()

            def get_parent_child_qs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                # check when character has real root joint OK
                c_parent_qs, c_child_qs, c_local_qs, c_parent_qs_inv = self.world.get_all_joint_local_angle(self.joint_c_id)
                return c_parent_qs, c_child_qs, c_local_qs, c_parent_qs_inv

            def get_local_q(self) -> np.ndarray:
                """
                joint' quaternion in parent's local coordinate
                """
                _, _, local_qs, _ = self.get_parent_child_qs()
                return local_qs

            def get_local_angvels(self, parent_qs_inv: Optional[np.ndarray] = None) -> np.ndarray:
                """
                param:
                parent_qs_inv: Optional. inverse of parent bodies' quaternion in global coordinate

                return: Joints' angular velocity in parent body's local coordinate, in shape (num joint, 3)
                """
                # if parent_qs_inv is None:
                #     parent_qs_inv = Rotation(self.parent_qs(), normalize=True, copy=False).inv().as_quat()
                global_angvel = self.get_global_angvels()
                return quat_apply_forward_fast(parent_qs_inv, global_angvel)
                # return Rotation(parent_qs_inv, False, False).apply(global_angvel)

            def get_global_angvels(self) -> np.ndarray:
                """
                return: Joints' angular velocity in global coordinate, in shape (num joint, 3)
                """
                ang_parent_body: np.ndarray = self.world.getBodyAngVel(self.parent_body_c_id).reshape((-1, 3))
                ang_child_body: np.ndarray = self.world.getBodyAngVel(self.child_body_c_id).reshape((-1, 3))
                if self.has_root:
                    ang_child_body[self.root_idx] = 2 * ang_parent_body[self.root_idx]
                return ang_child_body - ang_parent_body

            # Get Position relative to parent joint in world(global) coordinate
            def get_relative_global_pos(self) -> np.ndarray:
                # Emm..should load root joint when export to bvh...
                # assert self.has_root

                global_pos: np.ndarray = self.world.getBallAndHingeAnchor1(self.joint_c_id).reshape((-1, 3))
                pa_global_pos: np.ndarray = global_pos[self.pa_joint_id]
                if self.has_root:
                    pa_global_pos[self.root_idx, :] = self.joints[self.root_idx].getAnchorNumpy()
                else:
                    for idx, pa_idx in enumerate(self.pa_joint_id):
                        if pa_idx >= 0:
                            continue
                        pa_body: ode.Body = self.joints[idx].body2
                        pa_global_pos[idx, :] = pa_body.PositionNumpy

                return global_pos - pa_global_pos

            # Get Position relative to parent joint in parent joint's coordinate..
            def get_relative_local_pos(self) -> Tuple[np.ndarray, np.ndarray]:
                # check OK when has real root joint
                parent_qs, child_qs, local_qs, parent_qs_inv = self.get_parent_child_qs()
                global_offset = self.get_relative_global_pos()
                # joint's global rotation should be q2. joint's parent joint's global rotation should be q1..
                offset: np.ndarray = Rotation(parent_qs_inv, copy=False, normalize=False).apply(global_offset)
                return local_qs, offset

            # r' = vector from child body to joint in child body's local frame
            # Rb = rotation matrix of child body
            # xb = global position of child body
            # xj = joint position in world, that is, xj = xb + Rb * r'
            # for joint in in ode v0.12, body0 is child body, and body1 is parent body
            # assume there is only ball joint, hinge joint, and amotor joint in a character
            # this method returns r'
            def get_child_body_relative_pos(self) -> np.ndarray:
                # Note: this method only support dJointID as input.
                # if you takes other as input, the program will crash or fall in dead cycle
                return self.world.getBallAndHingeRawAnchor1(self.joint_c_id).reshape((-1, 3))

            def get_parent_body_relative_pos(self) -> np.ndarray:
                return self.world.getBallAndHingeRawAnchor2(self.joint_c_id).reshape((-1, 3))

            # actually, we need to get hinge joint's index
            def get_hinge_raw_axis1(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
                if hinges is None:
                    hinges = self.hinge_joints()

                res = np.zeros((len(hinges), 3))
                for idx, joint in enumerate(hinges):
                    res[idx, :] = joint.Axis1RawNumpy
                return np.ascontiguousarray(res)

            def get_hinge_raw_axis2(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
                if hinges is None:
                    hinges = self.hinge_joints()

                res = np.zeros((len(hinges), 3))
                for idx, joint in enumerate(hinges):
                    res[idx, :] = joint.Axis2RawNumpy
                return np.ascontiguousarray(res)

            def get_hinge_axis1(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
                if hinges is None:
                    hinges = self.hinge_joints()

                res = np.zeros((len(hinges), 3))
                for idx, joint in enumerate(hinges):
                    res[idx, :] = joint.HingeAxis1
                return np.ascontiguousarray(res)

            def get_hinge_axis2(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
                if hinges is None:
                    hinges = self.hinge_joints()

                res = np.zeros((len(hinges), 3))
                for idx, joint in enumerate(hinges):
                    res[idx, :] = joint.HingeAxis2
                return np.ascontiguousarray(res)

            def get_hinge_angle(self) -> np.ndarray:
                """
                return angle of each hinge joint
                """
                return self.world.get_all_hinge_angle(self.hinge_c_id)

            def get_global_anchor1(self) -> np.ndarray:
                """
                call dJointGetBallAnchor1 and dJointGetHingeAnchor1
                """
                global_pos: np.ndarray = self.world.getBallAndHingeAnchor1(self.joint_c_id).reshape((-1, 3))
                return global_pos

            def get_global_pos1(self) -> np.ndarray:
                return self.get_global_anchor1()

            def get_global_anchor2(self) -> np.ndarray:
                """
                call dJointGetBallAnchor2 and dJointGetHingeAnchor2
                if simulation is totally correct, result of GetAnchor2 should be equal to GetAnchor1
                """
                return self.world.getBallAndHingeAnchor2(self.joint_c_id).reshape((-1, 3))

            def get_global_pos2(self) -> np.ndarray:
                """
                get global joint anchor
                """
                return self.get_global_anchor2()

            def get_joint_euler_order(self) -> List[str]:
                return [joint.euler_order for joint in self.joints]

            def get_ball_erp(self, balls: Optional[List[Union[ode.BallJoint, ode.BallJointAmotor]]] = None) -> np.ndarray:
                """
                Get erp parameter of all ball joints
                """
                if balls is None:
                    balls = self.ball_joints()
                return np.asarray([joint.joint_erp for joint in balls])

            def get_hinge_erp(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
                """
                get erp parameter of all hinge joints
                """
                if hinges is None:
                    hinges = self.hinge_joints()
                return np.asarray([joint.joint_erp for joint in hinges])

            def get_hinge_lo(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
                if hinges is None:
                    hinges = self.hinge_joints()

                return np.array([joint.AngleLoStop for joint in hinges])

            def get_hinge_hi(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
                if hinges is None:
                    hinges = self.hinge_joints()

                return np.array([joint.AngleHiStop for joint in hinges])

            def get_erp(self) -> np.ndarray:
                """
                Get erp of all joints
                """
                return np.asarray([joint.joint_erp for joint in self.joints])

            def get_cfm(self) -> np.ndarray:
                """
                Get CFM parameter of all Joints
                """
                return np.asarray([joint.joint_cfm for joint in self.joints])

            def clear(self):
                super().clear()
                self.sample_win: Optional[np.ndarray] = None
                self.has_root: bool = False
                self.root_idx: Optional[int] = None
                self.parent_body_index: Optional[np.ndarray] = None
                self.child_body_index: Optional[np.ndarray] = None
                self.parent_body_c_id: Optional[np.ndarray] = None
                self.child_body_c_id: Optional[np.ndarray] = None
                self.pa_joint_id.clear()

                return self

            def gen_sample_mask(self, use_joint_names=None) -> np.ndarray:
                result_list = []
                for joint in self.joints:
                    if use_joint_names is not None and joint.name not in use_joint_names:
                        continue
                    buff = np.ones(3, dtype=np.float64)
                    if isinstance(joint, ode.HingeJoint):
                        buff[:] = 0
                        buff[ord(joint.euler_order) - ord('X')] = 1
                    result_list.append(buff)
                self.sample_mask = np.array(result_list)
                return self.sample_mask

            def get_mirror_index(self) -> List[int]:
                """
                Modified from Libin Liu's pymotionlib
                TODO: Test
                """
                joint_names = self.joint_names()
                return Common.Helper.mirror_name_list(joint_names)

            def set_joint_weights(self, weight_dict: Dict[str, float]):
                self.weights = np.ones(len(self.joints))
                try:
                    joint_name_dict: Dict[str, int] = {name: index for index, name in enumerate(self.joint_names())}
                    for key, value in weight_dict.items():
                        index: int = joint_name_dict[key]
                        self.weights[index] = value
                    self.weights = np.ascontiguousarray(self.weights)
                except:
                    pass
                return self.weights

            def get_adj_matrix(self) -> np.ndarray:
                """
                get adj matrix of each joints.
                """
                num_joints: int = len(self.joints)
                ret: np.ndarray = np.zeros((num_joints, num_joints), dtype=np.int32)
                for idx, parent in enumerate(self.pa_joint_id):
                    if parent == -1:
                        continue
                    ret[idx, parent] = 1
                    ret[parent, idx] = 1
                return ret

            def get_neighbours(self) -> List[List[int]]:
                num_joints: int = len(self.joint_parents_idx)
                result: List[List[int]] = [[] for _ in range(num_joints)]
                for idx, parent_idx in enumerate(self.joint_parents_idx):
                    if parent_idx == -1:
                        continue
                    result[idx].append(parent_idx)
                    result[parent_idx].append(idx)
                for idx in range(num_joints):
                    result[idx].sort()
                return result

    class JointInfoWrapper:
        class JointInfoWrapper:
            """
            Wrapper of JointInfos
            """
            def __init__(self, joint_info=None):
                self.joint_info = joint_info

            @property
            def world(self) -> ode.World:
                return self.joint_info.world

            @property
            def joints(self) -> List[ode.Joint]:
                return self.joint_info.joints

            @joints.setter
            def joints(self, value: Optional[ode.Joint]):
                self.joint_info.joints = value

            def __len__(self) -> int:
                return len(self.joint_info)

            def joint_names(self) -> List[str]:
                return self.joint_info.joint_names()

            @property
            def pa_joint_id(self) -> List[int]:
                return self.joint_info.pa_joint_id

            @property
            def kps(self) -> Optional[np.ndarray]:
                return self.joint_info.kps

            @property
            def kds(self) -> Optional[np.ndarray]:
                return self.joint_info.kps

            @property
            def euler_axis_local(self) -> Optional[np.ndarray]:
                return self.joint_info.euler_axis_local

            @euler_axis_local.setter
            def euler_axis_local(self, value: np.ndarray):
                self.joint_info.euler_axis_local = value

    class JointInfoInit:
        class JointInfoInit:
            def __init__(self, joint_info):
                self.joint_info = joint_info
                self.euler_axis_local = None

            @property
            def world(self) -> ode.World:
                return self.joint_info.world

            @property
            def joints(self) -> List[ode.Joint]:
                return self.joint_info.joints

            @joints.setter
            def joints(self, value: Optional[ode.Joint]):
                self.joint_info.joints = value

            def __len__(self) -> int:
                return len(self.joint_info)

            def joint_names(self) -> List[str]:
                return self.joint_info.joint_names()

            @property
            def pa_joint_id(self) -> List[int]:
                return self.joint_info.pa_joint_id

            @property
            def kps(self) -> Optional[np.ndarray]:
                return self.joint_info.kps

            @property
            def kds(self) -> Optional[np.ndarray]:
                return self.joint_info.kps

            @property
            def euler_axis_local(self) -> Optional[np.ndarray]:
                return self.joint_info.euler_axis_local

            @euler_axis_local.setter
            def euler_axis_local(self, value: np.ndarray):
                self.joint_info.euler_axis_local = value

            def calc_joint_c_id(self):
                self.joint_info.joint_c_id = self.world.jointListToNumpy(self.joints)
                hinges: List[ode.HingeJoint] = self.joint_info.hinge_joints()
                if len(hinges) > 0:
                    self.joint_info.hinge_c_id = self.world.jointListToNumpy(hinges)

            def init_after_load(self):
                self.calc_joint_c_id()
                self.joint_info.weights = np.ones(len(self.joints))

                self.joint_info.resize_euler_axis_local()
                for idx, joint in enumerate(self.joints):
                    self.euler_axis_local[idx, :, :] = joint.euler_axis
                self.euler_axis_local = np.ascontiguousarray(self.euler_axis_local)

            @staticmethod
            def set_ball_joint_limit(joint: ode.BallJointAmotor, euler_order: str,
                                    angle_limits: Union[List, np.ndarray], raw_axis: Optional[np.ndarray] = None):
                raw_axis = np.eye(3) if raw_axis is None else raw_axis
                angle_limits = np.deg2rad(np.asarray(angle_limits))
                assert raw_axis.shape == (3, 3) and len(euler_order) == 3

                euler_order = euler_order.upper()
                if euler_order in ["XZY", "YXZ", "ZYX"]:  # swap angle limit
                    idx = ord(euler_order[2]) - ord('X')
                    angle_limits[idx] = np.array([-angle_limits[idx][1], -angle_limits[idx][0]])

                joint.setAmotorMode(ode.AMotorEuler)
                joint.setAmotorNumAxes(3)
                # if raw axis are same, the program will crash
                joint.setAmotorAxisNumpy(0, 1, raw_axis[ord(euler_order[0]) - ord("X")])  # Axis 0, body 1
                joint.setAmotorAxisNumpy(2, 2, raw_axis[ord(euler_order[2]) - ord("X")])  # Axis 2, body 2

                joint.setAngleLim1(*angle_limits[0])
                joint.setAngleLim2(*angle_limits[1])
                joint.setAngleLim3(*angle_limits[2])

    class ODECharacter:
        class DRootInitInfo:
            def __init__(self) -> None:
                self.pos: Optional[np.ndarray] = None
                self.quat: Optional[np.ndarray] = None

            def clear(self):
                self.pos: Optional[np.ndarray] = None
                self.quat: Optional[np.ndarray] = None

        class ODECharacter:
            def __init__(self, world: ode.World, space: ode.SpaceBase):
                self.name: str = "character"  # character name
                self.label: str = ""  # character label
                self.world: ode.World = world  # world in open dynamics engine
                self.space: ode.SpaceBase = space  # for collision detection
                self.character_id: int = 0  # The id of character
                self.root_init_info = None
                self.joint_info = ODESim.JointInfo.JointInfos(self.world)  # joint information
                self.body_info = ODESim.BodyInfo.BodyInfo(self.world, self.space)  # rigid body information
                self.end_joint = ODESim.EndJointInfo.EndJointInfo(self.world)  # End Joint in xml Human file. Just like end site of bvh file.

                self.joint_to_child_body: List[int] = []  # idx is joint id, joint_to_child_body[idx] is body id
                self.child_body_to_joint: List[int] = []  # idx is body id, child_body_to_joint[idx] is joint id

                self.joint_to_parent_body: List[int] = [] # idx is joint id, joint_to_parent_body[idx] is body id

                self.init_body_state = None  # initial body state

                self.height: float = 0.0  # height of character

                self.simulation_failed: bool = False
                self.fall_down: bool = False  # Fall Down Flag. Will be set in collision callback
                self.falldown_ratio = 0.0  # if com <= falldown_ratio * initial_com, we can say the character fall down
                self._self_collision: bool = True  # self collision detaction

                self._is_enable: bool = True
                self._is_kinematic: bool = False

                self.accum_loss: float = 0.0

                self.accum_energy: float = 0.0
                self.curr_frame_index = 0

            @property
            def bodies(self) -> List[ode.Body]:
                return self.body_info.bodies

            # get list of joints of rigid body
            @property
            def joints(self) -> List[ode.Joint]:
                return self.joint_info.joints

            # get the root rigid body
            @property
            def root_body(self) -> ode.Body:
                return self.body_info.root_body

            @property
            def root_body_pos(self) -> np.ndarray:
                return self.root_body.PositionNumpy

            @property
            def root_body_quat(self) -> np.ndarray:
                return self.root_body.getQuaternionScipy()

            @property
            def root_joint(self) -> Optional[ode.Joint]:
                return self.joint_info.root_joint

            def set_character_id(self, new_id: int):
                self.character_id = new_id
                return self

            # get global position of index-th rigid body
            def get_body_pos_at(self, index: int) -> np.ndarray:
                return self.body_info.get_body_pos_at(index)

            # get global velocity of index-th rigid body
            def get_body_velo_at(self, index: int) -> np.ndarray:
                return self.body_info.get_body_velo_at(index)

            # 
            def get_body_quat_at(self, index: int) -> np.ndarray:
                return self.body_info.get_body_quat_at(index)

            def get_body_rot_mat_at(self, index: int) -> np.ndarray:
                return self.body_info.get_body_rot_mat_at(index)

            def get_body_angvel_at(self, index: int) -> np.ndarray:
                return self.body_info.get_body_angvel_at(index)

            def get_body_name_list(self) -> List[str]:
                return self.body_info.get_name_list()

            def get_body_pos(self) -> np.ndarray:
                return self.body_info.get_body_pos()

            def get_body_velo(self) -> np.ndarray:
                return self.body_info.get_body_velo()

            def get_body_mat(self) -> np.ndarray:
                raise NotImplementedError

            # get quaternion of all of bodies
            def get_body_quat(self) -> np.ndarray:
                return self.body_info.get_body_quat()

            # get angular velocity of all of bodies
            def get_body_ang_velo(self) -> np.ndarray:
                return self.body_info.get_body_ang_velo()

            # get position of all of bodies
            def set_body_pos(self, pos: np.ndarray):
                self.body_info.set_body_pos(pos)

            def set_body_velo(self, velo: np.ndarray):
                self.body_info.set_body_velo(velo)

            # def set_body_quat(self, quat: np.ndarray):
            #    self.body_info.set_body_quat_rot

            def set_body_ang_velo(self, ang_velo: np.ndarray):
                self.body_info.set_body_ang_velo(ang_velo)

            def get_aabb(self) -> np.ndarray:
                """
                get character aabb
                """
                return self.body_info.get_aabb()

            @property
            def has_end_joint(self) -> bool:
                return self.end_joint is not None and len(self.end_joint) > 0

            @property
            def joint_weights(self) -> Optional[np.ndarray]:
                return self.joint_info.weights

            @property
            def end_joint_weights(self) -> Optional[np.ndarray]:
                return self.end_joint.weights if self.end_joint is not None else None

            @property
            def is_kinematic(self) -> bool:
                return self._is_kinematic

            @is_kinematic.setter
            def is_kinematic(self, value: bool):
                if self._is_kinematic == value:
                    return
                if value:
                    for body in self.bodies:
                        body.setKinematic()
                else:
                    for body in self.bodies:
                        body.setDynamic()
                self._is_kinematic = value

            # self collision of this character
            @property
            def self_collision(self) -> bool:
                return self._self_collision

            @self_collision.setter
            def self_collision(self, value: bool):
                if self._self_collision == value:
                    return
                for body in self.bodies:
                    for _geom in body.geom_iter():
                        geom: ode.GeomObject = _geom
                        geom.character_self_collide = int(value)
                self._self_collision = value

            @property
            def is_enable(self) -> bool:
                return self._is_enable

            @is_enable.setter
            def is_enable(self, value: bool):
                if self._is_enable == value:
                    return
                if value:
                    for body in self.bodies:
                        body.enable()
                        for geom in body.geom_iter():
                            geom.enable()
                    for joint in self.joints:
                        joint.enable()
                else:
                    for body in self.bodies:
                        body.disable()
                        for geom in body.geom_iter():
                            geom.disable()
                    for joint in self.joints:
                        joint.disable()

            def set_ode_space(self, space: Optional[ode.SpaceBase]):
                """
                set space of each geometry in character.
                """
                for body in self.bodies:
                    for g_iter in body.geom_iter():
                        geom: ode.GeomObject = g_iter
                        geom.space = space
                self.space = space

            def set_root_pos(self, pos: np.ndarray):
                # maybe can write in cython..
                raise NotImplementedError

            def save_init_state(self):
                """
                Save init state
                :return: initial state
                """
                if self.init_body_state is None:
                    self.init_body_state = self.save()
                return self.init_body_state

            def load_init_state(self) -> None:
                """
                load initial state
                """
                self.load(self.init_body_state)

            def save(self):
                """
                Save to BodyInfoState
                """
                if self.body_info.body_c_id is None:
                    self.body_info.calc_body_c_id()
                    # body id is created from body list.
                    # so, if the order of body list is fixed, the order of body id is fixed.
                body_state = ODESim.BodyInfoState.BodyInfoState()
                body_state.save(self.world, self.body_info.body_c_id)
                return body_state

            def load(self, body_state):
                """
                Load BodyInfoState
                """
                if self.body_info.body_c_id is None:
                    self.body_info.calc_body_c_id()
                body_state.load(self.world, self.body_info.body_c_id)
                self.accum_energy: float = 0.0

            # get joint name list.
            def get_joint_names(self, with_root: bool = False) -> List[str]:
                result = self.joint_info.joint_names()
                if with_root:
                    result = ["RootJoint"] + result
                return result

            def get_raw_anchor(self) -> Tuple[np.ndarray, np.ndarray]:
                """
                joint's body1 raw anchor, joint's body2 raw anchor
                """
                return self.world.getBallAndHingeRawAnchor(self.joint_info.joint_c_id)

            def character_facing_coor_end_pos_old(self, facing_rot_inv: Union[Rotation, np.ndarray, None] = None) -> np.ndarray:
                """
                End Joints' Position in character's facing coordinate
                """
                end_a = self.end_joint.get_global_pos()
                if facing_rot_inv is None:
                    facing_rot_inv: Rotation = Rotation(self.body_info.calc_facing_quat(), copy=False).inv()
                elif isinstance(facing_rot_inv, np.ndarray):
                    facing_rot_inv: Rotation = Rotation(facing_rot_inv)

                root_pos = Common.MathHelper.vec_axis_to_zero(self.root_body.PositionNumpy, 1)
                return facing_rot_inv.apply(end_a - root_pos)

            # get position of end effectors in facing coordinate
            def character_facing_coor_end_pos(self, facing_rot_inv: Optional[np.ndarray] = None) -> np.ndarray:
                end_a = self.end_joint.get_global_pos()
                if facing_rot_inv is None:
                    facing_rot_inv = quat_inv_single_fast(self.body_info.calc_facing_quat())
                root_pos = self.bodies[0].PositionNumpy
                root_pos[1] = 0
                return quat_apply_forward_one2many_fast(facing_rot_inv[None, :], end_a - root_pos)

            def character_facing_coor_com_old(self, facing_rot_inv: Union[Rotation, np.ndarray, None] = None) -> np.ndarray:
                """
                character's CoM in facing coordinate
                """
                if facing_rot_inv is None:
                    facing_rot_inv: Rotation = Rotation(self.body_info.calc_facing_quat(), copy=False).inv()
                elif isinstance(facing_rot_inv, np.ndarray):
                    facing_rot_inv: Rotation = Rotation(facing_rot_inv)

                root_pos = Common.MathHelper.vec_axis_to_zero(self.root_body.PositionNumpy, 1)
                return facing_rot_inv.apply(self.body_info.calc_center_of_mass() - root_pos)

            def character_facing_coor_com(self, facing_rot_inv: Optional[np.ndarray] = None, com: Optional[np.ndarray] = None) -> np.ndarray:
                """
                character's CoM in facing coordinate
                """
                if facing_rot_inv is None:
                    facing_rot_inv = quat_inv_single_fast(self.body_info.calc_facing_quat())
                if com is None:
                    com = self.body_info.calc_center_of_mass()
                root_pos = self.bodies[0].PositionNumpy
                root_pos[1] = 0
                return quat_apply_single_fast(facing_rot_inv, com - root_pos)

            def character_facing_coor_com_velo(self) -> np.ndarray:
                """
                character's CoM's velocity in facing coordinate
                """
                ry_rot_inv: Rotation = Rotation(self.body_info.calc_facing_quat()).inv()
                com_velo: np.ndarray = self.body_info.calc_momentum() / self.body_info.sum_mass
                root_velo: np.ndarray = Common.MathHelper.vec_axis_to_zero(self.root_body.LinearVelNumpy, 1)
                return ry_rot_inv.apply(com_velo - root_velo)

            def character_facing_coord_angular_momentum(self) -> np.ndarray:
                """
                character's angular momentum in facing coordinate
                """
                ry_rot_inv: Rotation = Rotation(self.body_info.calc_facing_quat(), copy=False).inv()
                angular_momentum = self.body_info.calc_angular_momentum()
                return ry_rot_inv.apply(angular_momentum)

            def calc_kinetic_energy(self) -> np.ndarray:
                """
                1/2*m*v^2 + 1/2*w^T*I*w
                """
                velo: np.ndarray = self.body_info.get_body_velo()
                omega: np.ndarray = self.body_info.get_body_ang_velo()
                mass: np.ndarray = self.body_info.mass_val
                inertia: np.ndarray = self.body_info.get_body_inertia()
                v2 = np.sum(velo ** 2, axis=-1)  # in shape (num body,)
                eng1 = np.sum(mass * v2)
                eng2 = omega.reshape((-1, 1, 3)) @ inertia @ omega.reshape((-1, 3, 1))
                eng2 = np.sum(eng2)
                res: np.ndarray = 0.5 * (eng1 + eng2)
                return res

            def cat_root_child_body_value(self, root_value: np.ndarray, child_body_value: np.ndarray, dtype=np.float64):
                """
                cat value for root body and child body
                root_value.shape == (batch size, num value)
                child_body.shape == (batch size, num body - 1, num value)
                """
                assert not self.joint_info.has_root
                assert root_value.ndim == 2 and child_body_value.ndim == 3 and child_body_value.shape[1] == len(self.bodies) - 1
                assert root_value.shape[0] == child_body_value.shape[0] and root_value.shape[-1] == child_body_value.shape[-1]
                res: np.ndarray = np.zeros((root_value.shape[0], len(self.bodies), root_value.shape[-1]), dtype=dtype)
                res[:, self.body_info.root_body_id, :] = root_value
                res[:, self.joint_to_child_body, :] = child_body_value
                return np.ascontiguousarray(res)

            def init_root_body_pos(self) -> np.ndarray:
                """
                initial root position
                """
                return self.init_body_state.pos.reshape((-1, 3))[self.body_info.root_body_id].copy()

            def init_root_quat(self) -> np.ndarray:
                """
                initial root quaternion
                """
                return self.init_body_state.quat.reshape((-1, 4))[self.body_info.root_body_id].copy()

            # set the render color in draw stuff
            def set_render_color(self, color: np.ndarray):
                for body in self.bodies:
                    for geom_ in body.geom_iter():
                        geom: ode.GeomObject = geom_
                        geom.render_by_default_color = 0
                        geom.render_user_color = color

            def enable_all_clung_env(self) -> None:
                for body in self.bodies:
                    for geom in body.geom_iter():
                        geom.clung_env = True

            def disable_all_clung_env(self) -> None:
                for body in self.bodies:
                    for geom in body.geom_iter():
                        geom.clung_env = False

            def set_clung_env(self, body_names: Iterable[str], value: bool = True):
                names: Set[str] = set(body_names)
                for body in self.bodies:
                    if body.name in names:
                        for geom in body.geom_iter():
                            geom.clung_env = value

            # set the max friction for each geometry when contact mode is max force
            def set_geom_max_friction(self, coef: float = 3.0) -> None:
                value: float = coef * self.body_info.sum_mass * np.linalg.norm(self.world.getGravityNumpy())
                for body in self.bodies:
                    for geom in body.geom_iter():
                        geom.max_friction = value

            # clear the character 
            def clear(self):
                self.joint_info.clear()
                self.body_info.clear()
                self.end_joint.clear()
                self.joint_to_child_body.clear()
                self.child_body_to_joint.clear()
                self.joint_to_parent_body.clear()
                self.init_body_state = None
                self.fall_down: bool = False

                return self

            def check_root(self):
                """
                check root joint and root body
                :return:
                """
                if self.bodies and self.root_body is None:
                    raise ValueError("There should be a root body")

                if self.root_joint is not None:
                    if self.root_joint.getNumBodies() != 1:
                        raise ValueError("root joint should only has 1 child")
                    if self.root_joint.body1 != self.root_body and self.root_joint.body2 != self.root_body:
                        raise ValueError("Root Body should be attached to root joint.")

            # move the character to new position
            # new_pos.shape == (3,)
            def move_character(self, new_pos: np.ndarray) -> None:
                assert new_pos.shape == (3,)
                delta_pos: np.ndarray = (new_pos - self.root_body.PositionNumpy).reshape((1, 3))
                old_pos = self.body_info.get_body_pos()
                new_pos = delta_pos + old_pos
                self.body_info.set_body_pos(new_pos)

            def move_character_by_delta(self, delta_pos: np.ndarray) -> None:
                assert delta_pos.shape ==  (3,)
                old_pos = self.body_info.get_body_pos()
                new_pos = delta_pos + old_pos
                self.body_info.set_body_pos(new_pos)

            def rotate_character(self):
                pass

            @staticmethod
            def rotate_body_info_state_y_axis(state, angle: float, use_delta_angle: bool = False):
                """
                rotate the BodyInfoState by y axis
                return: BodyInfoState

                For position, move to the original position, and rotate, then move back
                For rotation, rotate directly. note that rotation matrix should be recomputed.
                For linear velocity and angular velocity, rotate directly.
                Test:
                After rotate, the simulation result should match
                """
                result = state.copy().reshape()
                num_body = result.pos.shape[0]
                delta_pos: np.ndarray = result.pos[None, 0].copy()
                delta_pos[0, 1] = 0
                root_quat = result.quat[0]
                facing_quat = decompose_rotation_single_fast(root_quat, np.array([0.0, 1.0, 0.0]))
                facing_angle = 2 * np.arctan2(facing_quat[1], facing_quat[3])
                if use_delta_angle:
                    delta_angle = angle
                else:
                    delta_angle = angle - facing_angle
                delta_quat = np.array([0.0, np.sin(0.5 * delta_angle), 0.0, np.cos(0.5 * delta_angle)])
                delta_quat = np.ascontiguousarray(np.tile(delta_quat, (num_body, 1)))
                result.pos = quat_apply_forward_fast(delta_quat, result.pos - delta_pos).reshape(-1)
                result.quat = quat_multiply_forward_fast(delta_quat, result.quat).reshape(-1)
                result.rot = quat_to_matrix_fast(result.quat).reshape(-1)
                result.linear_vel = quat_apply_forward_fast(delta_quat, result.linear_vel).reshape(-1)
                result.angular_vel = quat_apply_forward_fast(delta_quat, result.angular_vel).reshape(-1)
                return result

            def rotate_y_axis(self, angle: float, use_delta_angle: bool = False):
                next_state = self.rotate_body_info_state_y_axis(self.save(), angle, use_delta_angle)
                self.load(next_state)
                return next_state

    class CharacterWrapper:
        """
        Wrapper of ODE Character
        """
        def __init__(self, character = None):
            self.character = character
            self.root_body = None

        @property
        def body_info(self):
            """
            get body info
            """
            return self.character.body_info

        @property
        def joint_info(self):
            """
            get joint info
            """
            return self.character.joint_info

        def joint_names(self) -> List[str]:
            """
            get joint names
            """
            return self.joint_info.joint_names()

        def body_names(self) -> List[str]:
            """
            get body names
            """
            return self.body_info.get_name_list()

        @property
        def end_joint(self):
            return self.character.end_joint

        @property
        def world(self) -> ode.World:
            return self.character.world

        @property
        def space(self) -> ode.SpaceBase:
            return self.character.space

        @property
        def bodies(self) -> List[ode.Body]:
            return self.character.bodies

        @property
        def joints(self) -> List[Union[ode.Joint, ode.BallJoint, ode.BallJointAmotor, ode.HingeJoint]]:
            return self.character.joints

        @property
        def root_joint(self) -> Optional[ode.Joint]:
            return self.character.root_joint

        @property
        def joint_to_child_body(self) -> List[int]:
            return self.character.joint_to_child_body

        @property
        def child_body_to_joint(self) -> List[int]:
            return self.character.child_body_to_joint

        @property
        def joint_to_parent_body(self) -> List[int]:
            return self.character.joint_to_parent_body

        @property
        def has_end_joint(self) -> bool:
            return self.character.has_end_joint

    class ODECharacterInit(CharacterWrapper):
        def __init__(self, character):
            super().__init__(character)
            self.joint_init = ODESim.JointInfoInit.JointInfoInit(self.joint_info)

        def init_after_load(self, character_id: int = 0,
                            ignore_parent_collision: bool = True,
                            ignore_grandpa_collision: bool = True,
                            ):
            """
            initialize character after loading from (xml) configure file
            """
            self.character.character_id = character_id
            self.joint_init.init_after_load()
            self.body_info.init_after_load(ignore_parent_collision, ignore_grandpa_collision)
            self.calc_map_body_joint()
            self.calc_joint_parent_idx()
            self.calc_height()
            # self.set_geom_clung()
            self.set_geom_character_id(character_id)
            self.set_geom_max_friction()
            self.set_geom_index()
            self.calc_joint_parent_body_c_id()
            self.calc_joint_child_body_c_id()
            self.character.save_init_state()
            self.space.ResortGeoms()

        def calc_map_body_joint(self):
            """

            """
            # self.character.joint_to_child_body = [-1] * len(self.joint_info)
            self.character.child_body_to_joint = [-1] * len(self.body_info)
            self.character.joint_to_parent_body = [-1] * len(self.joint_info)  # used in PDController
            # self.character.parent_body_to_joint = [-1] * len(self.body_info)

            # for j_idx, ch_body_name in enumerate(self.joint_info.child_body_name):  # joint->child_body
            #    self.joint_to_child_body[j_idx] = self.body_info.body_idx_dict[ch_body_name]
            # print(self.joint_to_child_body)
            for j_idx, b_idx in enumerate(self.joint_to_child_body):
                self.child_body_to_joint[b_idx] = j_idx

            for j_idx, b_idx in enumerate(self.joint_to_child_body):
                self.joint_to_parent_body[j_idx] = self.body_info.parent[b_idx]

        # Calc parent joint's id
        def calc_joint_parent_idx(self):
            """
            Calc parent joint id of each joint.
            requirement:
            :return:
            """
            self.joint_info.pa_joint_id = [-1] * len(self.joint_info)
            for j_idx, b_idx in enumerate(self.joint_to_child_body):
                if b_idx == -1:
                    continue
                pa_b_idx = self.body_info.parent[b_idx]
                if pa_b_idx == -1:  # Root Body
                    continue
                self.joint_info.pa_joint_id[j_idx] = self.child_body_to_joint[pa_b_idx]

        def init_end_joint_pa_joint_id(self, init_c_id: bool = True):
            """
            Calc parent joint id of each end joint.
            requirement: self.end_joint.pa_body_id, self.child_body_to_joint
            :param init_c_id:
            :return:
            """
            self.end_joint.pa_joint_id = np.array([self.child_body_to_joint[i] for i in self.end_joint.pa_body_id])
            if init_c_id:
                self.end_joint.pa_joint_c_id = np.array([
                    self.joints[i].get_jid()
                    for i in self.end_joint.pa_joint_id], dtype=np.uint64)

        def init_end_joint(self, names: List[str], parent_body_ids: List[int], end_pos: np.ndarray):
            """
            initialize end joints
            """
            # name
            self.end_joint.name = names
            self.end_joint.weights = np.ones(len(names))

            # parent body id
            self.end_joint.pa_body_id = np.array(parent_body_ids)
            # np.array([self.body_info.body_idx_dict[i] for i in names])
            self.end_joint.pa_body_c_id = np.array([self.bodies[i].get_bid()
                                                    for i in self.end_joint.pa_body_id], dtype=np.uint64)

            # parent body quaternion
            body_quat = self.world.getBodyQuatScipy(self.end_joint.pa_body_c_id).reshape((-1, 4))
            rot_inv: Rotation = Rotation(body_quat).inv()

            # position relative to parent body
            self.end_joint.init_global_pos = end_pos
            self.end_joint.jtob_init_global_pos = end_pos - self.world.getBodyPos(
                self.end_joint.pa_body_c_id).reshape((-1, 3))
            self.end_joint.jtob_init_local_pos = rot_inv.apply(self.end_joint.jtob_init_global_pos)

            # parent joint id
            self.init_end_joint_pa_joint_id(True)

            # global position of parent joint (anchor 1)
            pa_joint_global_pos = self.world.getBallAndHingeAnchor1(self.end_joint.pa_joint_c_id).reshape((-1, 3))

            # position relative to parent joint in world coordinate
            self.end_joint.jtoj_init_global_pos = end_pos - pa_joint_global_pos

            # position relative to parent joint in local coordinate
            self.end_joint.jtoj_init_local_pos = rot_inv.apply(self.end_joint.jtoj_init_global_pos)

        def calc_height(self):
            """
            compute character's height by AABB bounding box
            """
            aabb = self.body_info.get_aabb()
            self.character.height = aabb[3] - aabb[2]

        def set_has_root(self):
            self.joint_info.root_idx = len(self.joint_info)
            self.joint_info.has_root = True

        def add_root_joint(self):
            # load Root Joint as Ball Joint
            joint = ode.BallJoint(self.world)
            # assume that body 0 is the root body..
            root_body = self.body_info.root_body
            joint.setAnchor(root_body.PositionNumpy)
            joint.attach(root_body, ode.environment)
            joint.name = "RootJoint"
            self.set_has_root()
            self.joint_info.joints.append(joint)

        @staticmethod
        def compute_geom_mass_attr(
            body: ode.Body,
            create_geom: List[ode.GeomObject],
            gmasses: List[ode.Mass],
            gcenters: List,
            grots: List[Rotation],
            update_body_pos_by_com: bool = True
        ):
            # Body's Position is com
            # Calc COM and set body's position
            ms: np.ndarray = np.array([i.mass for i in gmasses])
            com: np.ndarray = ms[np.newaxis, :] @ np.asarray(gcenters) / np.sum(ms)
            if update_body_pos_by_com:
                body.PositionNumpy = com
            mass_total = ode.Mass()
            for g_idx, geom in enumerate(create_geom):
                geom.body = body
                geom.setOffsetWorldPositionNumpy(gcenters[g_idx])
                geom.setOffsetWorldRotationNumpy(grots[g_idx].as_matrix().flatten())
                # Rotation of body is 0, so setOffsetWorldRotation and setOffsetRotation is same.

                geom_inertia = ode.Inertia()
                geom_inertia.setFromMassClass(gmasses[g_idx])
                geom_inertia.RotInertia(grots[g_idx].as_matrix().flatten())
                geom_inertia.TransInertiaNumpy(-gcenters[g_idx] + com)
                mass_total.add(geom_inertia.toMass())
            return mass_total

        def append_body(self, body: ode.Body, mass_total: ode.Mass, name: str, parent: Optional[int]):
            """
            param:
            body: ode.Body,
            mass_total: total mass of body
            name: body's name
            idx: body's index
            parent: body's parent's index
            """
            body.setMass(mass_total)
            body.name = name
            self.body_info.parent.append(parent if parent is not None else -1)
            self.body_info.bodies.append(body)

        def set_geom_character_id(self, character_id: int = 0):
            """
            set character_id of each ode GeomObject.
            used in collision detection: To judge whether one character is collided with other character..
            """
            for body in self.bodies:
                for geom in body.geom_iter():
                    geom.character_id = character_id
                    geom.character = self.character  # character attr in geom is weakref

        def set_geom_max_friction(self, coef: float = 3.0):
            self.character.set_geom_max_friction(coef)

        def set_geom_index(self):
            cnt = 0
            for body in self.bodies:
                for g in body.geom_iter():
                    geom: ode.GeomObject = g
                    geom.geom_index = cnt
                    cnt += 1

        def calc_joint_parent_body_c_id(self):
            """

            """
            self.joint_info.parent_body_index = np.zeros(len(self.joints), dtype=np.uint64)
            self.joint_info.parent_body_c_id = np.zeros(len(self.joints), dtype=np.uint64)
            for idx, joint in enumerate(self.joints):
                pa_body_index = self.character.joint_to_parent_body[idx]
                if pa_body_index != -1:
                    pa_body = self.bodies[pa_body_index]
                else:
                    pa_body = None

                assert joint.body2 == pa_body
                self.joint_info.parent_body_index[idx] = pa_body_index
                self.joint_info.parent_body_c_id[idx] = pa_body.get_bid() if pa_body is not None else 0

            # print(self.joint_info.parent_body_c_id)
            # set root joint's parent body c id to NULL
            if self.joint_info.has_root:
                self.joint_info.parent_body_c_id[self.joint_info.root_idx] = 0

        def calc_joint_child_body_c_id(self):
            """

            """
            self.joint_info.child_body_index = np.zeros(len(self.joints), dtype=np.uint64)
            self.joint_info.child_body_c_id = np.zeros(len(self.joints), dtype=np.uint64)
            # joint should always have child body.
            for j_idx, b_idx in enumerate(self.joint_to_child_body):
                joint = self.joint_info.joints[j_idx]
                child_body = self.bodies[b_idx]
                assert child_body == joint.body1
                self.joint_info.child_body_index[j_idx] = b_idx
                self.joint_info.child_body_c_id[j_idx] = child_body.get_bid()



    class BVHToTarget:
        class BVHToTargetBase():
            """
            Convert bvh motion data to target pose
            """
            def __init__(
                self,
                bvh_data: Union[str, pymotionlib.MotionData.MotionData],
                bvh_fps: int,
                character,
                ignore_root_xz_pos: bool = False,
                bvh_start: Optional[int] = None,
                bvh_end: Optional[int] = None,
                set_init_state_as_offset: bool = False,
                smooth_type: Union[Common.SmoothOperator.GaussianBase, Common.SmoothOperator.ButterWorthBase, None] = None,
                flip = None
            ):
                self.character = character
                # load the character as initial state
                self.character.load_init_state()

                # assume there is no root joint
                # if self.character.joint_info.has_root:
                #    raise ValueError("Assume there is no root joint.")

                # assume there are end joints
                if not self.character.has_end_joint:
                    raise ValueError("End Joint required.")

                # Load BVH File
                if isinstance(bvh_data, str):
                    self.bvh = pymotionlib.BVHLoader.load(bvh_data, ignore_root_xz_pos=ignore_root_xz_pos)

                    # TODO: Modify input bvh..
                    # dh1 = self.character.body_info.get_aabb()[2] - np.min(self.bvh.joint_position[0, :, 1])
                    # dh = -self.character.root_body.PositionNumpy[1] + self.bvh.joint_position[0, 0, 1]
                    # MotionHelper.adjust_root_height(self.bvh, dh)
                elif isinstance(bvh_data, pymotionlib.MotionData.MotionData):
                    self.bvh = bvh_data
                else:
                    raise ValueError
                if flip:
                    self.bvh.flip(flip)
                self.bvh = self.bvh.sub_sequence(bvh_start, bvh_end, copy=False)

                if set_init_state_as_offset:
                    self.set_init_state_as_bvh_offset()

                self.smooth_bvh: Optional[pymotionlib.MotionData.MotionData] = None
                if smooth_type is not None:
                    self.do_smooth(smooth_type)  # if we don't use smooth, inverse dynamics result will be very noisy..
                    self.smooth_bvh.resample(int(bvh_fps))

                if int(bvh_fps) != self.bvh.fps:
                    self.bvh.resample(int(bvh_fps))  # resample

                self.mapper = ODESim.Utils.BVHJointMap(self.bvh, character)

                # Get Raw anchor
                anchor_res = self.character.get_raw_anchor()
                self.raw_anchor1: np.ndarray = anchor_res[0].reshape((-1, 3))  # (joint, 3)

                # actually, raw anchor 2 is not used in this program..
                # self.raw_anchor2: np.ndarray = anchor_res[1].reshape((-1, 3))  # (joint, 3)

                if self.character.root_init_info is not None:
                    self.root_body_offset = self.character.init_root_body_pos() - self.character.root_init_info.pos
                else:
                    self.root_body_offset = None

            @property
            def body_info(self):
                """
                get body info
                """
                return self.character.body_info

            @property
            def joint_info(self):
                """
                get joint info
                """
                return self.character.joint_info

            def joint_names(self) -> List[str]:
                """
                get joint names
                """
                return self.joint_info.joint_names()

            def body_names(self) -> List[str]:
                """
                get body names
                """
                return self.body_info.get_name_list()

            @property
            def end_joint(self):
                return self.character.end_joint

            @property
            def world(self) -> ode.World:
                return self.character.world

            @property
            def space(self) -> ode.SpaceBase:
                return self.character.space

            @property
            def bodies(self) -> List[ode.Body]:
                return self.character.bodies

            @property
            def joints(self) -> List[Union[ode.Joint, ode.BallJoint, ode.BallJointAmotor, ode.HingeJoint]]:
                return self.character.joints

            @property
            def root_joint(self) -> Optional[ode.Joint]:
                return self.character.root_joint

            @property
            def joint_to_child_body(self) -> List[int]:
                return self.character.joint_to_child_body

            @property
            def child_body_to_joint(self) -> List[int]:
                return self.character.child_body_to_joint

            @property
            def joint_to_parent_body(self) -> List[int]:
                return self.character.joint_to_parent_body

            @property
            def has_end_joint(self) -> bool:
                return self.character.has_end_joint

            @property
            def bvh_children(self):
                return self.mapper.bvh_children

            @property
            def character_to_bvh(self):
                return self.mapper.character_to_bvh

            @property
            def end_to_bvh(self):
                return self.mapper.end_to_bvh

            @property
            def bvh_joint_cnt(self):
                """
                bvh joint count
                """
                return len(self.bvh.joint_names)

            @property
            def frame_cnt(self):
                """
                frame count
                """
                return self.bvh.num_frames

            @property
            def character_jcnt(self):
                return len(self.character.joint_info)

            def set_bvh_offset(self, pos_offset: Optional[np.ndarray] = None, quat_offset: Optional[np.ndarray] = None):
                if pos_offset is not None:
                    assert pos_offset.shape == (3,)
                    delta_pos: np.ndarray = pos_offset - self.bvh.joint_position[0, 0, :]
                    self.bvh._joint_translation[:, 0, :] += delta_pos

                if quat_offset is not None:
                    assert quat_offset.shape == (4,)
                    dq = (Rotation(quat_offset, normalize=False, copy=False)
                        * Rotation(self.bvh.joint_rotation[0, 0, :], copy=False).inv()).as_quat()
                    dq, _ = Common.MathHelper.facing_decompose(dq)
                    dr = Rotation(dq, copy=False)
                    self.bvh._joint_rotation[:, 0, :] = (dr * Rotation(self.bvh.joint_rotation[:, 0, :], copy=False)).as_quat()
                    trans_0 = self.bvh.joint_translation[0, 0, :]
                    d_trans = self.bvh.joint_translation[:, 0, :] - trans_0
                    self.bvh.joint_translation[:, 0, :] = trans_0 + dr.apply(d_trans)

                if pos_offset is not None or quat_offset is not None:
                    self.bvh.recompute_joint_global_info()

            def set_init_state_as_bvh_offset(self):
                self.set_bvh_offset(self.character.init_root_body_pos(), self.character.init_root_quat())

            # def refine_hinge_rotation(self):
            #     """
            #     Sometimes, elbow and knee joint may have rotation along other axies..
            #     We should remove these rotations...
            #     """
            #     hinge_id = self.joint_info.hinge_id()
            #     for hid in hinge_id:
            #         axis = self.joints[hid].Axis1RawNumpy
            #         bvh_idx = self.character_to_bvh[hid]
            #         qa, q_noise = MathHelper.axis_decompose(self.bvh.joint_rotation[:, bvh_idx, :], np.array(axis))
            #         self.bvh._joint_rotation[:, bvh_idx, :] = qa

            #     self.bvh.recompute_joint_global_info()

            def do_smooth(self, smooth_type: Union[Common.SmoothOperator.GaussianBase, Common.SmoothOperator.ButterWorthBase], test_out_fname: Optional[str] = None):
                self.smooth_bvh: Optional[pymotionlib.MotionData.MotionData] = Utils.smooth_motion_data(copy.deepcopy(self.bvh), smooth_type, test_out_fname)

            def init_target_global(self, target, vel_forward: bool = False):
                """
                initialize target in global coordinate
                """
                #  (frame, joint, 3 or 4)
                target.globally.quat = self.bvh.joint_orientation[:, self.character_to_bvh, :]
                target.globally.pos = self.bvh.joint_position[:, self.character_to_bvh, :]

                global_lin_vel = self.bvh.compute_linear_velocity(vel_forward)
                target.globally.linvel = global_lin_vel[:, self.character_to_bvh, :]
                global_ang_vel = self.bvh.compute_angular_velocity(vel_forward)
                target.globally.angvel = global_ang_vel[:, self.character_to_bvh, :]

                return global_lin_vel, global_ang_vel

            def init_target_root(
                self,
                target,
                global_lin_vel: np.ndarray,
                global_ang_vel: np.ndarray
            ):
                """
                Convert bvh to root info
                This method is OK with root joint
                """
                target.root.pos = self.bvh.joint_position[:, 0, :]
                target.root.quat = self.bvh.joint_orientation[:, 0, :]
                target.root.linvel = global_lin_vel[:, 0, :]
                target.root.angvel = global_ang_vel[:, 0, :]

                # compute root body info..
                # there is offset between root joint and root body
                if self.root_body_offset is not None:
                    target.root_body.pos = Rotation(target.root.quat).apply(self.root_body_offset) + target.root.pos
                    target.root_body.linvel = Common.MathHelper.vec_diff(target.root_body.pos, False, self.bvh.fps)
                    target.root_body.quat = target.root.quat
                    target.root_body.angvel = target.root.angvel
                else:
                    target.root_body = target.root

            def init_facing_root(
                self,
                target,
                global_lin_vel: np.ndarray,
                global_ang_vel: np.ndarray
            ):
                target.facing_root.pos = Common.MathHelper.vec_axis_to_zero(self.bvh.joint_position[:, 0, :], [0, 2])
                ry, facing = Common.MathHelper.facing_decompose(self.bvh.joint_orientation[:, 0, :])
                target.facing_root.quat = facing
                target.facing_root.linvel = Common.MathHelper.vec_axis_to_zero(global_lin_vel[:, :], [0, 2])
                target.facing_root.angvel = Rotation(ry, copy=False).apply(global_ang_vel[:, :])

            def init_locally_coor(self, target, vel_forward: bool = False):
                """
                convert bvh local rotation to target
                """
                target.locally.quat = self.bvh.joint_rotation[:, self.character_to_bvh, :]
                local_ang_vel: np.ndarray = self.bvh.compute_rotational_speed(vel_forward)
                target.locally.angvel = local_ang_vel[:, self.character_to_bvh, :]

            def init_end(
                self,
                target
            ):
                """
                initialize end joints' target info
                """
                target.end.pos = self.bvh.joint_position[:, self.end_to_bvh, :]

            @staticmethod
            def calc_facing_quat(target):
                target.facing_quat, _ = Common.MathHelper.facing_decompose(target.root.quat)

            def init_facing_coor_end(self, target):
                """
                convert bvh end sites to target in facing coordinate
                """
                root_pos = Common.MathHelper.vec_axis_to_zero(target.root.pos, 1)
                ry_rot_inv = Rotation(target.facing_quat).inv()
                target.facing_coor_end.pos = np.copy(target.end.pos)
                for end_idx in range(len(self.end_to_bvh)):
                    target.facing_coor_end.pos[:, end_idx, :] = \
                        ry_rot_inv.apply(target.end.pos[:, end_idx, :] - root_pos)
                target.facing_coor_end.pos = target.facing_coor_end.pos

            def init_global_child_body(self, target, vel_forward: bool = False):
                """
                convert bvh global info to target body in global coordinate
                """
                #

                target.child_body.pos = np.zeros((self.frame_cnt, self.character_jcnt, 3))
                target.child_body.quat = np.copy(target.globally.quat)
                for jidx in range(self.character_jcnt):
                    rot = Rotation(target.globally.quat[:, jidx, :], copy=False)
                    target.child_body.pos[:, jidx, :] = \
                        target.globally.pos[:, jidx, :] - rot.apply(self.raw_anchor1[jidx])
                target.child_body.linvel = Common.MathHelper.vec_diff(target.child_body.pos, vel_forward, self.bvh.fps)

                # Calc Global Angular Velocity
                target.child_body.angvel = target.globally.angvel.copy()

            def init_all_joint_and_body(self, target):
                """
                joint with root global and local info, all child body info
                """
                facing_rot_inv = Rotation(target.facing_quat)
                if target.globally is not None:
                    # build all joint global
                    if self.joint_info.has_root:
                        target.all_joint_global = target.globally.deepcopy()
                    else:
                        target.all_joint_global.pos = np.concatenate([target.root.pos[:, None, :], target.globally.pos], axis=1)
                        target.all_joint_global.quat = np.concatenate([target.root.quat[:, None, :], target.globally.quat], axis=1)

                    # build facing joint pos and quat..
                    target.all_joint_facing = copy.deepcopy(target.all_joint_global)
                    target.all_joint_facing.pos[:, :, [0, 2]] -= target.all_joint_facing.pos[:, 0:1, [0, 2]]
                    for index in range(target.all_joint_global.pos.shape[1]):
                        target.all_joint_facing.pos[:, index, :] = facing_rot_inv.apply(target.all_joint_facing.pos[:, index, :])
                        target.all_joint_facing.quat[:, index, :] = (facing_rot_inv * Rotation(target.all_joint_global.quat[:, index, :])).as_quat()

                # build all joint local
                if target.locally is not None:
                    if self.joint_info.has_root:
                        target.all_joint_local = target.locally.deepcopy()
                    else:
                        res = target.all_joint_local
                        # res.locally.pos = np.concatenate([pose.root.pos[:, None, :], pose.locally.pos], axis=1)
                        res.quat = np.concatenate([target.root.quat[:, None, :], target.locally.quat], axis=1)
                        res.angvel = np.concatenate([target.root.angvel[:, None, :], target.locally.angvel], axis=1)
                        # res.locally.linvel = np.concatenate([pose.root.linvel[:, None, :], pose.locally.linvel], axis=1)

                if not self.joint_info.has_root:
                    if target.child_body is not None:  # get body info correspond to character
                        res = target.character_body
                        cat_func = self.character.cat_root_child_body_value
                        res.pos = cat_func(target.root_body.pos, target.child_body.pos)
                        res.quat = cat_func(target.root_body.quat, target.child_body.quat)
                        res.rot_mat = Rotation(res.quat.reshape((-1, 4)), copy=False).as_matrix().reshape(
                            res.quat.shape[:-1] + (3, 3))
                        res.linvel = cat_func(target.root_body.linvel, target.child_body.linvel)
                        res.angvel = cat_func(target.root_body.angvel, target.child_body.angvel)

                    if target.child_body is not None:  # child body info (with root body..)
                        res = target.all_child_body
                        res.pos = np.concatenate([target.root_body.pos[:, None, :], target.child_body.pos], axis=1)
                        res.quat = np.concatenate([target.root_body.quat[:, None, :], target.child_body.quat], axis=1)
                        res.linvel = np.concatenate([target.root_body.linvel[:, None, :], target.child_body.linvel], axis=1)
                        res.angvel = np.concatenate([target.root_body.angvel[:, None, :], target.child_body.angvel], axis=1)
                else:
                    if target.child_body is not None:
                        res = target.character_body
                        res.pos = target.child_body.pos[:, self.character.joint_to_child_body, :]
                        res.quat = target.child_body.quat[:, self.character.joint_to_child_body, :]
                        res.rot_mat = Rotation(res.quat.reshape((-1, 4)), copy=False).as_matrix().reshape(
                            res.quat.shape[:-1] + (3, 3))
                        res.linvel = target.child_body.linvel[:, self.character.joint_to_child_body, :]
                        res.angvel = target.child_body.angvel[:, self.character.joint_to_child_body, :]

            def init_smooth_target(self, target=None, vel_forward: bool = False):
                res = self.init_target(target, self.smooth_bvh, vel_forward)
                res.smoothed = True
                return res

            def only_init_global_target(self, vel_forward: bool = False):
                target = ODESim.TargetPose.TargetPose()
                global_vel = self.init_target_global(target, vel_forward)
                self.init_target_root(target, *global_vel)
                self.init_global_child_body(target, vel_forward)
                return target

            def init_target(
                self,
                target=None,
                bvh: Optional[pymotionlib.MotionData.MotionData] = None,
                vel_forward: bool = False,
                ):
                """
                Note:
                in ODE engine,
                a_t = F(x_t, v_t),
                v_{t + 1} = v_{t} + h * a_{t}
                x_{t + 1} = x_{t} + h * v_{t + 1}
                """
                if target is None:
                    target = ODESim.TargetPose.TargetPose()

                bvh_backup = self.bvh
                if bvh is not None:
                    self.bvh = bvh

                # Calc Target Pose. The index is character's joint index
                global_vel = self.init_target_global(target, vel_forward)
                self.init_target_root(target, *global_vel)
                self.init_global_child_body(target, vel_forward)

                self.init_locally_coor(target, vel_forward)

                self.calc_facing_quat(target)

                self.init_facing_root(target, target.root.linvel, target.root.angvel)
                self.init_end(target)
                self.init_facing_coor_end(target)
                self.init_all_joint_and_body(target)

                target.num_frames = self.bvh.num_frames
                target.fps = self.bvh.fps
                target.to_continuous()

                self.bvh = bvh_backup
                return target


            def calc_posi_by_rot(self, quat, root_posi):
                """
                calculate joints' global position from their global rotation
                """
                parent_idx_ = self.bvh.joint_parents_idx # 23
                joint_offset_ = self.bvh.joint_offsets # 23
                joint_num = len(parent_idx_)

                rot = np.zeros([joint_num, 4]) # 23
                for i in range(len(quat)):
                    rot[self.character_to_bvh[i]] = quat[i]

                joint_posi = np.zeros([joint_num, 3])

                for i in range(joint_num):
                    if parent_idx_[i] == -1:
                        joint_posi[i, :] = root_posi
                    else:
                        joint_posi[i, :] = joint_posi[parent_idx_[i], :] + Rotation.from_quat(rot[parent_idx_[i]]).apply(joint_offset_[i])

                return joint_posi # 23

            def calc_body_posi_by_rot(self, quat, joint_posi):
                joint_posi = joint_posi[self.character_to_bvh] # 18

                body_posi = np.zeros((self.character_jcnt, 3))
                body_quat = quat
                for jidx in range(self.character_jcnt):
                    rot = Rotation(quat[jidx, :], copy=False)
                    body_posi[jidx, :] = \
                        joint_posi[jidx, :] - rot.apply(self.raw_anchor1[jidx])

                return body_posi, body_quat

    class CharacterJointInfoRoot(CharacterWrapper):

        def __init__(self, character):
            super().__init__(character)

        def get_joint_dof(self) -> np.ndarray:
            dofs = np.zeros(len(self.joints), dtype=np.int32)
            for idx, joint in enumerate(self.joints):
                dofs[idx] = joint.joint_dof
            if self.joint_info.has_root:
                return dofs
            else:
                return np.concatenate([np.array([3], dtype=np.int32), dofs])

        def get_parent_joint_dof(self) -> np.ndarray:
            """
            get parent joint dof for each body
            used in Inverse Dynamics
            return: np.ndarray in shape (num body,)
            """
            dofs = np.zeros(len(self.bodies), dtype=np.int32)
            for body_idx, body in enumerate(self.bodies):
                pa_joint_idx: int = self.child_body_to_joint[body_idx]
                if pa_joint_idx != -1:  # has parent joint
                    dofs[body_idx] = self.joints[pa_joint_idx].joint_dof
                else:  # There is no parent joint
                    dofs[body_idx] = 3
            return dofs

        def get_parent_joint_pos(self) -> np.ndarray:
            """
            Get global position of parent joint of each body
            used in Inverse Dynamics
            return: np.ndarray in shape
            """
            result = np.zeros((len(self.bodies), 3), dtype=np.float64)
            joint_pos = self.joint_info.get_global_pos1()  # shape == (njoints, 3)
            index = np.asarray(self.child_body_to_joint, dtype=np.int32)
            if not self.joint_info.has_root:
                result[0, :] = self.root_body.PositionNumpy
                result[1:, :] = joint_pos[index[1:]]
            else:
                result = joint_pos
                result = result[index]
            # resort joints. result[i] is parent joint position of i-th body
            return np.ascontiguousarray(result)

        def get_parent_joint_euler_order(self) -> List[str]:
            """
            used in Inverse Dynamics
            return List[str] with length {num body}
            """
            res = self.joint_info.get_joint_euler_order()
            if not self.joint_info.has_root:
                res = ["XYZ"] + res
            return res

        def get_parent_joint_euler_axis(self) -> np.ndarray:
            """
            return
            """
            euler_axis = self.joint_info.euler_axis_local
            if not self.joint_info.has_root:
                euler_axis = np.concatenate([np.eye(3)[None, ...], euler_axis], axis=0)
            return np.ascontiguousarray(euler_axis)

        def get_parent_body_index(self) -> np.ndarray:
            result: np.ndarray = np.zeros(len(self.bodies), dtype=np.int32)
            for body_idx, joint_idx in enumerate(self.child_body_to_joint):
                if joint_idx == -1 or self.joints[joint_idx].body2 is None:
                    result[body_idx] = -1
                else:
                    result[body_idx] = self.joints[joint_idx].body2.instance_id
            return result


    class PDController:
        class PDControlerBase:
            def __init__(self, joint_info):
                self.tor_lim = joint_info.torque_limit[:, np.newaxis]
                self.kps = joint_info.kps[:, None]
                self.kds = joint_info.kds[:, None]
                self.world = joint_info.world
                self.joint_info = joint_info
                self.cache_global_torque: Optional[np.ndarray] = None
                self.cache_local_torque: Optional[np.ndarray] = None

            def _add_local_torque(self, parent_qs: np.ndarray, local_torques: np.ndarray) -> np.ndarray:
                """
                param: parent_qs: parent bodies' quaternion in global coordinate
                """
                global_torque: np.ndarray = Rotation(parent_qs, False, False).apply(local_torques)
                self.cache_global_torque = global_torque
                self.cache_local_torque = local_torques
                self.world.add_global_torque(global_torque, self.joint_info.parent_body_c_id, self.joint_info.child_body_c_id)
                return global_torque

            def _add_clipped_torque(self, parent_qs: np.ndarray, local_torques: np.ndarray) -> np.ndarray:
                """
                Clip torque to avoid Numerical explosion.
                Param:
                parent_qs: parent bodies' quaternion in global coordinate
                local_torques: torques added to joints in parent local coordinate
                """
                tor_len = np.linalg.norm(local_torques, axis=-1, keepdims=True)
                tor_len[tor_len < 1e-10] = 1
                ratio = np.clip(tor_len, -self.tor_lim, self.tor_lim)

                new_local_torque = (local_torques / tor_len) * ratio
                return self._add_local_torque(parent_qs, new_local_torque)

            def add_torques_by_quat(self, tar_local_qs: np.ndarray) -> np.ndarray:
                raise NotImplementedError


        # For ode.World.dampedStep
        class DampedPDControlerSlow(PDControlerBase):
            def __init__(self, joint_info):
                super().__init__(joint_info)

            def add_torques_by_quat(self, tar_local_qs: np.ndarray) -> np.ndarray:
                """
                Param:
                tar_local_qs: target joints' quaternion in parent local coordinate
                """
                parent_qs, child_qs, local_qs, parent_qs_inv = self.joint_info.get_parent_child_qs()

                delta_local_tar_now = Rotation(tar_local_qs, False, False) * Rotation(local_qs, False, False).inv()
                local_torques: np.ndarray = self.kps * delta_local_tar_now.as_rotvec()

                ret = self._add_clipped_torque(parent_qs, local_torques)

                # test C++ version of pd controller
                pd_ret = self.world.get_pd_control_torque(self.joint_info.joint_c_id, tar_local_qs, self.kps.reshape(-1), self.tor_lim.reshape(-1))
                c_local_torque, c_global_torque, tot_pow = pd_ret
                print("delta local", np.max(np.abs(c_local_torque - self.cache_local_torque)))
                print("delta global", np.max(np.abs(c_global_torque - self.cache_global_torque)))
                print()
                return ret


        class DampedPDControler:
            """
            using stable PD control.
            Please refer to [Liu et al. 2013 Simulation and Control of Skeleton-driven Soft Body Characters] for details
            """
            def __init__(self, character):
                self.character = character
                joint_info = character.joint_info
                self.joint_c_id: np.ndarray = joint_info.joint_c_id
                self.tor_lim = joint_info.torque_limit.flatten()
                self.kps = joint_info.kps.flatten()
                self.world = joint_info.world
                self.joint_info = joint_info
                self.cache_global_torque: Optional[np.ndarray] = None
                self.cache_local_torque: Optional[np.ndarray] = None

            def add_torques_by_quat(self, tar_local_qs: np.ndarray) -> np.ndarray:
                c_local_torque, c_global_torque, tot_power = self.world.get_pd_control_torque(self.joint_c_id, tar_local_qs, self.kps, self.tor_lim)
                self.cache_global_torque = c_global_torque
                self.cache_local_torque = c_local_torque
                self.world.add_global_torque(c_global_torque, self.joint_info.parent_body_c_id, self.joint_info.child_body_c_id)
                self.character.accum_energy += tot_power
                return c_global_torque


        # For ode.World.step
        class PDControler(PDControlerBase):
            def __init__(self, joint_info):
                super().__init__(joint_info)

            def add_torques_by_quat(self, tar_local_qs: np.ndarray) -> np.ndarray:
                parent_qs, child_qs, local_qs, parent_qs_inv = self.joint_info.get_parent_child_qs()

                delta_local_tar_now = Common.MathHelper.flip_quat_by_w(
                    Rotation(tar_local_qs, normalize=False, copy=False) *
                    (Rotation(local_qs, normalize=False, copy=False).inv()).as_quat())

                local_torques: np.ndarray = self.kps * Rotation(delta_local_tar_now, copy=False).as_rotvec() - \
                                            self.kds * self.joint_info.get_local_angvels(parent_qs_inv)

                return self._add_clipped_torque(parent_qs, local_torques)


    class TargetPose:

        class TargetBaseType:

            __slots__ = ("pos", "quat", "rot_mat", "linvel", "angvel", "linacc", "angacc")

            def __init__(self):
                self.pos: Optional[np.ndarray] = None  # Position
                self.quat: Optional[np.ndarray] = None  # Quaternion
                self.rot_mat: Optional[np.ndarray] = None

                self.linvel: Optional[np.ndarray] = None  # linear velocity
                self.angvel: Optional[np.ndarray] = None  # angular velocity

                self.linacc: Optional[np.ndarray] = None  # linear acceleration
                self.angacc: Optional[np.ndarray] = None  # angular acceleration

            def append(self, other):
                if self.pos is None:
                    self.pos = copy.deepcopy(other.pos)
                else:
                    self.pos = np.concatenate([self.pos, other.pos], axis=0)

                if self.quat is None:
                    self.quat = copy.deepcopy(other.quat)
                else:
                    self.quat = np.concatenate([self.quat, other.quat], axis=0)

                if self.rot_mat is None:
                    self.rot_mat = copy.deepcopy(other.rot_mat)
                else:
                    self.rot_mat = np.concatenate([self.rot_mat, other.rot_mat], axis=0)

                if self.linvel is None:
                    self.linvel = copy.deepcopy(other.linvel)
                else:
                    self.linvel = np.concatenate([self.linvel, other.linvel], axis=0)

                if self.angvel is None:
                    self.angvel = copy.deepcopy(other.angvel)
                else:
                    self.angvel = np.concatenate([self.angvel, other.angvel], axis=0)

                if self.linacc is None:
                    self.linacc = copy.deepcopy(other.linacc)
                else:
                    self.linacc = np.concatenate([self.linacc, other.linacc], axis=0)

                if self.angacc is None:
                    self.angacc = copy.deepcopy(other.angacc)
                else:
                    self.angacc = np.concatenate([self.angacc, other.angacc], axis=0)

                return self

            def duplicate(self, times: int = 1):
                res = ODESim.TargetPose.TargetBaseType()
                if self.pos is not None:
                    res.pos = np.concatenate([self.pos] * times, axis=0)
                if self.quat is not None:
                    res.quat = np.concatenate([self.quat] * times, axis=0)
                if self.rot_mat is not None:
                    res.rot_mat = np.concatenate([self.rot_mat] * times, axis=0)
                if self.linvel is not None:
                    res.linvel = np.concatenate([self.linvel] * times, axis=0)
                if self.angvel is not None:
                    res.angvel = np.concatenate([self.angvel] * times, axis=0)
                if self.linacc is not None:
                    res.linacc = np.concatenate([self.linacc] * times, axis=0)
                if self.angacc is not None:
                    res.angacc = np.concatenate([self.angacc] * times, axis=0)

                return res

            def deepcopy(self):
                return copy.deepcopy(self)

            def to_continuous(self):
                if self.pos is not None:
                    self.pos = np.ascontiguousarray(self.pos)

                if self.quat is not None:
                    self.quat = np.ascontiguousarray(self.quat)

                if self.rot_mat is not None:
                    self.rot_mat = np.ascontiguousarray(self.rot_mat)

                if self.linvel is not None:
                    self.linvel = np.ascontiguousarray(self.linvel)

                if self.angvel is not None:
                    self.angvel = np.ascontiguousarray(self.angvel)

                if self.linacc is not None:
                    self.linacc = np.ascontiguousarray(self.linacc)

                if self.angacc is not None:
                    self.angacc = np.ascontiguousarray(self.angacc)

            def resize(self, shape: Union[int, Iterable, Tuple[int]], dtype=np.float64):
                self.pos = np.zeros(shape + (3,), dtype=dtype)
                self.quat = np.zeros(shape + (4,), dtype=dtype)
                self.linvel = np.zeros(shape + (3,), dtype=dtype)
                self.angvel = np.zeros(shape + (3,), dtype=dtype)
                self.linacc = np.zeros(shape + (3,), dtype=dtype)
                self.angacc = np.zeros(shape + (3,), dtype=dtype)
                return self

            def __len__(self) -> int:
                if self.pos is not None:
                    return self.pos.shape[0]
                elif self.quat is not None:
                    return self.quat.shape[0]
                elif self.rot_mat is not None:
                    return self.rot_mat.shape[0]
                elif self.linvel is not None:
                    return self.linvel.shape[0]
                elif self.angvel is not None:
                    return self.angvel.shape[0]
                elif self.linacc is not None:
                    return self.linacc.shape[0]
                elif self.angacc is not None:
                    return self.angacc.shape[0]
                else:
                    return 0

            def sub_seq(self, start: int = 0, end: Optional[int] = None, skip: Optional[int] = None, is_copy: bool = True):
                """
                Get sub sequence of TargetBaseType
                """
                assert skip is None or isinstance(skip, int)
                res = ODESim.TargetPose.TargetBaseType()
                if end is None:
                    end = len(self)
                if end == 0:
                    return res

                piece = slice(start, end, skip)
                if self.pos is not None:
                    res.pos = self.pos[piece].copy() if is_copy else self.pos[piece]
                if self.quat is not None:
                    res.quat = self.quat[piece].copy() if is_copy else self.quat[piece]
                if self.rot_mat is not None:
                    res.rot_mat = self.rot_mat[piece].copy() if is_copy else self.rot_mat[piece]
                if self.linvel is not None:
                    res.linvel = self.linvel[piece].copy() if is_copy else self.linvel[piece]
                if self.angvel is not None:
                    res.angvel = self.angvel[piece].copy() if is_copy else self.angvel[piece]
                if self.linacc is not None:
                    res.linacc = self.linacc[piece].copy() if is_copy else self.linacc[piece]
                if self.angacc is not None:
                    res.angacc = self.angacc[piece].copy() if is_copy else self.angacc[piece]

                return res

            def __str__(self):
                res = "pos" + (" is None" if self.pos is None else ".shape = " + str(self.pos.shape)) + \
                    ". quat" + (" is None" if self.quat is None else ".shape = " + str(self.quat.shape)) + \
                    ". linvel" + (" is None" if self.linvel is None else ".shape = " + str(self.linvel.shape)) + \
                    ". angvel" + (" is None" if self.angvel is None else ".shape = " + str(self.angvel.shape))
                return res

            def set_value(self, pos: Optional[np.ndarray] = None, quat: Optional[np.ndarray] = None,
                        rot_mat: Optional[np.ndarray] = None,
                        linvel: Optional[np.ndarray] = None, angvel: Optional[np.ndarray] = None,
                        linacc: Optional[np.ndarray] = None, angacc: Optional[np.ndarray] = None):
                self.pos = pos
                self.quat = quat
                self.rot_mat = rot_mat
                self.linvel = linvel
                self.angvel = angvel
                self.linacc = linacc
                self.angacc = angacc

        class TargetPose:

            __slots__ = ("globally", "locally", "child_body", "root", "root_body", "facing_root", "end", "facing_coor_end",
                        "all_joint_global", "all_joint_local", "all_joint_facing", "all_child_body", "character_body", "facing_quat",
                        "num_frames", "fps", "smoothed", "dup_pos_off_mix", "dup_rot_off_mix",
                        "dup_root_pos", "dup_root_quat")

            def __init__(self):
                # joint info in global coordinate
                # component shape is (num frame, num joint, 3)
                self.globally = ODESim.TargetPose.TargetBaseType()

                # joint info in parent local coordinate
                # component shape is (num frame, character num joint, 3)
                self.locally = ODESim.TargetPose.TargetBaseType()

                # child body's position in global coordinate
                # component shape is (num frame, num child body, 3)
                self.child_body = ODESim.TargetPose.TargetBaseType()

                # root info in global coordinate
                # component shape is (num frame, 3)
                self.root = ODESim.TargetPose.TargetBaseType()

                self.root_body = ODESim.TargetPose.TargetBaseType()

                # root info in facing coordinate
                # component shape is (num frame, 3)
                self.facing_root = ODESim.TargetPose.TargetBaseType()

                # end info in global coordinate
                # component shape is (num frame, num joint, 3)
                self.end = ODESim.TargetPose.TargetBaseType()

                # end site in y rotation (heading) coordinate
                # component shape is (num frame, num joint, 3)
                self.facing_coor_end = ODESim.TargetPose.TargetBaseType()

                # joint global info with root joint
                # component shape is (num frame, num body, 3)
                self.all_joint_global = ODESim.TargetPose.TargetBaseType()

                # joint local info with root joint
                self.all_joint_local = ODESim.TargetPose.TargetBaseType()

                # joint facing info with root joint
                self.all_joint_facing = ODESim.TargetPose.TargetBaseType()

                # all body global info
                # component shape is (num frame, num body, 3)
                # note: body order may be different from ode bodies...
                self.all_child_body = ODESim.TargetPose.TargetBaseType()

                # all body global info, body order matches ode order..
                self.character_body = ODESim.TargetPose.TargetBaseType()

                # shape = (num frame, 4)
                self.facing_quat: Optional[np.ndarray] = None

                self.num_frames: int = 0
                self.fps: int = 0

                self.smoothed: bool = False
                self.dup_pos_off_mix: Optional[np.ndarray] = None  # delta position from (frame - 1) to (frame) for motion duplicate
                self.dup_rot_off_mix: Union[np.ndarray, Rotation, None] = None  # delta quaternion from (frame - 1) to (frame) for motion duplicate
                self.dup_root_pos: Optional[np.ndarray] = None
                self.dup_root_quat: Optional[np.ndarray] = None

            def set_dup_off_mix(self, pos_off_mix: np.ndarray, rot_off_mix: Union[np.ndarray, Rotation]):
                self.dup_pos_off_mix = pos_off_mix
                self.dup_rot_off_mix = rot_off_mix
                self.dup_root_pos = self.root.pos.copy()
                self.dup_root_quat = self.root.quat.copy()

            def compute_global_root_dup(self, dup_count: int):
                if dup_count > 1:
                    assert self.dup_pos_off_mix is not None and self.dup_rot_off_mix is not None
                    self.compute_global_root_dup_impl(dup_count, self.dup_pos_off_mix, self.dup_rot_off_mix)
                elif dup_count == 1:
                    self.dup_root_pos = self.root.pos.copy()
                    self.dup_root_quat = self.root.quat.copy()

            def compute_global_root_dup_impl(self, dup_count: int, pos_off_mix: Optional[np.ndarray], rot_off_mix: Union[np.ndarray, Rotation, None]):
                if dup_count <= 1:
                    return

                dt = 1.0 / self.fps
                if pos_off_mix is None:
                    pos_off_mix = 0.5 * dt * (self.root.linvel[0] + self.root.linvel[-1])
                if rot_off_mix is None: # calc by omega
                    omega_ = self.root.angvel[-1].copy()
                    last_q_ = self.root.quat[-1].copy()
                    end_q_ = Common.MathHelper.quat_integrate(last_q_[None, :], omega_[None, :], dt)
                    rot_off_mix: Rotation = (Rotation(last_q_).inv() * Rotation(end_q_))
                if isinstance(rot_off_mix, np.ndarray):
                    rot_off_mix: Rotation = Rotation(rot_off_mix)

                self.dup_root_pos = np.zeros((dup_count, self.num_frames, 3), dtype=np.float64)
                self.dup_root_quat = Common.MathHelper.unit_quat_arr((dup_count, self.num_frames, 4))

                self.dup_root_pos[0, :, :] = self.root.pos.copy()
                self.dup_root_quat[0, :, :] = self.root.quat.copy()
                frame_0_rot = Rotation(self.root.quat[0])
                frame_0_rot_inv = frame_0_rot.inv()
                frame_0_coor_rot: Rotation = frame_0_rot_inv * Rotation(self.root.quat)
                frame_0_coor_pos: np.ndarray = frame_0_rot_inv.apply(self.root.pos - self.root.pos[None, 0])

                pos_off_mix = Rotation(self.root.quat[-1]).inv().apply(pos_off_mix)
                for i in range(1, dup_count):
                    end_rot = Rotation(self.dup_root_quat[i - 1, -1])
                    next_rot_0 = rot_off_mix * end_rot
                    self.dup_root_quat[i, :, :] = (next_rot_0 * frame_0_coor_rot).as_quat()
                    next_pos_0 = self.dup_root_pos[i - 1, -1] + end_rot.apply(pos_off_mix)
                    self.dup_root_pos[i, :, :] = next_pos_0.reshape((1, 3)) + next_rot_0.apply(frame_0_coor_pos)
                    self.dup_root_pos[i, :, 1] = self.root.pos[:, 1].copy()

                def debug_func():
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    pos_ = self.dup_root_pos
                    for i in range(dup_count):
                        ax.plot(pos_[i, :, 0], pos_[i, :, 1], pos_[i, :, 2])
                    # pos_ = self.dup_root_pos.reshape((-1, 3))
                    # ax.plot(pos_[:, 0], pos_[:, 1], pos_[:, 2])
                    ax.set_xlim(-3, 3)
                    ax.set_ylim(0, 2)
                    ax.set_zlim(-3, 3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                    plt.show()
                    exit(0)

                # debug_func()
                self.dup_root_pos = np.concatenate([self.dup_root_pos.reshape((-1, 3)), self.root.pos[None, 0]], axis=0)
                self.dup_root_quat = np.concatenate([self.dup_root_quat.reshape((-1, 4)), self.root.quat[None, 0]], axis=0)

            def append(self, other):
                if self.globally is None:
                    self.globally = copy.deepcopy(other.globally)
                else:
                    self.globally.append(other.globally)

                if self.locally is None:
                    self.locally = copy.deepcopy(other.locally)
                else:
                    self.locally.append(other.locally)

                if self.child_body is None:
                    self.child_body = copy.deepcopy(other.child_body)
                else:
                    self.child_body.append(other.child_body)

                if self.root is None:
                    self.root = copy.deepcopy(other.root)
                else:
                    self.root.append(other.root)

                if self.root_body is None:
                    self.root_body = copy.deepcopy(other.root_body)
                else:
                    self.root_body.append(other.root_body)

                if self.facing_root is None:
                    self.facing_root = copy.deepcopy(other.facing_root)
                else:
                    self.facing_root.append(other.facing_root)

                if self.end is None:
                    self.end = copy.deepcopy(other.end)
                else:
                    self.end.append(other.end)

                if self.facing_coor_end is None:
                    self.facing_coor_end = copy.deepcopy(other.facing_coor_end)
                else:
                    self.facing_coor_end.append(other.facing_coor_end)

                if self.all_joint_global is None:
                    self.all_joint_global = copy.deepcopy(other.all_joint_global)
                else:
                    self.all_joint_global.append(other.all_joint_global)

                if self.all_joint_local is None:
                    self.all_joint_local = copy.deepcopy(other.all_joint_local)
                else:
                    self.all_joint_local.append(other.all_joint_local)

                if self.all_joint_facing is not None:
                    self.all_joint_facing = copy.deepcopy(other.all_joint_facing)
                else:
                    self.all_joint_facing.append(other.all_joint_facing)

                if self.all_child_body is None:
                    self.all_child_body = copy.deepcopy(other.all_child_body)
                else:
                    self.all_child_body.append(other.all_child_body)

                if self.character_body is None:
                    self.character_body = copy.deepcopy(other.character_body)
                else:
                    self.character_body.append(other.character_body)

                if self.facing_quat is None:
                    self.facing_quat: Optional[np.ndarray] = copy.deepcopy(other.facing_quat)
                else:
                    self.facing_quat = np.concatenate([self.facing_quat, other.facing_quat], axis=0)

                self.num_frames += other.num_frames
                return self

            def duplicate(self, times: int = 0):
                res = ODESim.TargetPose.TargetPose()
                if self.globally is not None:
                    res.globally = self.globally.duplicate(times)
                if self.locally is not None:
                    res.locally = self.locally.duplicate(times)
                if self.child_body is not None:
                    res.child_body = self.child_body.duplicate(times)
                if self.root is not None:
                    res.root = self.root.duplicate(times)
                if self.root_body is not None:
                    res.root_body = self.root_body.duplicate(times)
                if self.facing_root is not None:
                    res.facing_root = self.facing_root.duplicate(times)
                if self.end is not None:
                    res.end = self.end.duplicate(times)
                if self.facing_coor_end is not None:
                    res.facing_coor_end = self.facing_coor_end.duplicate(times)
                if self.all_joint_global is not None:
                    res.all_joint_global = self.all_joint_global.duplicate(times)
                if self.all_joint_local is not None:
                    res.all_joint_local = self.all_joint_local.duplicate(times)
                if self.all_joint_facing is not None:
                    res.all_joint_facing = self.all_joint_facing.duplicate(times)
                if self.all_child_body is not None:
                    res.all_child_body = self.all_child_body.duplicate(times)
                if self.character_body is not None:
                    res.character_body = self.character_body.duplicate(times)
                if self.facing_quat is not None:
                    res.facing_quat = np.concatenate([self.facing_quat] * times, axis=0)
                res.num_frames = self.num_frames * times
                res.fps = self.fps
                res.smoothed = self.smoothed

                return res

            def __len__(self) -> int:
                return max([len(i) for i in [self.globally, self.locally, self.root, []] if i is not None])

            def deepcopy(self):
                return copy.deepcopy(self)

            def to_continuous(self):
                if self.globally is not None:
                    self.globally.to_continuous()

                if self.locally is not None:
                    self.locally.to_continuous()

                if self.child_body is not None:
                    self.child_body.to_continuous()

                if self.root is not None:
                    self.root.to_continuous()

                if self.root_body is not None:
                    self.root_body.to_continuous()

                if self.facing_root is not None:
                    self.facing_root.to_continuous()

                if self.end is not None:
                    self.end.to_continuous()

                if self.facing_coor_end is not None:
                    self.facing_coor_end.to_continuous()

                if self.all_joint_global is not None:
                    self.all_joint_global.to_continuous()

                if self.all_joint_local is not None:
                    self.all_joint_local.to_continuous()

                if self.all_joint_facing is not None:
                    self.all_joint_facing.to_continuous()

                if self.all_child_body is not None:
                    self.all_child_body.to_continuous()

                if self.character_body is not None:
                    self.character_body.to_continuous()

                self.facing_quat = np.ascontiguousarray(self.facing_quat)

                return self

            def sub_seq(self, start: Optional[int] = None, end_: Optional[int] = None, skip: Optional[int] = None, is_copy: bool = True):
                """
                Get sub sequence of TargetPose
                """
                assert skip is None or isinstance(skip, int)
                res = ODESim.TargetPose.TargetPose()
                if start is None:
                    start = 0
                if end_ is None:
                    end_ = self.num_frames
                if end_ == 0:
                    return res

                piece = slice(start, end_, skip)
                if self.globally is not None:
                    res.globally = self.globally.sub_seq(start, end_, skip, is_copy)

                if self.locally is not None:
                    res.locally = self.locally.sub_seq(start, end_, skip, is_copy)

                if self.child_body is not None:
                    res.child_body = self.child_body.sub_seq(start, end_, skip, is_copy)

                if self.root is not None:
                    res.root = self.root.sub_seq(start, end_, skip, is_copy)

                if self.root_body is not None:
                    res.root_body = self.root_body.sub_seq(start, end_, skip, is_copy)

                if self.end is not None:
                    res.end = self.end.sub_seq(start, end_, skip, is_copy)

                if self.facing_coor_end is not None:
                    res.facing_coor_end = self.facing_coor_end.sub_seq(start, end_, skip, is_copy)

                if self.facing_root is not None:
                    res.facing_root = self.facing_root.sub_seq(start, end_, skip, is_copy)

                if self.all_joint_global is not None:
                    res.all_joint_global = self.all_joint_global.sub_seq(start, end_, skip, is_copy)

                if self.all_joint_local is not None:
                    res.all_joint_local = self.all_joint_local.sub_seq(start, end_, skip, is_copy)

                if self.all_joint_facing is not None:
                    res.all_joint_facing = self.all_joint_facing.sub_seq(start, end_, skip, is_copy)

                if self.all_child_body is not None:
                    res.all_child_body = self.all_child_body.sub_seq(start, end_, skip, is_copy)

                if self.character_body is not None:
                    res.character_body = self.character_body.sub_seq(start, end_, skip, is_copy)

                if self.facing_quat is not None:
                    res.facing_quat = self.facing_quat[piece].copy() if is_copy else self.facing_quat[piece]

                res.num_frames = len(res)
                res.fps = self.fps
                res.smoothed = self.smoothed

                return res

            def get_facing_body_info(self):
                result = ODESim.TargetPose.TargetBaseType()
                root_pos: np.ndarray = Common.MathHelper.vec_axis_to_zero(self.root.pos, 1)
                ry, _ = Common.MathHelper.y_decompose(self.root.quat)
                ry_inv = Rotation(ry).inv()
                result.pos = self.character_body.pos - root_pos[:, None, :]
                for body_index in range(1, result.pos.shape[1]):
                    result.pos[:, body_index] = ry_inv.apply(result.pos[:, body_index])

                result.quat = self.character_body.quat.copy()
                for body_index in range(1, result.quat.shape[1]):
                    result.quat[:, body_index] = None

                result.linvel = None
                result.angvel = None
                return result

        class SetTargetToCharacter:
            """
            use for load {frame} to ODE Character
            """
            def __init__(self, character, target):
                self.character = character
                self.target = target

            @property
            def body_info(self):
                """
                get body info
                """
                return self.character.body_info

            @property
            def joint_info(self):
                """
                get joint info
                """
                return self.character.joint_info

            def joint_names(self) -> List[str]:
                """
                get joint names
                """
                return self.joint_info.joint_names()

            def body_names(self) -> List[str]:
                """
                get body names
                """
                return self.body_info.get_name_list()

            @property
            def end_joint(self):
                return self.character.end_joint

            @property
            def world(self) -> ode.World:
                return self.character.world

            @property
            def space(self) -> ode.SpaceBase:
                return self.character.space

            @property
            def bodies(self) -> List[ode.Body]:
                return self.character.bodies

            @property
            def joints(self) -> List[Union[ode.Joint, ode.BallJoint, ode.BallJointAmotor, ode.HingeJoint]]:
                return self.character.joints

            @property
            def root_joint(self) -> Optional[ode.Joint]:
                return self.character.root_joint

            @property
            def joint_to_child_body(self) -> List[int]:
                return self.character.joint_to_child_body

            @property
            def child_body_to_joint(self) -> List[int]:
                return self.character.child_body_to_joint

            @property
            def joint_to_parent_body(self) -> List[int]:
                return self.character.joint_to_parent_body

            @property
            def has_end_joint(self) -> bool:
                return self.character.has_end_joint

            @property
            def num_frames(self):
                return self.target.num_frames

            def set_character_byframe(self, frame: int = 0, other_character=None):
                if other_character is None:
                    other_character = self.character
                c_id = other_character.body_info.body_c_id
                ch_body = self.target.character_body
                other_character.world.loadBodyPos(c_id, ch_body.pos[frame].flatten())
                other_character.world.loadBodyQuat(c_id, ch_body.quat[frame].flatten())
                other_character.world.loadBodyLinVel(c_id, ch_body.linvel[frame].flatten())
                other_character.world.loadBodyAngVel(c_id, ch_body.angvel[frame].flatten())

                # state = other_character.save()
                # other_character.load(state)
                # return state

            def set_character_byframe_old(self, frame: int = 0, other_character=None):
                """
                load {frame} to ODE Character
                we don't need to resort joint, because we have joint c id..
                """
                if other_character is None:
                    other_character = self.character

                # Set Root Body's Position, Rotation, Linear Velocity, Angular Velocity
                other_character.root_body.PositionNumpy = self.target.root.pos[frame]
                other_character.root_body.setQuaternionScipy(self.target.root.quat[frame])  # rot
                other_character.root_body.LinearVelNumpy = self.target.root.linvel[frame]
                other_character.root_body.setAngularVelNumpy(self.target.root.angvel[frame])

                # Set global position and quaternion via child_body's c id
                other_character.world.loadBodyPos(other_character.joint_info.child_body_c_id,
                                                self.target.child_body.pos[frame].flatten())
                other_character.world.loadBodyQuat(other_character.joint_info.child_body_c_id,
                                                self.target.child_body.quat[frame].flatten())

                # Set child_body's linear velocity
                other_character.world.loadBodyLinVel(other_character.joint_info.child_body_c_id,
                                                    self.target.child_body.linvel[frame].flatten())

                # Set child_body's angular velocity
                other_character.world.loadBodyAngVel(other_character.joint_info.child_body_c_id,
                                                    self.target.child_body.angvel[frame].flatten())

                state = other_character.save()
                other_character.load(state)
                return state
                # self.check(frame)

            def check(self, frame: int):
                # check root body
                assert np.all(self.character.root_body.PositionNumpy - self.target.root.pos[frame] == 0)
                assert np.all(np.abs(self.character.root_body.getQuaternionScipy() - self.target.root.quat[frame]) < 1e-10)
                assert np.all(self.character.root_body.LinearVelNumpy - self.target.root.linvel[frame] == 0)
                assert np.all(self.character.root_body.getAngularVelNumpy() - self.target.root.angvel[frame] == 0)

                # check body pos
                assert np.all(self.world.getBodyPos(self.body_info.body_c_id[1:]).reshape((-1, 3)) -
                            self.target.child_body.pos[frame] == 0)
                # check body linear velocity
                assert np.all(self.world.getBodyLinVel(self.body_info.body_c_id[1:]).reshape((-1, 3)) -
                            self.target.child_body.linvel[frame] == 0)
                # check body quat
                assert np.all(np.abs(self.world.getBodyQuatScipy(self.body_info.body_c_id[1:]).reshape((-1, 4)) -
                                    self.target.child_body.quat[frame]) < 1e-10)

                # check body angvel
                assert np.all(np.abs(self.world.getBodyAngVel(self.body_info.body_c_id[1:]).reshape((-1, 3)) -
                                    self.target.child_body.angvel[frame]) < 1e-10)
                # check global angular velocity..

                angvel = np.linalg.norm(self.character.joint_info.get_local_angvels(), axis=1)
                angvel_real = np.linalg.norm(self.target.locally.angvel[frame], axis=1)

                print(np.max(self.character.joint_info.get_local_angvels() - self.target.locally.angvel[frame]))

    class ODEScene:
        class SceneContactLocalInfo:
            """

            """
            def __init__(self) -> None:
                self.global_pos: Optional[np.ndarray] = None
                self.local_pos: Optional[np.ndarray] = None
                self.normal: Optional[np.ndarray] = None
                self.depth: Optional[np.ndarray] = None
                self.body1_cid: Optional[np.ndarray] = None

            def set_value(
                    self,
                    global_pos: np.ndarray,
                    local_pos: np.ndarray,
                    normal: np.ndarray,
                    depth: np.ndarray,
                    body1_cid: np.ndarray,
                    to_continuous: bool = True
            ):
                self.global_pos = np.ascontiguousarray(global_pos) if to_continuous else global_pos
                self.local_pos = np.ascontiguousarray(local_pos) if to_continuous else local_pos
                self.normal = np.ascontiguousarray(normal) if to_continuous else normal
                self.depth = np.ascontiguousarray(depth) if to_continuous else normal
                self.body1_cid = np.ascontiguousarray(body1_cid) if to_continuous else normal

            def get_global_pos(self, world: ode.World) -> np.ndarray:
                if self.global_pos is None:
                    body1_pos: np.ndarray = world.getBodyPos(self.body1_cid)
                    body1_quat: np.ndarray = world.getBodyQuatScipy(self.body1_cid)
                    self.global_pos: Optional[np.ndarray] = Rotation(body1_quat).apply(self.local_pos) + body1_pos
                    
                return self.global_pos

            def clear(self):
                self.global_pos: Optional[np.ndarray] = None
                self.local_pos: Optional[np.ndarray] = None
                self.normal: Optional[np.ndarray] = None
                self.depth: Optional[np.ndarray] = None
                self.body1_cid: Optional[np.ndarray] = None

                return self


        class SceneContactInfo:
            """
            Contact Info Extractor, for visualize in Unity..
            """
            __slots__ = (
                "pos",
                "force",
                "torque",
                "geom1_name",
                "geom2_name",
                "body1_index",
                "body2_index",
                "contact_label",
                "body_contact_force"
            )

            def __init__(
                self,
                pos: Union[np.ndarray, List, None] = None,  # contact position (in global coordinate)
                force: Union[np.ndarray, List, None] = None,  # contact force (in global coordinate)
                geom1_name: Optional[List[str]] = None,  # name of geometry 1
                geom2_name: Optional[List[str]] = None,  # name of geometry 2
                contact_label: Union[np.ndarray, List[float], None] = None,  #
                body_contact_force: Union[np.ndarray, None] = None # sum of contact force on each body
            ):
                self.pos: Union[np.ndarray, List, None] = pos
                self.force: Union[np.ndarray, List, None] = force
                self.torque: Union[np.ndarray, List, None] = None
                # note: the contact torque is unused, because rendering the contact torque is hard..

                self.geom1_name: Optional[List[str]] = geom1_name
                self.geom2_name: Optional[List[str]] = geom2_name

                self.body1_index: Optional[List[int]] = None
                self.body2_index: Optional[List[int]] = None

                self.contact_label: Union[np.ndarray, List[float], None] = contact_label
                self.body_contact_force: Union[np.ndarray, None] = body_contact_force

            def __len__(self) -> int:
                if self.pos is None:
                    return 0
                else:
                    return len(self.pos)

            def merge_force_by_body1(self):
                """
                merge the total force by body1 index..
                There is only one character in the scene
                """
                # 1. divide the contact into several groups
                body1_index: np.ndarray = np.asarray(self.body1_index)
                unique_body1: np.ndarray = np.unique(body1_index)
                forces: np.ndarray = np.asarray(self.force)
                if self.torque is not None:
                    torques: Optional[np.ndarray] = np.asarray(self.torque)
                else:
                    torques: Optional[np.ndarray] = None

                ret_force = self.force.copy()
                for sub_body in unique_body1:
                    try:
                        sub_contact = np.where(body1_index == sub_body)[0]
                        # divide the sub force..
                        sub_force = forces[sub_contact]
                        sub_force_len: np.ndarray = np.linalg.norm(sub_force, axis=-1)
                        sub_force_len = (sub_force_len / np.sum(sub_force_len)).reshape((-1, 1))  # in shape (sum contact, 1)
                        sub_force_avg: np.ndarray = np.mean(sub_force, axis=0).reshape((1, 3))  # in shape (1, 3,)
                        divide_force: np.ndarray = sub_force_len * sub_force_avg
                        ret_force[sub_contact] = divide_force

                        # divide the sub torque..
                        if torques is not None and False:
                            sub_torque: np.ndarray = torques[sub_contact]
                            sub_torque_len: np.ndarray = np.linalg.norm(sub_torque, axis=-1)
                            sub_torque_len: np.ndarray = (sub_torque_len / np.sum(sub_torque_len)).reshape((-1, 1))
                            sub_torque_avg: np.ndarray = np.mean(sub_torque, axis=0).reshape((1, 3))
                            divide_torque: np.ndarray = sub_torque_len * sub_torque_avg
                            self.torque: np.ndarray = divide_torque
                    except Exception as err:
                        raise err

                self.force = ret_force
                return self

            def set_value(
                self,
                pos: Optional[np.ndarray],
                force: Optional[np.ndarray],
                geom1_name: Optional[List[str]],
                geom2_name: Optional[List[str]],
                contact_label: Union[np.ndarray, List[float], None] = None,
                body_contact_force: Union[np.ndarray, None] = None
            ):
                self.pos: Optional[np.ndarray] = pos
                self.force: Optional[np.ndarray] = force
                self.geom1_name = geom1_name
                self.geom2_name = geom2_name
                self.contact_label = contact_label
                self.body_contact_force = body_contact_force

            def clear(self):
                self.pos = None
                self.force = None
                if self.geom1_name is not None:
                    del self.geom1_name[:]
                if self.geom2_name is not None:
                    del self.geom2_name[:]
                self.contact_label = None
                self.body_contact_force = None

            def check_delta(self, other):
                if self.pos is None and other.pos is None:
                    return True
                if len(self) != len(other):
                    print(f"Contact count not match. self is {len(self)}, other is {len(other)}")
                    return False
                res = np.all(self.pos - other.pos == 0) and np.all(self.force - other.force == 0)
                if res is False:
                    print(f"self.pos {self.pos}, other.pos {other.pos}")
                    print(f"self.force {self.force}, other.force {other.force}")
                return res

            def out_iter(self):
                """
                get 
                """
                if self.contact_label is None:
                    self.contact_label = np.ones(len(self))
                if isinstance(self.contact_label, np.ndarray):
                    self.contact_label = self.contact_label.tolist()
                try:
                    return zip(range(len(self)), self.pos, self.force, self.contact_label) if len(self) > 0 else zip((), (), (), ())
                except Exception as err:
                    print(self.pos, self.force, self.contact_label)
                    raise err


        # TODO: add some other type, such as lemke, dart, and etc.
        """
        ODE_LCP: use normal LCP model for contact
        MAX_FORCE_ODE_LCP: for contact, f <= F_max (a constant value), rather than f <= u F_n
        BALL: use ball joint for contact
        """
        class ContactType(IntEnum):
            ODE_LCP = 0
            MAX_FORCE_ODE_LCP = 1
            BALL = 2  # NOTE: Ball Contact doesn't work at all!


        """
        DAMPED_STEP / DAMPED_FAST_STEP: use stable PD control & damping in forward simulation (larger simulation step is required)
        STEP / FAST_STEP: use PD control in forward simulation (smaller simulation step is required)
        """
        class SimulationType(IntEnum):
            DAMPED_STEP = 0
            DAMPED_FAST_STEP = 1
            STEP = 2
            FAST_STEP = 3


        class ODEScene:
            default_gravity: List[float] = [0.0, -9.8, 0.0]

            def __init__(
                self,
                render_fps: int = 60,
                sim_fps: int = 120,
                gravity: Union[Iterable, float, None] = None,
                friction: float = 0.8,
                bounce: float = 0.0,
                self_collision: bool = True,
                contact_type = None,
                contact_count: int = 4,
                extract_contact: bool = True,
                hand_scene: bool = False
            ):
                if contact_type is None:
                    contact_type = ODESim.ODEScene.ContactType.ODE_LCP
                self.render_dt, self.render_fps = 0.0, 0  # Frame Time in Unity
                self.sim_dt, self.sim_fps = 0.0, 0.0  # Simulate Step in ODE
                self.step_cnt: int = 1
                self.set_render_fps(render_fps)
                self.set_sim_fps(sim_fps)
                self.bounce, self.friction = bounce, friction

                self.world: Optional[ode.World] = None  # use for forward simulation
                self.space: Optional[ode.SpaceBase] = None  # use for collision detection

                self._self_collision: bool = self_collision  # enable self collision for characters
                self.contact_type = contact_type  #

                self._use_soft_contact: bool = False
                self._soft_cfm: Optional[float] = None
                self.soft_cfm_tan: Optional[float] = None

                self._soft_erp: Optional[float] = None

                self.contact: ode.JointGroup = ode.JointGroup()  # contact joints used in contact detection
                self.characters = []  # store all of characters
                self.environment = ODESim.Environment.Environment(self.world, self.space)

                self.ext_joints = None
                self._contact_count: int = contact_count
                self.contact_info = None  # for unity render
                self.contact_local_info = None  # for hack in DiffODE

                self.extract_contact: bool = extract_contact

                # in zyl's hand scene mode
                self.hand_scene: bool = hand_scene
                self.r_character_id = None
                self.l_character_id = None
                self.obj_character_id = None

                self.str_info: str = ""  # text information for Unity client
                self.clear(gravity)

                self.simu_func = self.damped_simulate_once
                self.simu_type = ODESim.ODEScene.SimulationType.DAMPED_STEP

            @property
            def soft_erp(self):
                """
                erp value for soft contact
                """
                return self._soft_erp

            @soft_erp.setter
            def soft_erp(self, value: float):
                self._soft_erp = value
                self.world.soft_erp = value

            @property
            def soft_cfm(self):
                """
                cfm value for soft contact
                """
                return self._soft_cfm

            @soft_cfm.setter
            def soft_cfm(self, value: float):
                self._soft_cfm = value
                self.world.soft_cfm = value

            @property
            def use_soft_contact(self):
                return self._use_soft_contact

            @use_soft_contact.setter
            def use_soft_contact(self, value: bool):
                self._use_soft_contact = value
                self.world.use_soft_contact = value

            def set_simulation_type(self, sim_type):
                if self.simu_type == sim_type:
                    return sim_type
                if sim_type == ODESim.ODEScene.SimulationType.DAMPED_STEP:
                    self.simu_func = self.damped_simulate_once
                    self.disable_implicit_damping()
                elif sim_type ==  ODESim.ODEScene.SimulationType.DAMPED_FAST_STEP:
                    self.simu_func = self.fast_simulate_once
                    self.use_implicit_damping()
                elif sim_type ==  ODESim.ODEScene.SimulationType.STEP:
                    self.simu_func = self.simulate_once
                    self.disable_implicit_damping()
                elif sim_type ==  ODESim.ODEScene.SimulationType.FAST_STEP:
                    self.simu_func = self.fast_simulate_once
                    self.disable_implicit_damping()
                else:
                    raise NotImplementedError

                self.simu_type = sim_type
                print(f"set sim type to {self.simu_type}")
                return self.simu_type

            def step_range(self) -> range:
                return range(self.step_cnt)

            def use_implicit_damping(self):
                for character in self.characters:
                    for joint in character.joint_info.joints:
                        joint.enable_implicit_damping()

            def disable_implicit_damping(self):
                for character in self.characters:
                    for joint in character.joint_info.joints:
                        joint.disable_implicit_damping()

            def set_gravity(self, gravity: Union[Iterable, float, None] = None):
                if gravity is None:
                    g = self.default_gravity
                elif isinstance(gravity, float):
                    g = [0, gravity, 0]
                elif isinstance(gravity, Iterable):
                    g = list(gravity)
                else:
                    raise NotImplementedError

                self.world.setGravity(g)

            @property
            def gravity_numpy(self) -> np.ndarray:
                """
                Get the gravity. default gravity is [0, -9.8, 0]
                """
                return self.world.getGravityNumpy()

            @property
            def self_collision(self) -> bool:
                """
                consider self collision detection is enabled for each character
                """
                return self._self_collision

            @self_collision.setter
            def self_collision(self, value: bool):
                if self._self_collision == value:
                    return
                for character in self.characters:
                    character.self_collision = value
                self._self_collision = value
                self.world.self_collision = value

            def build_world_and_space(self, gravity: Union[Iterable, float, None] = None):
                self.world = ode.World()
                self.set_gravity(gravity)
                # self.space = ode.Space()  # simple space. using AABB for collision detection.
                self.space = ode.HashSpace()
                return self.world, self.space

            @property
            def floor(self) -> Optional[ode.GeomPlane]:  # TODO: add floor id in unity
                return self.environment.floor

            # Get the first character in the scene
            @property
            def character0(self):
                return self.characters[0] if self.characters else None

            def get_character_id_map(self):
                return {character.character_id: character for character in self.characters}

            def set_render_fps(self, render_fps: int):
                self.render_dt = 1.0 / render_fps  # Frame Time in Unity
                self.render_fps = render_fps
                self.step_cnt = self.sim_fps // self.render_fps if self.render_fps > 0 else 1

            def set_sim_fps(self, sim_fps: int):
                self.sim_dt = 1.0 / sim_fps
                self.sim_fps = sim_fps
                self.step_cnt = self.sim_fps // self.render_fps if self.render_fps > 0 else 1

            def create_floor(self) -> ode.GeomPlane:
                """
                Create floor geometry
                """
                return self.environment.create_floor()

            def reset(self):
                """
                reset each character to initial state
                """
                for character in self.characters:
                    character.load_init_state()
                return self

            @staticmethod
            def set_falldown_flag(geom1: ode.GeomObject, geom2: ode.GeomObject):
                if geom1.is_environment and not geom2.clung_env:
                    geom2.character.fall_down = True
                if geom2.is_environment and not geom1.clung_env:
                    geom1.character.fall_down = True

            @property
            def contact_count(self) -> int:  # TODO: support contact count in Unity, load contact param in json file
                return self._contact_count  # in ode.dCollide

            @contact_count.setter
            def contact_count(self, value: int):
                self._contact_count = value
                self.world.max_contact_num = value

            def contact_save(self):
                """
                save contact position, force.
                render in Unity.
                """
                if len(self.contact) > 0:
                    pos = np.zeros((len(self.contact), 3), dtype=np.float64)
                    force = np.zeros((len(self.contact), 3), dtype=np.float64)
                    geom1_name, geom2_name = [], []
                    if self.hand_scene is False:
                        for idx, contact_joint in enumerate(self.contact.joints):
                            joint: ode.ContactJoint = contact_joint
                            contact: ode.Contact = joint.contact
                            pos[idx, :] = contact.contactPosNumpy  # contact position
                            force[idx, :] = joint.FeedBackForce()  # contact force
                            geom1_name.append(joint.contactGeom1.name)
                            geom2_name.append(joint.contactGeom2.name)
                        if self.contact_info is None:
                            self.contact_info = ODESim.ODEScene.SceneContactInfo(pos, force, geom1_name, geom2_name)
                        else:
                            self.contact_info.set_value(pos, force, geom1_name, geom2_name)
                    else:
                        body_contact_force_r = np.zeros((len(self.characters[0].bodies), 3), dtype=np.float64)
                        body_contact_force_l = np.zeros((len(self.characters[1].bodies), 3), dtype=np.float64)
                        body_contact_label_r = np.zeros(len(self.characters[0].bodies), dtype=np.bool8)
                        body_contact_label_l = np.zeros(len(self.characters[1].bodies), dtype=np.bool8)
                        for idx, contact_joint in enumerate(self.contact.joints):
                            joint: ode.ContactJoint = contact_joint
                            contact: ode.Contact = joint.contact
                            pos[idx, :] = contact.contactPosNumpy  # contact position
                            force[idx, :] = joint.FeedBackForce()  # contact force
                            geom1_name.append(joint.contactGeom1.name)
                            geom2_name.append(joint.contactGeom2.name)
                            body1: ode.Body = joint.contactGeom1.body
                            body2: ode.Body = joint.contactGeom2.body
                            if body1 is None or body2 is None:
                                continue
                            if joint.contactGeom1.character_id == self.r_character_id and joint.contactGeom2.character_id == self.obj_character_id:
                                body_contact_label_r[body1.instance_id] = True
                                body_contact_force_r[body1.instance_id] += force[idx]
                            elif joint.contactGeom1.character_id == self.l_character_id and joint.contactGeom2.character_id == self.obj_character_id:
                                body_contact_label_l[body1.instance_id] = True
                                body_contact_force_l[body1.instance_id] += force[idx]
                            elif joint.contactGeom2.character_id == self.r_character_id and joint.contactGeom1.character_id == self.obj_character_id:
                                body_contact_label_r[body2.instance_id] = True
                                body_contact_force_r[body2.instance_id] += force[idx]
                            elif joint.contactGeom2.character_id == self.l_character_id and joint.contactGeom1.character_id == self.obj_character_id:
                                body_contact_label_l[body2.instance_id] = True
                                body_contact_force_l[body2.instance_id] += force[idx]
                        body_contact_label = np.concatenate((body_contact_label_r, body_contact_label_l), axis=0)
                        body_contact_force = np.concatenate((body_contact_force_r, body_contact_force_l), axis=0)
                        if self.contact_info is None:
                            self.contact_info = ODESim.ODEScene.SceneContactInfo(pos, force, geom1_name, geom2_name, contact_label=body_contact_label, body_contact_force=body_contact_force)
                        else:
                            self.contact_info.set_value(pos, force, geom1_name, geom2_name, contact_label=body_contact_label, body_contact_force=body_contact_force)
                else:
                    if self.contact_info is not None:
                        self.contact_info.clear()

                return self.contact_info

            def contact_local_save(self):
                """
                we need only save position in body 1 coordinate
                we can remain global normal vector
                """
                if self.contact_local_info is not None:
                    self.contact_local_info.clear()

                len_contact: int = len(self.contact)
                if len(self.contact) > 0:
                    global_pos_ret: np.ndarray = np.zeros((len_contact, 3), dtype=np.float64)
                    local_pos_ret: np.ndarray = np.zeros((len_contact, 3), dtype=np.float64)
                    normal_ret: np.ndarray = np.zeros_like(local_pos_ret)
                    depth_ret: np.ndarray = np.zeros(len_contact)
                    body1_cid_ret: np.ndarray = np.zeros(len_contact, dtype=np.uint64)
                    for index, contact_joint in enumerate(self.contact.joints):
                        joint: ode.ContactJoint = contact_joint
                        contact: ode.Contact = joint.contact
                        contact_pos: np.ndarray = contact.contactPosNumpy
                        body1: ode.Body = joint.body1
                        assert body1 is not None
                        global_pos_ret[index] = contact_pos
                        contact_local_pos: np.ndarray = body1.getPosRelPointNumpy(contact_pos)
                        local_pos_ret[index] = contact_local_pos
                        normal: np.ndarray = contact.contactNormalNumpy
                        normal_ret[index] = normal
                        depth_ret[index] = contact.contactDepth
                        body1_cid_ret[index] = body1.get_bid()
                    if self.contact_local_info is None:
                        self.contact_local_info = ODESim.ODEScene.SceneContactLocalInfo()
                    self.contact_local_info.set_value(global_pos_ret, local_pos_ret, normal_ret, depth_ret, body1_cid_ret)

                return self.contact_local_info

            def contact_basic(self, geom1: ode.GeomObject, geom2: ode.GeomObject) -> Optional[List[ode.Contact]]:
                if geom1.character_id == geom2.character_id and (not self.self_collision or not geom1.character.self_collision):
                    return None

                # self.set_falldown_flag(geom1, geom2)
                if geom1.body is None and geom2.body is not None:
                    geom1, geom2 = geom2, geom1
                contacts: List[ode.Contact] = ode.collide(geom1, geom2, self._contact_count)

                return contacts

            def _generate_contact_joint(self, geom1: ode.GeomObject, geom2: ode.GeomObject, contacts: List[ode.Contact]):
                if geom1.body is None and geom2.body is not None:
                    geom1, geom2 = geom2, geom1

                # Create contact joints. Add contact joint position and contact force in Unity
                if self.contact_type == ODESim.ODEScene.ContactType.ODE_LCP:
                    mu: float = min(geom1.friction, geom2.friction)
                    for c in contacts:
                        c.bounce = self.bounce
                        c.mu = mu
                        if self.use_soft_contact:
                            if self.soft_cfm is not None and self.soft_erp is not None:
                                c.enable_soft_cfm_erp(self.soft_cfm, self.soft_erp)
                            if self.soft_cfm_tan is not None:
                                c.enable_contact_slip(self.soft_cfm_tan)
                        j = ode.ContactJoint(self.world, self.contact, c)  # default mode is ode.ContactApprox1
                        j.setFeedback(self.extract_contact)
                        j.attach(geom1.body, geom2.body)
                elif self.contact_type == ODESim.ODEScene.ContactType.MAX_FORCE_ODE_LCP:
                    max_fric: float = min(geom1.max_friction, geom2.max_friction)  # TODO: set max fric of plane..
                    for c in contacts:
                        c.bounce = 0.0
                        c.mu = max_fric

                        # default world erp in ODE is 0.2
                        # in Samcon implement of Libin Liu,
                        # default dSoftCFM = 0.007;
                        # default dSoftERP = 0.8;

                        if self.use_soft_contact:
                            if self.soft_cfm is not None and self.soft_erp is not None:
                                c.enable_soft_cfm_erp(self.soft_cfm, self.soft_erp)
                            if self.soft_cfm_tan is not None:
                                c.enable_contact_slip(self.soft_cfm_tan)

                        j: ode.ContactJointMaxForce = ode.ContactJointMaxForce(self.world, self.contact, c)
                        if self.use_soft_contact:
                            j.joint_cfm = self.soft_cfm
                            j.joint_erp = self.soft_erp
                        j.setFeedback(self.extract_contact)
                        j.attach(geom1.body, geom2.body)

                elif self.contact_type == ODESim.ODEScene.ContactType.BALL:
                    for c in contacts:
                        j = ode.BallJoint(self.world, self.contact)
                        j.setFeedback(True)
                        j.attach(geom1.body, geom2.body)
                        j.setAnchorNumpy(c.contactPosNumpy)
                else:
                    raise ValueError

            def near_callback(self, args, geom1: ode.GeomObject, geom2: ode.GeomObject):
                contacts = self.contact_basic(geom1, geom2)
                if not contacts:
                    return
                self._generate_contact_joint(geom1, geom2, contacts)

            def _compute_collide_callback(self, args, geom1: ode.GeomObject, geom2: ode.GeomObject):
                contacts = self.contact_basic(geom1, geom2)
                if not contacts:
                    return
                for c in contacts:
                    self.contact_info.pos.append(c.contactPosNumpy)
                    self.contact_info.force.append(c.contactNormalNumpy * c.contactDepth)
                    self.contact_info.geom1_name.append(geom1.name)
                    self.contact_info.geom2_name.append(geom2.name)

            def compute_collide_info(self):
                for character in self.characters:
                    character.fall_down = False

                self.contact_info = ODESim.ODEScene.SceneContactInfo([], [], [], [])
                self.space.collide(None, self._compute_collide_callback)
                self.resort_geoms()
                if self.contact_info.pos:
                    self.contact_info.pos = np.concatenate([i[None, :] for i in self.contact_info.pos], axis=0)
                if self.contact_info.force:
                    self.contact_info.force = np.concatenate([i[None, :] for i in self.contact_info.force], axis=0)

                return self.contact_info

            def extract_body_contact_label(self) -> np.ndarray:
                """
                extract contact label (0/1). here we need not to create contact joints.
                """
                assert len(self.characters) == 1
                contact_label = np.zeros(len(self.character0.bodies), dtype=np.int32)
                def callback(args, geom1: ode.GeomObject, geom2: ode.GeomObject):
                    if geom1.character_id == geom2.character_id and (not self.self_collision or not geom1.character.self_collision):
                        return
                    body1: ode.Body = geom1.body
                    if body1 is not None:
                        contact_label[body1.instance_id] = 1
                    body2: ode.Body = geom2.body
                    if body2 is not None:
                        contact_label[body2.instance_id] = 1

                self.space.collide(None, callback)
                self.space.ResortGeoms()

                return contact_label

            # collision detection
            def pre_simulate_step(self) -> ode.JointGroup:
                for character in self.characters:
                    character.fall_down = False
                self.space.collide((self.world, self.contact), self.near_callback)
                return self.contact
                # self.space.fast_collide()

            def post_simulate_step(self):
                self.contact.empty()  # clear contact joints after simulation
                self.space.ResortGeoms()

            def resort_geoms(self):
                # in Open Dynamics Engine, the order of geometries are changed after a step of forward simulation.
                # make sure the order of geometries are not changed.
                self.space.ResortGeoms()

            def step_fast_collision(self):  # do collision detection in cython, not python
                self.world.step_fast_collision(self.space, self.sim_dt)
                self.space.ResortGeoms()

            def damped_step_fast_collision(self):  # do collision detection in cython, not python
                self.world.damped_step_fast_collision(self.space, self.sim_dt)

            def simulate_once(self):
                self.pre_simulate_step()
                self.world.step(self.sim_dt)  # This will change geometry order in ode space
                if self.extract_contact:
                    self.contact_save()  # save contact info
                self.post_simulate_step()

            def simulate(self, n: int = 0):
                cnt = n if n > 0 else self.step_cnt
                for _ in range(cnt):
                    self.simulate_once()

            def damped_simulate_once(self):
                # self.world.damped_step_fast_collision(self.space, self.sim_dt)
                self.pre_simulate_step()
                self.world.dampedStep(self.sim_dt)  # This will change geometry order in ode space
                # if self.extract_contact:
                #     self.contact_save()
                self.post_simulate_step()

            def fast_simulate_once(self):  # use quick step in ode engine
                self.pre_simulate_step()
                # This will change geometry order in ode space
                self.world.quickStep(self.sim_dt)
                if self.extract_contact:
                    self.contact_save()
                self.post_simulate_step()

            def damped_simulate(self, n: int = 0):
                cnt = n if n > 0 else self.step_cnt
                for _ in range(cnt):
                    self.damped_simulate_once()

            def simulate_no_collision(self, n: int = 0):
                cnt = n if n > 0 else self.step_cnt
                for _ in range(cnt):
                    self.world.step(self.sim_dt)
                    self.space.ResortGeoms()  # make sure simulation result is same

            def damped_simulate_no_collision_once(self):
                self.world.dampedStep(self.sim_dt)  # This will change geometry order in ode space
                self.space.ResortGeoms()

            def damped_simulate_no_collision(self, n: int = 0):
                cnt = n if n > 0 else self.step_cnt
                for _ in range(cnt):
                    self.damped_simulate_no_collision_once()

            def clear(self, gravity: Union[Iterable, float, None] = None):
                """
                clear the scene
                """
                if self.environment is not None:
                    self.environment.clear()

                if self.characters is not None:
                    self.characters.clear()

                self.build_world_and_space(self.default_gravity if gravity is None else gravity)

                if self.ext_joints is not None:
                    self.ext_joints.clear()

                self.environment = ODESim.Environment.Environment(self.world, self.space)
                self.ext_joints = None # ExtJointList(self.world, self.characters)
                self.contact_info = ODESim.ODEScene.SceneContactInfo()

                self.str_info = ""  # use for print or show information in console or Unity

                return self

    class DRigidBodyMassMode(IntEnum):
        """
        Compute the mass of rigid body by density or given mass value
        """
        Density = 0
        MassValue = 1

    class DRigidBodyInertiaMode(IntEnum):
        """
        Compute the inertia of rigid body by density or given inertia value
        """
        Density = 0
        InertiaValue = 1

    class GeomType:
        """
        parse geometry type
        """

        def __init__(self):
            self._sphere: Tuple[str] = ("sphere",)
            self._capsule: Tuple[str, str] = ("capsule", "ccylinder")
            self._box: Tuple[str, str] = ("box", "cube")
            self._plane: Tuple[str] = ("plane",)

        def is_sphere(self, geom_type: str):
            return geom_type.lower() in self._sphere

        def is_capsule(self, geom_type: str):
            return geom_type.lower() in self._capsule

        def is_box(self, geom_type: str):
            return geom_type.lower() in self._box

        def is_plane(self, geom_type: str):
            return geom_type.lower() in self._plane

        @property
        def sphere_type(self) -> str:
            return self._sphere[0]

        @property
        def capsule_type(self) -> str:
            return self._capsule[0]

        @property
        def box_type(self) -> str:
            return self._box[0]

        @property
        def plane_type(self) -> str:
            return self._plane[0]

    class MessDictScale:
        @staticmethod
        def handle_value(key, value, scale: float):
            if key in ["Position", "Scale", "LinearVelocity"]:
                return scale * np.asarray(value)
            elif key in ["Kps", "Inertia"]:
                return (scale ** 5) * np.asarray(value)
            elif key == "Damping":
                return (scale ** 5) * value
            elif key == "Mass":
                return (scale ** 3) * value
            elif isinstance(value, Dict):
                return ODESim.MessDictScale.handle_dict(value, scale)
            elif isinstance(value, List):
                return ODESim.MessDictScale.handle_list(value, scale)
            else:
                return value

        @staticmethod
        def handle_list(mess_list: List, load_scale: float):
            result: List = []
            for value in mess_list:
                result.append(ODESim.MessDictScale.handle_value(None, value, load_scale))
            return result

        @staticmethod
        def handle_dict(mess_dict: Dict[str, Any], load_scale: float):
            result: Dict[str, Any] = {}
            for key, value in mess_dict.items():
                result[key] = ODESim.MessDictScale.handle_value(key, value, load_scale)
            return result

    class JsonCharacterLoader:
        def __init__(
            self,
            world: ode.World,
            space: ode.Space,
            use_hinge: bool = True,
            use_angle_limit: bool = True,
            ignore_parent_collision: bool = True,
            ignore_grandpa_collision: bool = True,
            load_scale: float = 1.0,
            use_as_base_class: bool = False
        ):
            """
            Our character model is defined at world coordinate.
            """
            self.world = world
            self.space = space
            self.use_hinge = use_hinge
            self.use_angle_limit = use_angle_limit

            self.ignore_parent_collision: bool = ignore_parent_collision
            self.ignore_grandpa_collision: bool = ignore_grandpa_collision
            self.load_scale: float = load_scale
            self.ignore_list: List[List[int]] = []

            if not use_as_base_class:
                self.character = ODESim.ODECharacter.ODECharacter(world, space)
                self.character_init = ODESim.ODECharacterInit(self.character)
            else:
                self.character = None
                self.character_init = None

            self.geom_type = ODESim.GeomType()
            self.default_friction: float = 0.8

        @property
        def body_info(self):
            """
            get body info
            """
            return self.character.body_info

        @property
        def joint_info(self):
            """
            get joint info
            """
            return self.character.joint_info

        def joint_names(self) -> List[str]:
            """
            get joint names
            """
            return self.joint_info.joint_names()

        def body_names(self) -> List[str]:
            """
            get body names
            """
            return self.body_info.get_name_list()

        @property
        def end_joint(self):
            return self.character.end_joint

        @property
        def bodies(self) -> List[ode.Body]:
            return self.character.bodies

        @property
        def joints(self) -> List[Union[ode.Joint, ode.BallJoint, ode.BallJointAmotor, ode.HingeJoint]]:
            return self.character.joints

        @property
        def root_joint(self) -> Optional[ode.Joint]:
            return self.character.root_joint

        @property
        def joint_to_child_body(self) -> List[int]:
            return self.character.joint_to_child_body

        @property
        def child_body_to_joint(self) -> List[int]:
            return self.character.child_body_to_joint

        @property
        def joint_to_parent_body(self) -> List[int]:
            return self.character.joint_to_parent_body

        @property
        def has_end_joint(self) -> bool:
            return self.character.has_end_joint

        def set_character(self, character=None):
            self.character = character
            self.character_init.character = character

        def create_geom_object(
            self,
            json_geom: Dict[str, Any],
            calc_mass: bool = True,
            default_density: float = 1000.0,
            friction: Optional[float] = None
        ) -> Tuple[ode.GeomObject, Optional[ode.Mass]]:
            """
            create geometry object
            """
            geom_type: str = json_geom["GeomType"]
            geom_scale = np.array(json_geom["Scale"])

            mass_geom = ode.Mass() if calc_mass else None
            if self.geom_type.is_sphere(geom_type):
                geom_radius: float = geom_scale[0]
                geom = ode.GeomSphere(self.space, geom_radius)
                if calc_mass:
                    mass_geom.setSphere(default_density, geom_radius)
            elif self.geom_type.is_capsule(geom_type):
                geom_radius, geom_length = geom_scale[0], geom_scale[1]
                geom = ode.GeomCCylinder(self.space, geom_radius, geom_length)
                if calc_mass:
                    mass_geom.setCapsule(default_density, 3, geom_radius, geom_length)
            elif self.geom_type.is_box(geom_type):
                geom = ode.GeomBox(self.space, geom_scale)
                if calc_mass:
                    mass_geom.setBox(default_density, *geom_scale)
            elif self.geom_type.is_plane(geom_type):
                # convert position and quaternion to n.x*x+n.y*y+n.z*z = dist
                normal_vec = Rotation(json_geom["Quaternion"]).apply(Common.MathHelper.up_vector())
                dist = np.dot(np.asarray(json_geom["Position"]), normal_vec).item()
                geom = ode.GeomPlane(self.space, normal_vec, dist)
                if calc_mass:
                    raise ValueError("Plane Geom Object dosen't have mass.")
            else:
                raise NotImplementedError(geom_type)

            geom.name = json_geom["Name"]
            geom.collidable = json_geom["Collidable"]
            # print(geom.name, geom.collidable)

            if friction is None:
                friction = self.default_friction
            geom.friction = json_geom["Friction"] if "Friction" in json_geom else friction

            clung_env = json_geom.get("ClungEnv")
            if clung_env is not None:
                geom.clung_env = clung_env

            geom.instance_id = json_geom["GeomID"]

            if not self.geom_type.is_plane(geom_type):
                geom.PositionNumpy = np.asarray(json_geom["Position"])
                geom.QuaternionScipy = np.asarray(json_geom["Quaternion"])

            return geom, mass_geom

        def add_body(self, json_body: Dict[str, Any], update_body_pos_by_com: bool = True) -> ode.Body:
            """
            @param: recompute_body_pos:
            return: ode.Body
            """
            assert json_body["BodyID"] == len(self.bodies)

            body = ode.Body(self.world)
            body.instance_id = json_body["BodyID"]
            geom_info_list: List = json_body["Geoms"]
            geom_info_list.sort(key=lambda x: x["GeomID"])

            body_density: float = json_body["Density"]
            if body_density == 0.0:
                body_density = 1
            create_geom: List[ode.GeomObject] = []

            def geom_creator():
                gmasses: List[ode.Mass] = []
                gcenters = []
                grots = []

                for json_geom in geom_info_list:
                    geom, mass_geom = self.create_geom_object(
                        json_geom, True, body_density,
                        json_body["Friction"] if "Friction" in json_body else None
                    )

                    create_geom.append(geom)
                    gmasses.append(mass_geom)
                    gcenters.append(np.array(json_geom["Position"]))
                    grots.append(Rotation(np.asarray(json_geom["Quaternion"])))

                mass_total_ = self.character_init.compute_geom_mass_attr(body, create_geom, gmasses, gcenters, grots, update_body_pos_by_com)
                return mass_total_

            mass_total = geom_creator()

            mass_mode = ODESim.DRigidBodyMassMode[json_body.get("MassMode", "Density")]
            inertia_mode = ODESim.DRigidBodyInertiaMode[json_body.get("InertiaMode", "Density")]
            # load fixed inertia of this body, rather than compute by geometry
            if inertia_mode == ODESim.DRigidBodyInertiaMode.InertiaValue:
                mass_total.inertia = np.asarray(json_body["Inertia"])

            # load fixed mass value of this body, rather than compute by geometry
            if mass_mode == ODESim.DRigidBodyMassMode.Density:
                if json_body["Density"] == 0.0:
                    mass_total = ode.Mass()
            elif mass_mode == ODESim.DRigidBodyMassMode.MassValue:
                mass_total.mass = json_body["Mass"]
            else:
                raise ValueError(f"mass_mode = {mass_mode}, which is not supported")

            # for debug
            # print(f"name {json_body['Name']}, mass = {mass_total.mass:4f}, "
            #      f"inertia = \n{mass_total.inertia.reshape((3, 3))}\n")

            body.setQuaternionScipy(np.asarray(json_body["Quaternion"]))

            lin_vel = json_body.get("LinearVelocity")  # initial linear velocity
            if lin_vel is not None and len(lin_vel) > 0:
                body.LinearVelNumpy = np.asarray(lin_vel)

            ang_vel = json_body.get("AngularVelocity")  # initial angular velocity
            if ang_vel:
                body.setAngularVelNumpy(np.asarray(ang_vel))

            self.character_init.append_body(body, mass_total, json_body["Name"], json_body["ParentBodyID"])
            return body

        def joint_attach(self, joint: ode.Joint, joint_pos: np.ndarray, joint_parent: int, joint_child: int):
            """
            attach bodies to joint
            """
            if joint_parent == -1:
                joint.attach_ext(self.bodies[joint_child], ode.environment)
            else:
                joint.attach_ext(self.bodies[joint_child], self.bodies[joint_parent])
            if type(joint) == ode.FixedJoint:
                return
            joint.setAnchorNumpy(np.asarray(joint_pos))

        @staticmethod
        def calc_hinge_axis(euler_order: str, axis_mat: Optional[np.ndarray] = None) -> np.ndarray:
            if axis_mat is None:
                axis_mat = np.eye(3)
            return axis_mat[ord(euler_order[0].upper()) - ord('X')]

        @staticmethod
        def set_ball_limit(
            joint: ode.BallJointAmotor,
            euler_order: str,
            angle_limits: Union[List, np.ndarray],
            raw_axis: Optional[np.ndarray] = None
        ):
            return ODESim.JointInfoInit.JointInfoInit.set_ball_joint_limit(joint, euler_order, angle_limits, raw_axis)

        # Not Attached
        @staticmethod
        def create_joint_base(world: ode.World, json_joint: Dict[str, Any], load_hinge: bool = True, use_ball_limit: bool = True):
            if json_joint["JointType"] == "BallJoint":
                if "Character0ID" in json_joint or "Character1ID" in json_joint:
                    joint = ode.BallJointAmotor(world)
                elif json_joint["ParentBodyID"] == -1 or json_joint["Name"] == "RootJoint":
                    joint = ode.BallJoint(world)
                else:
                    if use_ball_limit:
                        joint = ode.BallJointAmotor(world)
                    else:
                        joint = ode.BallJoint(world)

            elif json_joint["JointType"] == "HingeJoint":
                if load_hinge:
                    joint = ode.HingeJoint(world)
                else:
                    joint = ode.BallJointAmotor(world)
            elif json_joint["JointType"] == "FixedJoint":
                joint = ode.FixedJoint(world)
            else:
                raise NotImplementedError

            joint.name = json_joint["Name"]
            joint.euler_order = json_joint["EulerOrder"]
            if "Damping" in json_joint:
                joint.setSameKd(json_joint["Damping"])

            joint.instance_id = json_joint["JointID"]
            return joint

        @staticmethod
        def post_create_joint(
            joint: Union[ode.BallJointAmotor, ode.HingeJoint, ode.BallJoint, ode.FixedJoint],
            json_joint: Dict[str, Any],
            load_limits: bool = True
        ):
            axis_q: Optional[List[float]] = json_joint.get("EulerAxisLocalRot")
            axis_q: np.ndarray = np.asarray(Common.MathHelper.unit_quat() if not axis_q else axis_q)
            axis_mat = Rotation(axis_q).as_matrix()
            joint.euler_axis = axis_mat

            if type(joint) == ode.BallJointAmotor:
                if load_limits and json_joint["JointType"] == "BallJoint":
                    angle_lim = np.vstack([json_joint["AngleLoLimit"], json_joint["AngleHiLimit"]]).T
                    ODESim.JsonCharacterLoader.set_ball_limit(joint, json_joint["EulerOrder"], angle_lim, axis_mat)
            elif type(joint) == ode.HingeJoint:
                hinge_axis = ODESim.JsonCharacterLoader.calc_hinge_axis(json_joint["EulerOrder"], axis_mat)
                joint.setAxis(hinge_axis)
                if load_limits:
                    angle_lim = np.deg2rad([json_joint["AngleLoLimit"][0], json_joint["AngleHiLimit"][0]])
                    joint.setAngleLimit(angle_lim[0], angle_lim[1])
            elif type(joint) == ode.BallJoint:  # need to do nothing here
                pass
            elif type(joint) == ode.FixedJoint:
                joint.setFixed()
            else:
                raise NotImplementedError
            return joint

        def create_joint(self, json_joint: Dict[str, Any], load_hinge: bool = True, load_limits: bool = True) -> ode.Joint:
            """

            :param json_joint:
            :param load_hinge:
            :param load_limits:
            :return:
            """
            joint = self.create_joint_base(self.world, json_joint, load_hinge, load_limits)

            if json_joint["ParentBodyID"] == -1 or json_joint["Name"] == "RootJoint":
                self.joint_info.root_idx = json_joint["JointID"]
                self.joint_info.has_root = True

            self.joint_attach(joint, json_joint["Position"], json_joint["ParentBodyID"], json_joint["ChildBodyID"])
            if json_joint["JointType"] == "HingeJoint" and not load_hinge:  # if load_hinge == False, ignore amotor limit
                return joint
            else:
                return self.post_create_joint(joint, json_joint, load_limits)

        def add_joint(self, json_joint: Dict[str, Any], load_hinge: bool = True, load_limits: bool = True) -> ode.Joint:
            """
            parse joint info
            :param json_joint: joint in json format
            :param load_hinge:
            :param load_limits:
            :return: joint
            """
            assert json_joint["JointID"] == len(self.joints)
            joint = self.create_joint(json_joint, load_hinge, load_limits)

            damping = json_joint.get("Damping")
            if damping:
                self.joint_info.kds[json_joint["JointID"]] = damping
                joint.setSameKd(damping)

            self.joint_info.weights[json_joint["JointID"]] = json_joint.get("Weight", 1.0)
            self.joint_to_child_body.append(json_joint["ChildBodyID"])
            self.joint_info.joints.append(joint)

            return joint

        def load_from_file(self, fname: str):
            """
            Load character from json file
            """
            with open(fname, "r") as f:
                mess_dict: Dict[str, Any] = json.load(f)
            return self.load(mess_dict)

        def load_bodies(self, json_bodies: List, update_body_pos_by_com: bool = True):
            """
            Load bodies in json
            """
            json_bodies.sort(key=lambda x: x["BodyID"])  # sort bodies by BodyID

            for json_body in json_bodies:
                self.add_body(json_body, update_body_pos_by_com)
                ignore = json_body.get("IgnoreBodyID", [])
                self.ignore_list.append(ignore)

            self.parse_ignore_list()

        def parse_ignore_list(self):
            """
            ignore collision detection between some bodies
            """
            for body_idx, ignores in enumerate(self.ignore_list):
                res: List[int] = []
                for ignore_body_id in ignores:
                    if ignore_body_id >= len(self.bodies):
                        logging.warning(f"{ignore_body_id} out of range. ignore.")
                        continue
                    for geom in self.bodies[ignore_body_id].geom_iter():
                        res.append(geom.get_gid())

                for geom in self.bodies[body_idx].geom_iter():
                    geom.extend_ignore_geom_id(res)

        def load_joints(self, json_joints: List):
            """
            load joints in json
            """
            json_joints.sort(key=lambda x: x["JointID"])  # sort joints by JointID
            self.joint_info.kds = np.zeros(len(json_joints))
            self.joint_info.weights = np.ones(len(json_joints))
            for json_joint in json_joints:
                self.add_joint(json_joint, self.use_hinge, self.use_angle_limit)

        def load_endjoints(self, json_endjoints: List):
            """
            Load end joints in json
            """
            joint_id, name, end_pos = [], [], []
            for json_endjoint in json_endjoints:
                joint_id.append(json_endjoint["ParentJointID"])
                name.append(json_endjoint["Name"])
                end_pos.append(np.array(json_endjoint["Position"]))

            body_id = [self.joint_to_child_body[i] for i in joint_id]
            self.character_init.init_end_joint(name, body_id, np.array(end_pos))

        def load_pd_control_param(self, json_pd_param: Dict[str, Any]):
            """
            Load PD Control Param in json
            """
            kps = np.asarray(json_pd_param["Kps"])
            torque_lim = np.asarray(json_pd_param["TorqueLimit"])
            if len(kps) == 0 or len(torque_lim) == 0:
                return
            assert kps.size == len(self.joints)
            assert torque_lim.size == len(self.joints)
            self.joint_info.kps = kps
            self.joint_info.torque_limit = torque_lim

        def load_init_root_info(self, init_root_param: Dict[str, Any]):
            info = ODESim.ODECharacter.DRootInitInfo()
            info.pos = np.array(init_root_param["Position"])
            info.quat = np.array(init_root_param["Quaternion"])
            self.character.root_init_info = info

        def load(self, mess_dict: Dict[str, Any]):
            """
            Load ODE Character from json file
            """
            # for test
            # mess_dict = MessDictScale.handle_dict(mess_dict, 5.0)
            self.ignore_parent_collision &= mess_dict.get("IgnoreParentCollision", True)
            self.ignore_grandpa_collision &= mess_dict.get("IgnoreGrandpaCollision", True)

            name: Optional[str] = mess_dict.get("CharacterName")
            if name:
                self.character.name = name

            label: Optional[str] = mess_dict.get("CharacterLabel")
            if label:
                self.character.label = label

            self.load_bodies(mess_dict["Bodies"])

            joints: List[Dict[str, Any]] = mess_dict.get("Joints")
            if joints:
                self.load_joints(joints)

            self.character_init.init_after_load(mess_dict["CharacterID"],
                                                self.ignore_parent_collision,
                                                self.ignore_grandpa_collision)

            self_colli: Optional[bool] = mess_dict.get("SelfCollision")
            if self_colli is not None:
                self.character.self_collision = self_colli

            kinematic: Optional[bool] = mess_dict.get("Kinematic")
            if kinematic is not None:
                self.character.is_kinematic = kinematic

            pd_param: Dict[str, List[float]] = mess_dict.get("PDControlParam")
            if pd_param:
                self.load_pd_control_param(pd_param)

            end_joints: List[Dict[str, Any]] = mess_dict.get("EndJoints")
            if end_joints:
                self.load_endjoints(end_joints)

            init_root: Dict[str, Any] = mess_dict.get("RootInfo")
            if init_root:
                self.load_init_root_info(init_root)

            return self.character

    class JsonSceneLoader:
        """
        Load Scene in json format generated from Unity
        """
        class AdditionalConfig:
            """
            Additional Configuration in loading json scene
            """
            def __init__(self):
                self.gravity: Optional[List[float]] = None
                self.step_count: Optional[int] = None
                self.render_fps: Optional[int] = None

                self.cfm: Optional[float] = None
                self.simulate_fps: Optional[int] = None
                self.use_hinge: Optional[bool] = None
                self.use_angle_limit: Optional[bool] = None
                self.self_collision: Optional[bool] = None

            def update_config_dict(self, mess: Optional[Dict[str, Any]]) -> Dict[str, Any]:
                if mess is None:
                    mess = {}

                change_attr = mess.get("ChangeAttr", {})
                if self.gravity is not None:
                    change_attr["Gravity"] = self.gravity
                if self.step_count is not None:
                    change_attr["StepCount"] = self.step_count
                if self.render_fps is not None:
                    change_attr["RenderFPS"] = self.render_fps

                fixed_attr = mess.get("FixedAttr", {})
                if self.cfm is not None:
                    fixed_attr["CFM"] = self.cfm
                if self.simulate_fps is not None:
                    fixed_attr["SimulateFPS"] = self.simulate_fps
                if self.use_hinge is not None:
                    fixed_attr["UseHinge"] = self.use_hinge
                if self.use_angle_limit is not None:
                    fixed_attr["UseAngleLimit"] = self.use_angle_limit
                if self.self_collision is not None:
                    fixed_attr["SelfCollision"] = self.self_collision

                if len(fixed_attr) > 0:
                    mess["FixedAttr"] = fixed_attr
                if len(change_attr) > 0:
                    mess["ChangeAttr"] = change_attr

                return mess

        def __init__(self, scene=None, is_running: bool = False):
            self.scene = scene
            if self.scene is None:
                self.scene = ODESim.ODEScene.ODEScene()

            self.use_hinge: bool = True
            self.use_angle_limit: bool = True
            self.is_running = is_running

        @property
        def world(self) -> ode.World:
            return self.scene.world

        @property
        def space(self) -> ode.SpaceBase:
            return self.scene.space

        @property
        def characters(self):
            return self.scene.characters

        @property
        def character0(self):
            return self.scene.character0

        @property
        def environment(self):
            return self.scene.environment

        @property
        def ext_joints(self) :
            return self.scene.ext_joints

        def file_load(self, fname: str, config=None):
            if fname.endswith(".pickle"):
                return self.load_from_pickle_file(fname, config)
            elif fname.endswith(".json"):
                return self.load_from_file(fname, config)
            else:
                raise NotImplementedError

        def load_from_file(self, fname: str, config= None):
            with open(fname, "r") as f:
                mess_dict = json.load(f)
            return self.load_json(mess_dict, config)

        def load_from_pickle_file(self, fname: str, config=None):
            fname = os.path.abspath(fname)
            with open(fname, "rb") as f:
                mess_dict = pickle.load(f)
            logging.info(f"load from pickle file {fname}")
            return self.load_json(mess_dict, config)

        def load_environment(self, mess_dict: Dict[str, Any]):
            geom_info_list: List[Dict] = mess_dict["Geoms"]
            geom_info_list.sort(key=lambda x: x["GeomID"])
            helper = ODESim.JsonCharacterLoader(self.world, self.space, use_as_base_class = True)
            for geom_json in geom_info_list:
                geom, _ = helper.create_geom_object(geom_json, False)
                self.environment.geoms.append(geom)
                # if helper.geom_type.is_plane(geom_json["GeomType"]):  # assume there is only 1 plane
                #    self.environment.floor = geom
                geom.character_id = -1

            self.environment.get_floor_in_list()
            return self.environment

        def load_ext_joints(self, mess_dict: Dict[str, Any]):  # load constraint joints
            joints: List[Dict[str, Any]] = mess_dict["Joints"]
            joints.sort(key=lambda x: x["JointID"])
            for joint_json in joints:
                ext_joint = ODESim.JsonCharacterLoader.create_joint_base(self.world, joint_json, self.use_hinge)
                self.ext_joints.append_and_attach(ext_joint, joint_json["Character0ID"], joint_json["Body0ID"],
                                                joint_json["Character1ID"], joint_json["Body1ID"])
                ODESim.JsonCharacterLoader.post_create_joint(ext_joint, joint_json, self.use_angle_limit)

        def load_ext_forces(self, mess_dict: Dict[str, Any]):
            """
            Load external forces. such as force from mouse drag/push in Unity Scene
            """
            ch_dict = self.scene.get_character_id_map()  # key: character id. value: index in characterlist
            forces: List[Dict[str, Any]] = mess_dict["Forces"]
            for finfo in forces:
                character = ch_dict[finfo["CharacterID"]]
                body: ode.Body = character.bodies[finfo["BodyID"]]
                pos: np.ndarray = np.asarray(finfo["Position"])
                force: np.ndarray = np.asarray(finfo["Force"])
                # print(f"body pos = {body.PositionNumpy}, force pos = {pos}, force = {force}")
                body.addForceAtPosNumpy(force, pos)

        def load_world_attr(self, mess_dict: Dict[str, Any]):
            change_attr = mess_dict.get("ChangeAttr")
            if change_attr:
                self.scene.set_gravity(change_attr["Gravity"])
                self.scene.set_render_fps(change_attr["RenderFPS"])
                step_cnt: Optional[int] = change_attr.get("StepCount")
                if step_cnt:
                    self.scene.step_cnt = step_cnt

            fixed_attr: Optional[Dict[str, Any]] = mess_dict.get("FixedAttr")
            if fixed_attr and not self.is_running:
                if "SimulateFPS" in fixed_attr:
                    self.scene.set_sim_fps(fixed_attr["SimulateFPS"])
                if "UseHinge" in fixed_attr:
                    self.use_hinge = fixed_attr["UseHinge"]
                if "UseAngleLimit" in fixed_attr:
                    self.use_angle_limit = fixed_attr["UseAngleLimit"]
                if "CFM" in fixed_attr:
                    self.scene.world.CFM = fixed_attr["CFM"]
                if "SelfCollision" in fixed_attr:
                    self.scene.self_collision = fixed_attr["SelfCollision"]

        def load_character_list(self, mess_dict: Dict[str, Any]):
            character_info_list: List[Dict[str, Any]] = mess_dict["Characters"]
            # character_info_list.sort(key=lambda x: x["CharacterID"])
            for character_info in character_info_list:
                # for debug..
                loader = ODESim.JsonCharacterLoader(self.world, self.space, self.use_hinge, self.use_angle_limit)
                character = loader.load(character_info)
                self.characters.append(character)

            return self.characters

        def load_json(self, mess_dict: Dict[str, Any], config: Optional[AdditionalConfig] = None):
            world_attr: Optional[Dict[str, Any]] = mess_dict.get("WorldAttr")
            if config is not None:
                world_attr = config.update_config_dict(world_attr)

            if world_attr:
                self.load_world_attr(world_attr)

            if not self.is_running:  # for debug. TODO: remove this condition
                env_mess = mess_dict.get("Environment")
                if env_mess:
                    self.load_environment(env_mess)

            characters_mess = mess_dict.get("CharacterList")
            if characters_mess:
                self.load_character_list(characters_mess)

            ext_joints = mess_dict.get("ExtJointList")
            if ext_joints:
                self.load_ext_joints(ext_joints)

            ext_force = mess_dict.get("ExtForceList")
            if ext_force:
                self.load_ext_forces(ext_force)

            self.scene.resort_geoms()
            return self.scene


    class MeshCharacterLoader:
        def __init__(self, world: ode.World, space: ode.SpaceBase):
            self.world = world
            self.space = space
            self.character = ODESim.ODECharacter.ODECharacter(world, space)
            self.character_init = ODESim.ODECharacterInit(self.character)
            self.default_friction: float = 0.8

        def load_from_obj(self, obj_path, meshname, volume_scale=1, density_scale=1, inverse_xaxis=True):
            import trimesh
            self.character.name = meshname

            pymesh = trimesh.load_mesh(obj_path)
            pymesh.density *= density_scale
            pymesh.apply_scale(volume_scale)

            meshbody = ode.Body(self.world)
            meshData = ode.TriMeshData()
            if inverse_xaxis:
                pymesh.vertices[:, 0] = -pymesh.vertices[:, 0]
                trimesh.repair.fix_inversion(pymesh)
            meshData.build(pymesh.vertices-pymesh.center_mass, pymesh.faces)
            meshGeom = ode.GeomTriMesh(meshData, self.space)
            meshGeom.body = meshbody

            meshMass = ode.Mass()
            meshMass.setParameters(pymesh.mass_properties['mass'], 0.0, 0.0, 0.0,
                pymesh.moment_inertia[0, 0], pymesh.moment_inertia[1, 1], pymesh.moment_inertia[2, 2],
                pymesh.moment_inertia[0, 1], pymesh.moment_inertia[0, 2], pymesh.moment_inertia[1, 2])
            meshbody.setMass(meshMass)

            self.character_init.append_body(meshbody, meshMass, meshname, -1)
            self.character_init.init_after_load()
            return self.character


    class CharacterTOBVH:
        def __init__(self, character, sim_fps: int = 120):
            self.character = character
            self.motion = pymotionlib.MotionData.MotionData()
            self.buffer = []
            self.end_site_info = None
            self.motion_backup: Optional[pymotionlib.MotionData.MotionData] = None
            self.sim_fps = int(sim_fps)
            self.joints = character.joints
            self.body_info = character.body_info
            self.joint_info = character.joint_info

        @property
        def root_idx(self):
            return self.character.joint_info.root_idx

        def deepcopy(self):
            result = self.__class__(self.character)
            result.motion = copy.deepcopy(self.motion)
            result.end_site_info = copy.deepcopy(self.end_site_info)
            return result

        def build_hierarchy_base(self):
            self.motion._fps = self.sim_fps
            self.motion._num_frames = 0

        def bvh_hierarchy_no_root(self):
            if self.character.joint_info.has_root:
                raise ValueError("There should be no root joint.")

            old_state = self.character.save()
            self.character.load_init_state()
            self.build_hierarchy_base()
            self.motion._num_joints = len(self.joints) + 1

            # When insert virtual root joint at the front, index of other joints add by 1
            self.motion._skeleton_joint_parents = [-1] + (np.array(self.character.joint_info.pa_joint_id) + 1).tolist()
            self.motion._skeleton_joints = ["RootJoint"] + self.character.joint_info.joint_names()

            localqs, offset = self.character.joint_info.get_relative_local_pos()
            self.motion._skeleton_joint_offsets = np.concatenate([np.zeros((1, 3)), offset])

            self.end_site_info = dict(
                [(pa_jidx + 1, self.character.end_joint.jtoj_init_local_pos[idx])
                for idx, pa_jidx in enumerate(self.character.end_joint.pa_joint_id)]
            )

            self.motion_backup = copy.deepcopy(self.motion)
            self.character.load(old_state)

        def build_hierarchy_with_root(self):
            assert self.character.joint_info.has_root
            old_state = self.character.save()
            self.character.load_init_state()
            self.build_hierarchy_base()
            self.motion._num_joints = len(self.joints)
            self.motion._skeleton_joint_parents = copy.deepcopy(self.character.joint_info.pa_joint_id)
            self.motion._skeleton_joints = self.character.joint_info.joint_names()
            localqs, offset = self.character.joint_info.get_relative_local_pos()
            self.motion._skeleton_joint_offsets = offset
            self.end_site_info = dict(
                [(pa_jidx, self.character.end_joint.jtoj_init_local_pos[idx])
                for idx, pa_jidx in enumerate(self.character.end_joint.pa_joint_id)]
            )
            self.motion_backup = copy.deepcopy(self.motion)
            self.character.load(old_state)

        def build_hierarchy(self):
            if self.character.joint_info.has_root:
                self.build_hierarchy_with_root()
            else:
                self.bvh_hierarchy_no_root()

            return self

        def bvh_append_with_root(self):
            assert self.joint_info.has_root
            translation = np.zeros((1, len(self.joints), 3))
            translation[0, 0, :] = self.joint_info.root_joint.getAnchorNumpy()
            localqs: np.ndarray = self.joint_info.get_local_q()
            rotation = localqs[None, ...]

            # Append to the mocap data
            self.motion.append_trans_rotation(translation, rotation)
            return self.motion

        def bvh_append_no_root(self):
            # assume that root body's index is 0.
            assert self.body_info.root_body_id == 0
            root_pos = self.character.root_body.PositionNumpy
            translation = np.concatenate([root_pos[np.newaxis, :], np.zeros((len(self.joints), 3))], axis=0).reshape((1, -1, 3))

            # joint_rotation
            root_rot: np.ndarray = self.character.root_body.getQuaternionScipy()
            localqs: np.ndarray = self.joint_info.get_local_q()
            rotation = np.concatenate([root_rot.reshape((1, 4)), localqs], axis=0)[None, :, :]

            # Append to the mocap data
            self.motion.append_trans_rotation(translation, rotation)
            return self.motion

        def append_with_root_to_buffer(self):
            motion_back = copy.deepcopy(self.motion)
            self.bvh_append_with_root()
            self.buffer.append(self.motion)
            self.motion = motion_back

        def append_no_root_to_buffer(self):
            motion_back = copy.deepcopy(self.motion)
            self.bvh_append_no_root()
            self.buffer.append(self.motion)
            self.motion = motion_back

        def insert_end_site(self, motion: Optional[pymotionlib.MotionData.MotionData] = None):
            """
            insert end site to self.motion..
            """
            if motion is None:
                motion = self.motion

            end_site_list = [[key, value] for key, value in self.end_site_info.items()]
            end_site_list.sort(key=lambda x: x[0])
            motion._end_sites = []
            parent_res = motion.joint_parents_idx
            jnames = motion.joint_names
            joffs: List[np.ndarray] = [i for i in motion.joint_offsets]
            jtrans: List[np.ndarray] = [motion.joint_translation[:, i, :] for i in range(motion.num_joints)]
            jrots: List[np.ndarray] = [motion.joint_rotation[:, i, :] for i in range(motion.num_joints)]
            trans_zero: np.ndarray = np.zeros_like(jtrans[0])
            rots_zero: np.ndarray = Common.MathHelper.unit_quat_arr(jrots[0].shape)

            for enum_idx, end_node in enumerate(end_site_list):
                end_idx, end_off = end_node
                end_idx = int(end_idx + enum_idx + 1)
                motion.end_sites.append(end_idx)
                after_list = [end_idx - 1]
                jnames.insert(end_idx, jnames[end_idx - 1] + "_end")
                joffs.insert(end_idx, end_off)
                jtrans.insert(end_idx, trans_zero)
                jrots.insert(end_idx, rots_zero)

                for parent in parent_res[end_idx:]:
                    if parent < end_idx:
                        after_list.append(parent)
                    else:
                        after_list.append(parent + 1)

                parent_res = parent_res[:end_idx] + after_list


            children = [[] for _ in range(len(parent_res))]
            for i, p in enumerate(parent_res[1:]):
                children[p].append(i + 1)
            motion._num_joints = len(parent_res)
            motion._joint_translation = np.concatenate([i[:, None, :] for i in jtrans], axis=1)
            motion._joint_rotation = np.concatenate([i[:, None, :] for i in jrots], axis=1)
            motion._joint_position = None # np.zeros_like(motion._joint_translation)
            motion._joint_orientation = None
            motion._skeleton_joint_offsets = np.concatenate([i[None, ...] for i in joffs], axis=0)
            motion._skeleton_joint_parents = parent_res

            return motion

        def merge_buf(self):
            if self.buffer:
                self.motion._joint_rotation = np.concatenate([motion._joint_rotation for motion in self.buffer], axis=0)
                self.motion._joint_translation = np.concatenate([motion._joint_translation for motion in self.buffer], axis=0)
                self.motion._num_frames = len(self.buffer)
                self.buffer.clear()

            return self.motion

        def ret_merge_buf(self) -> pymotionlib.MotionData.MotionData:
            self.merge_buf()
            if self.end_site_info:
                self.insert_end_site()
            ret_motion = self.motion
            self.motion = copy.deepcopy(self.motion_backup)

            return ret_motion

        def forward_kinematics(
            self,
            root_pos: np.ndarray,
            root_quat: np.ndarray,
            joint_local_quat: np.ndarray
        ) -> pymotionlib.MotionData.MotionData:
            assert root_pos.shape[0] == root_quat.shape[0] == joint_local_quat.shape[0]
            assert root_pos.shape[-1] == 3 and root_quat.shape[-1] == 4 and joint_local_quat.shape[-1] == 4
            # make sure joint order are same..
            num_frame: int = root_pos.shape[0]
            ret_motion = self.motion.get_hierarchy(True)
            assert not ret_motion._end_sites
            ret_motion._num_frames = num_frame
            ret_motion._joint_translation = np.zeros((num_frame, ret_motion.num_joints, 3))
            ret_motion._joint_translation[:, 0, :] = root_pos[:, :]
            ret_motion._joint_rotation = Common.MathHelper.unit_quat_arr((num_frame, ret_motion.num_joints, 4))
            ret_motion._joint_rotation[:, 0, :] = root_quat[:, :]
            ret_motion._joint_rotation[:, 1:, :] = joint_local_quat[:, :, :]
            ret_motion._joint_rotation = np.ascontiguousarray(ret_motion._joint_rotation)

            # recompute global info
            ret_motion.recompute_joint_global_info()

            return ret_motion

        def to_file(self, fname: str = "test.bvh", print_info=True) -> pymotionlib.MotionData.MotionData:
            self.merge_buf()
            if self.end_site_info:
                self.insert_end_site()
            if fname and not os.path.isdir(fname):
                try:
                    pymotionlib.BVHLoader.save(self.motion, fname)
                    if print_info:
                        print(f"Write BVH file to {fname}, with frame = {self.motion.num_frames}, fps = {self.motion.fps}")
                except IOError as arg:
                    print(arg)

            ret_motion = self.motion
            self.motion = copy.deepcopy(self.motion_backup)

            return ret_motion


    class Utils:
        class BVHJointMap:
            def __init__(self, bvh: pymotionlib.MotionData.MotionData, character):
                self.character = character
                self.bvh = bvh

                bvh_name_idx = pymotionlib.MotionHelper.calc_name_idx(self.bvh)

                # build bvh children list
                self.bvh_children = pymotionlib.MotionHelper.calc_children(self.bvh)

                # index is joint index in character, and self.character_to_bvh[]
                self.character_to_bvh = np.array([bvh_name_idx[joint.name] for joint in self.joints])
                self.bvh_to_character: List[int] = [2 * self.bvh_joint_cnt for _ in range(self.bvh_joint_cnt)]
                for character_idx, bvh_idx in enumerate(self.character_to_bvh):
                    self.bvh_to_character[bvh_idx] = character_idx

                self.end_to_bvh: List[int] = []
                # self.refine_hinge_rotation()
                # assert silce is available
                joint_names = self.joint_names()
                assert np.all(np.array(self.bvh.joint_names)[self.character_to_bvh] == np.array(joint_names))
                for index, node in enumerate(self.bvh_to_character):
                    if node < 2 * self.bvh_joint_cnt:
                        assert self.bvh.joint_names[index] == joint_names[node]

                # consider there is no end joints
                if bvh.end_sites:
                    for pa_character_idx in self.end_joint.pa_joint_id:
                        bvh_pa_idx = self.character_to_bvh[pa_character_idx]
                        self.end_to_bvh.append(self.bvh_children[bvh_pa_idx][0])

            @property
            def body_info(self):
                """
                get body info
                """
                return self.character.body_info

            @property
            def joint_info(self):
                """
                get joint info
                """
                return self.character.joint_info

            def joint_names(self) -> List[str]:
                """
                get joint names
                """
                return self.joint_info.joint_names()

            def body_names(self) -> List[str]:
                """
                get body names
                """
                return self.body_info.get_name_list()

            @property
            def end_joint(self):
                return self.character.end_joint

            @property
            def world(self) -> ode.World:
                return self.character.world

            @property
            def space(self) -> ode.SpaceBase:
                return self.character.space

            @property
            def bodies(self) -> List[ode.Body]:
                return self.character.bodies

            @property
            def joints(self) -> List[Union[ode.Joint, ode.BallJoint, ode.BallJointAmotor, ode.HingeJoint]]:
                return self.character.joints

            @property
            def root_joint(self) -> Optional[ode.Joint]:
                return self.character.root_joint

            @property
            def joint_to_child_body(self) -> List[int]:
                return self.character.joint_to_child_body

            @property
            def child_body_to_joint(self) -> List[int]:
                return self.character.child_body_to_joint

            @property
            def joint_to_parent_body(self) -> List[int]:
                return self.character.joint_to_parent_body

            @property
            def has_end_joint(self) -> bool:
                return self.character.has_end_joint

            @property
            def bvh_joint_cnt(self):
                """
                bvh joint count
                """
                return len(self.bvh.joint_names)


class Render:

    class RenderWorld:
        def __init__(self, myworld):
            if isinstance(myworld, ODESim.ODEScene.ODEScene):
                myworld = myworld.world
            self.system = platform.system() == 'Windows'
            if self.system:
                ode.visSetWorld(myworld)
                import atexit
                atexit.register(self.kill)
            else:
                raise NotImplementedError

        def check_wid(self):
            if self.system:
                ode.visGetWorld()
            else:
                raise NotImplementedError

        def track_body(self, body, sync_y):
            if self.system:
                ode.visTrackBody(body, False, sync_y)
            else:
                raise NotImplementedError

        def look_at(self, pos, target, up):
            if self.system:
                ode.visLookAt(pos, target, up)
            else:
                raise NotImplementedError

        def set_color(self, col):
            if self.system:
                ode.visSetColor(col)
            else:
                raise NotImplementedError

        def set_joint_radius(self, r):
            if self.system:
                ode.visSetJointRadius(r)
            else:
                raise NotImplementedError

        def set_axis_length(self, x):
            if self.system:
                ode.visSetAxisLength(x)
            else:
                raise NotImplementedError

        def draw_background(self, x):
            if self.system:
                ode.visDrawBackground(x)
            else:
                raise NotImplementedError

        def draw_hingeaxis(self, x):
            if self.system:
                ode.visWhetherHingeAxis(x)
            else:
                raise NotImplementedError

        def draw_localaxis(self, x):
            if self.system:
                ode.visWhetherLocalAxis(x)
            else:
                raise NotImplementedError

        def start(self):
            if self.system:
                ode.visDrawWorld()
            else:
                raise NotImplementedError

        def kill(self):
            if self.system:
                import os
                print("Killing renderer!", "taskkill /f /pid " + str(os.getpid()), flush=True)
                os.system("taskkill /f /pid " + str(os.getpid()))
                ode.visKill()
                print("Killed renderer!", flush=True)
                os.system("taskkill /f /pid " + str(os.getpid()))
            else:
                raise NotImplementedError

        def pause(self, time_):
            if self.system:
                ode.visPause(time_)
            else:
                raise NotImplementedError

        @staticmethod
        def get_screen_buffer(self):
            """
            We should record video when update function in drawstuff is called..
            """
            img: np.ndarray = ode.visGetScreenBuffer()
            return img

        @staticmethod
        def start_record_video():
            ode.visStartRecordVideo()


class Utils:
    class SliceChangeMode(IntEnum):
        front = 0
        behind = 1

    class MergeMode(IntEnum):
        only_root = 0
        all_body = 1

    class InvDynAttr:
        def __init__(self, ref_start: int, ref_end: int) -> None:
            self.ref_start = ref_start
            self.ref_end = ref_end

    @staticmethod
    def smooth_motion_data(
        bvh: pymotionlib.MotionData.MotionData,
        smooth_type: Union[Common.SmoothOperator.GaussianBase, Common.SmoothOperator.ButterWorthBase],
        test_out_fname: Optional[str] = None,
        smooth_position: bool = True,
        smooth_rotation: bool = True,
    ):
        # smooth bvh joint rotation, position and recompute...

        # filter joint rotations as rotvec dosen't work.
        # because a[i] * a[i+1] < 0 may occurs.
        # we can not let a[i] := -a[i] directly.

        # filter joint rotations as 3x3 rotation matrix for filter is not plausible...
        # in butter worth filter, for walking motion with sample 0.12 KHz, cut off 5 Hz, the result will not change too much..

        print(f"Smooth motion data")
        if smooth_rotation:
            for j_idx in range(bvh.num_joints):
                vec6d: np.ndarray = Common.MathHelper.quat_to_vec6d(bvh.joint_rotation[:, j_idx, :]).reshape((-1, 6))
                vec6d_new: np.ndarray = Utils.smooth_operator(vec6d, smooth_type)
                bvh._joint_rotation[:, j_idx, :] = Common.MathHelper.flip_quat_by_dot(Common.MathHelper.vec6d_to_quat(vec6d_new.reshape((-1, 3, 2))))

        # filter root global position
        if smooth_position:
            bvh.joint_translation[:, 0, :] = Utils.smooth_operator(bvh.joint_translation[:, 0, :], smooth_type)

        bvh.recompute_joint_global_info()

        if test_out_fname:
            pymotionlib.BVHLoader.save(bvh, test_out_fname)
            print(f"Save to {test_out_fname}, with num frame = {bvh.num_frames}")

        return bvh


