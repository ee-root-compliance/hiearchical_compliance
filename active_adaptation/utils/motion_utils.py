from scipy.spatial.transform import Rotation as sRot
import numpy as np

def rotate_to_body(root_quat, vecs):
    """
    Rotate world-frame vectors into the body frame defined by the root quaternion.
    root_quat: (T,4) scalar-first (wxyz)
    vecs: (T, N, 3)
    Returns (T, N, 3)
    """
    r = sRot.from_quat(root_quat, scalar_first=True)
    inv = r.inv().as_matrix()  # (T,3,3)
    return np.einsum('tij,tnj->tni', inv, vecs)

from typing import Sequence, Tuple, List, Any, Union
import numpy as np

def select_in_order(
    original: Union[Sequence[Any], np.ndarray],
    whitelist: Sequence[Any],
    return_missing: bool = False,
) -> Union[
    Tuple[List[Any], List[int]],
    Tuple[np.ndarray, np.ndarray],
    Tuple[List[Any], List[int], List[Any]],
    Tuple[np.ndarray, np.ndarray, List[Any]]
]:
    """
    Select elements from original in whitelist order.
    - original: One-dimensional list/tuple/np.ndarray (elements are usually strings)
    - whitelist: Desired order (entries may be missing)
    - return_missing: When True, also return whitelist entries missing from original

    Returns:
      selected: Selected elements (in whitelist order)
      idx     : Indices of these elements in original (for slicing the source data)
      missing : Optional list of whitelist entries missing from original
    """
    # Convert to a Python list for mapping
    if isinstance(original, np.ndarray):
        if original.ndim != 1:
            raise ValueError("original must be a one-dimensional sequence/array")
        original_list = original.tolist()
        return_np = True
    else:
        original_list = list(original)
        return_np = False

    # Build a value-to-first-index map (use the first occurrence for duplicates)
    index_map = {}
    for i, v in enumerate(original_list):
        if v not in index_map:
            index_map[v] = i

    # Select elements present in original, following whitelist order
    selected = [x for x in whitelist if x in index_map]
    idx = [index_map[x] for x in selected]
    missing = [x for x in whitelist if x not in index_map] if return_missing else None

    # Preserve the return type of original
    if return_np:
        selected = np.array(selected, dtype=object if original.dtype == object else original.dtype)
        idx = np.array(idx, dtype=int)
        return (selected, idx, missing) if return_missing else (selected, idx)
    else:
        return (selected, idx, missing) if return_missing else (selected, idx)

import numpy as np
from typing import Union
from scipy.spatial.transform import Rotation as sRot

def angvel_from_rot(rot: Union[np.ndarray, sRot], fps: float, quat_order: str = "xyzw") -> np.ndarray:
    """
    Robustly estimate angular velocity from quaternion derivatives (world frame).
    Formula: omega_world = 2 * ( qdot ⊗ conj(q) ).vec

    Parameters
    ----
    rot : np.ndarray[T,4] (quaternions), np.ndarray[T,3,3] (rotation matrices), or scipy Rotation
        For quaternion arrays, the default order is xyzw; use quat_order to specify wxyz.
    fps : float
        Sampling frequency (Hz), used for numerical differentiation.
    quat_order : {"xyzw","wxyz"}
        Input quaternion order; default is "xyzw".

    Returns
    ----
    np.ndarray[T,3] : Angular velocity in the world frame (rad/s).
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")
    # Extract quaternions (xyzw)
    if isinstance(rot, sRot):
        quat_xyzw = rot.as_quat().astype(np.float64)  # (T,4), xyzw
    else:
        arr = np.asarray(rot)
        if arr.ndim == 2 and arr.shape[1] == 4:
            quat_xyzw = arr.astype(np.float64)  # (T,4)
            if quat_order.lower() == "wxyz":
                # Convert to xyzw
                quat_xyzw = np.concatenate([quat_xyzw[:, 1:], quat_xyzw[:, :1]], axis=-1)
            elif quat_order.lower() != "xyzw":
                raise ValueError("quat_order must be 'xyzw' or 'wxyz'.")
        elif arr.ndim == 3 and arr.shape[1:] == (3, 3):
            quat_xyzw = sRot.from_matrix(arr).as_quat().astype(np.float64)  # (T,4), xyzw
        else:
            raise ValueError("rot must be Rotation, (T,4) quats, or (T,3,3) matrices.")

    T = quat_xyzw.shape[0]
    if T == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if T == 1:
        return np.zeros((1, 3), dtype=np.float32)

    # Convert to wxyz and normalize
    q_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)  # (T,4)
    q_wxyz /= np.linalg.norm(q_wxyz, axis=1, keepdims=True).clip(min=1e-12)

    # Unwrap: enforce adjacent dot products >= 0 to avoid q/-q flips
    dots = np.sum(q_wxyz[1:] * q_wxyz[:-1], axis=1)
    flip_idx = np.where(dots < 0)[0] + 1  # Boolean indexing creates a copy; use explicit indices here
    if flip_idx.size > 0:
        q_wxyz[flip_idx] *= -1.0

    # Central differences / one-sided endpoint differences for qdot (1/s)
    qdot = np.zeros_like(q_wxyz)
    qdot[1:-1] = (q_wxyz[2:] - q_wxyz[:-2]) * (fps / 2.0)
    qdot[0]    = (q_wxyz[1]  - q_wxyz[0])  * fps
    qdot[-1]   = (q_wxyz[-1] - q_wxyz[-2]) * fps

    # Hamilton product (wxyz)
    def qmul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw
        ], axis=-1)

    # Conjugate (inverse of a unit quaternion)
    q_conj = q_wxyz.copy()
    q_conj[:, 1:] *= -1.0

    # omega_world = 2 * (qdot ⊗ q_conj).vec
    omega_quat = qmul_wxyz(qdot, q_conj) * 2.0
    omega_world = omega_quat[:, 1:]  # Extract the vector part

    return omega_world.astype(np.float32)
