import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp


def bidir_speed(pos, fps):
    """Bi-directional speed: max of forward/backward displacement at each frame.

    Args:
        pos: (N, 3) positions
        fps: frames per second

    Returns:
        (N,) speed in m/s
    """
    fwd = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    s = np.zeros(len(pos))
    s[0] = fwd[0]
    s[-1] = fwd[-1]
    s[1:-1] = np.maximum(fwd[:-1], fwd[1:])
    return s


def find_active_range(transl_list, fps, slow_thresh):
    """Find the first/last active frame across multiple hands.

    Trims leading/trailing frames where ANY hand is below slow_thresh.

    Args:
        transl_list: list of (N, 3) translation arrays, one per hand
        fps: frames per second
        slow_thresh: speed threshold (m/s) below which a hand is "stationary"

    Returns:
        (first, last) frame indices, or None if entirely stationary
    """
    N = len(transl_list[0])
    any_slow = np.zeros(N, dtype=bool)
    for transl in transl_list:
        any_slow |= (bidir_speed(transl, fps) < slow_thresh)

    active = ~any_slow
    if not active.any():
        return None

    first = int(np.argmax(active))
    last = int(N - 1 - np.argmax(active[::-1]))
    return first, last


def slerp_interp_rotmat(rotmats, valid_mask):
    """SLERP interpolation on rotation matrices for invalid frames.

    Args:
        rotmats: (N, 3, 3) rotation matrices
        valid_mask: (N,) boolean, True = valid frame

    Returns:
        (N, 3, 3) interpolated rotation matrices
    """
    valid_idx = np.where(valid_mask)[0]
    invalid_idx = np.where(~valid_mask)[0]
    if len(valid_idx) < 2 or len(invalid_idx) == 0:
        return rotmats

    result = rotmats.copy()
    for i in invalid_idx:
        left = valid_idx[valid_idx < i]
        right = valid_idx[valid_idx > i]
        if len(left) == 0 or len(right) == 0:
            continue
        l, r = left[-1], right[0]
        t = (i - l) / (r - l)
        r_l = Rotation.from_matrix(rotmats[l])
        r_r = Rotation.from_matrix(rotmats[r])
        slerp = Slerp([0, 1], Rotation.concatenate([r_l, r_r]))
        result[i] = slerp(t).as_matrix()
    return result


def interpolate_invalid(transl, global_orient, hand_pose, valid_mask):
    """Interpolate invalid frames: linear for transl, SLERP for rotations.

    Args:
        transl: (N, 3) translation
        global_orient: (N, 3, 3) rotation matrices
        hand_pose: (N, J, 3, 3) finger joint rotations
        valid_mask: (N,) boolean, True = valid frame

    Returns:
        (transl, global_orient, hand_pose) — interpolated copies
    """
    invalid_idx = np.where(~valid_mask)[0]
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2 or len(invalid_idx) == 0:
        return transl, global_orient, hand_pose

    transl = transl.copy()
    global_orient = global_orient.copy()
    hand_pose = hand_pose.copy()

    # Linear interpolation for translation
    for ax_i in range(3):
        transl[invalid_idx, ax_i] = np.interp(
            invalid_idx, valid_idx, transl[valid_idx, ax_i]
        )

    # SLERP for global_orient
    global_orient = slerp_interp_rotmat(global_orient, valid_mask)

    # SLERP for hand_pose (per joint)
    n_joints = hand_pose.shape[1]
    for j in range(n_joints):
        hand_pose[:, j] = slerp_interp_rotmat(hand_pose[:, j], valid_mask)

    return transl, global_orient, hand_pose


def clean_hand(transl, global_orient, hand_pose, fps, fast_thresh, smooth_sigma):
    """Clean a single hand: outlier removal (±1 frame) + Gaussian smooth.

    Args:
        transl: (N, 3) wrist translation
        global_orient: (N, 3, 3) wrist rotation matrices
        hand_pose: (N, J, 3, 3) finger joint rotation matrices
        fps: frames per second
        fast_thresh: outlier speed threshold (m/s)
        smooth_sigma: Gaussian sigma for translation smoothing (frames)

    Returns:
        (transl, global_orient, hand_pose) — cleaned copies
    """
    N = len(transl)
    transl = transl.copy()
    global_orient = global_orient.copy()
    hand_pose = hand_pose.copy()

    # Detect outliers by translation speed
    spd = bidir_speed(transl, fps)
    outlier = spd > fast_thresh

    # Expand ±1 frame
    expanded = outlier.copy()
    for i in np.where(outlier)[0]:
        expanded[max(0, i-1):min(N, i+2)] = True

    valid_idx = np.where(~expanded)[0]
    outlier_idx = np.where(expanded)[0]

    if len(valid_idx) > 1 and len(outlier_idx) > 0:
        valid_mask = ~expanded

        # Linear interpolation for translation
        for ax_i in range(3):
            transl[outlier_idx, ax_i] = np.interp(
                outlier_idx, valid_idx, transl[valid_idx, ax_i]
            )

        # SLERP for global_orient
        global_orient = slerp_interp_rotmat(global_orient, valid_mask)

        # SLERP for hand_pose (per joint)
        n_joints = hand_pose.shape[1]
        for j in range(n_joints):
            hand_pose[:, j] = slerp_interp_rotmat(hand_pose[:, j], valid_mask)

    # Gaussian smoothing on translation
    if smooth_sigma > 0:
        for ax_i in range(3):
            transl[:, ax_i] = gaussian_filter1d(transl[:, ax_i], sigma=smooth_sigma)

    return transl, global_orient, hand_pose
