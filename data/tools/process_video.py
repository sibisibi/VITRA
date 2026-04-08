import argparse
import copy
import glob
import json
import os
from os.path import join, dirname, exists

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from data.tools.hand_recon_core import Config
from data.tools.utils_hawor import HaworPipeline
from data.tools.postprocess import find_active_range, interpolate_invalid, clean_hand
from libs.models.mano_wrapper import MANO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Directory of JPEG frames: rgb_frames/{filename}/")
    parser.add_argument("--mano_model_dir", required=True)
    parser.add_argument("--index_path", required=True,
                        help="Path to index.csv")
    parser.add_argument("--img_focal_dir", required=True,
                        help="Directory of precomputed MoGe outputs: moge2/{filename}/")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory: hawor/{filename}/")
    parser.add_argument("--index", type=int, required=True,
                        help="Row index in index.csv to process")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply post-processing (trim, interpolate, outlier removal, smooth)")
    parser.add_argument("--visualize", action="store_true",
                        help="Render MANO mesh overlay on source frames")
    parser.add_argument("--slow_thresh", type=float, default=0.01,
                        help="Stationary speed threshold (m/s) for trimming")
    parser.add_argument("--fast_thresh", type=float, default=3.0,
                        help="Outlier speed threshold (m/s) for removal")
    parser.add_argument("--smooth_sigma", type=float, default=2.0,
                        help="Gaussian smoothing sigma (frames)")
    return parser.parse_args()


def load_frames(data_dir):
    """Load JPEG frames from directory, sorted by filename."""
    frame_paths = sorted(glob.glob(join(data_dir, "*.jpg")))
    frames = [cv2.imread(p) for p in frame_paths]
    return frames, frame_paths


def load_fov(img_focal_dir, frame_paths):
    """Read fov.json per frame, return median fov_x in degrees."""
    fov_values = []
    for p in frame_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        fov_path = join(img_focal_dir, stem, "fov.json")
        with open(fov_path) as f:
            fov_values.append(json.load(f)["fov_x"])
    return float(np.median(fov_values))


def adjust_translations(recon_results, mano_model, device, N):
    """Apply MANO wrist-alignment to HaWoR translations.

    HaWoR outputs translation relative to the MANO canonical wrist joint.
    This adds the wrist joint offset so transl represents the wrist position
    in camera space. Mirrors hand_recon_core.py lines 91-125.
    """
    adjusted = {'left': {}, 'right': {}}
    for img_idx in range(N):
        for hand_type in ['left', 'right']:
            if img_idx not in recon_results[hand_type]:
                continue
            result = recon_results[hand_type][img_idx]

            betas = torch.from_numpy(result['beta']).unsqueeze(0).to(device)
            hand_pose = torch.from_numpy(result['hand_pose']).unsqueeze(0).to(device)
            transl = torch.from_numpy(result['transl']).unsqueeze(0).to(device)

            with torch.no_grad():
                model_output = mano_model(betas=betas, hand_pose=hand_pose)
            joints_m = model_output.joints[0]

            if hand_type == 'left':
                joints_m[:, 0] *= -1

            wrist = joints_m[0]
            transl_new = wrist + transl[0]

            result_new = copy.deepcopy(result)
            result_new['transl'] = transl_new.cpu().numpy()
            adjusted[hand_type][img_idx] = result_new

    return adjusted


def sparse_to_dense(recon_results, hand_type, N):
    """Convert sparse HandReconstructor dict to dense (N, ...) arrays.

    Returns None if the hand has no detections.
    """
    hand_dict = recon_results[hand_type]
    if not hand_dict:
        return None

    kept_frames = np.zeros(N, dtype=np.int64)
    beta_list = []
    hand_pose = np.zeros((N, 15, 3, 3), dtype=np.float64)
    global_orient = np.zeros((N, 3, 3), dtype=np.float64)
    transl = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        global_orient[i] = np.eye(3)
        for j in range(15):
            hand_pose[i, j] = np.eye(3)

    for frame_idx, result in hand_dict.items():
        kept_frames[frame_idx] = 1
        beta_list.append(result["beta"])
        hand_pose[frame_idx] = result["hand_pose"]
        global_orient[frame_idx] = result["global_orient"]
        transl[frame_idx] = result["transl"]

    beta = np.median(np.stack(beta_list), axis=0).astype(np.float64)

    return {
        "beta": beta,
        "hand_pose": hand_pose,
        "global_orient": global_orient,
        "transl": transl,
        "kept_frames": kept_frames,
    }


def compute_joints(mano_model, dense, hand_type, device):
    """Run MANO forward pass to get joint positions in camera space."""
    N = len(dense["kept_frames"])
    hand_pose = dense["hand_pose"]
    beta = dense["beta"]
    global_orient = dense["global_orient"]
    transl = dense["transl"]

    all_joints = []
    all_wrist = []

    with torch.no_grad():
        for i in range(N):
            hp = torch.tensor(hand_pose[i:i+1], dtype=torch.float32).to(device)
            b = torch.tensor(beta, dtype=torch.float32).unsqueeze(0).to(device)

            out = mano_model(betas=b, hand_pose=hp)
            joints_m = out.joints[0].cpu()

            if hand_type == "left":
                joints_m[:, 0] *= -1

            wrist_m = joints_m[0:1].clone()

            R = torch.tensor(global_orient[i], dtype=torch.float32)
            T = torch.tensor(transl[i], dtype=torch.float32)

            joints_world = (R @ (joints_m - wrist_m).T).T + T

            all_joints.append(joints_world.numpy())
            all_wrist.append(wrist_m.numpy())

    joints = np.stack(all_joints).astype(np.float32)
    wrist = np.stack(all_wrist).astype(np.float32)
    return joints, wrist


def build_empty_hand(N):
    return {
        "beta": np.zeros(10, dtype=np.float64),
        "global_orient_camspace": np.tile(np.eye(3), (N, 1, 1)).astype(np.float64),
        "global_orient_worldspace": np.tile(np.eye(3), (N, 1, 1)).astype(np.float64),
        "hand_pose": np.tile(np.eye(3), (N, 15, 1, 1)).astype(np.float64),
        "transl_camspace": np.zeros((N, 3), dtype=np.float64),
        "transl_worldspace": np.zeros((N, 3), dtype=np.float64),
        "kept_frames": np.zeros(N, dtype=np.int64),
        "joints_camspace": np.zeros((N, 21, 3), dtype=np.float32),
        "joints_worldspace": np.zeros((N, 21, 3), dtype=np.float64),
        "wrist": np.zeros((N, 1, 3), dtype=np.float32),
        "max_translation_movement": None,
        "max_wrist_rotation_movement": None,
        "max_finger_joint_angle_movement": None,
    }


def build_hand_dict(dense, joints, wrist, N):
    return {
        "beta": dense["beta"],
        "global_orient_camspace": dense["global_orient"],
        "global_orient_worldspace": dense["global_orient"],
        "hand_pose": dense["hand_pose"],
        "transl_camspace": dense["transl"],
        "transl_worldspace": dense["transl"],
        "kept_frames": dense["kept_frames"],
        "joints_camspace": joints,
        "joints_worldspace": joints.astype(np.float64),
        "wrist": wrist,
        "max_translation_movement": None,
        "max_wrist_rotation_movement": None,
        "max_finger_joint_angle_movement": None,
    }


def postprocess_episode(episode, fps, slow_thresh, fast_thresh, smooth_sigma):
    """Apply post-processing: trim, interpolate, outlier removal, smooth."""
    transl_list = []
    active_keys = []
    for hand_key in ["left", "right"]:
        if episode[hand_key]["kept_frames"].astype(bool).any():
            transl_list.append(episode[hand_key]["transl_worldspace"])
            active_keys.append(hand_key)

    if not transl_list:
        return None

    trim = find_active_range(transl_list, fps, slow_thresh)
    if trim is None:
        return None
    first, last = trim
    sl = slice(first, last + 1)

    for key in ["video_decode_frame", "extrinsics", "video_clip_id_segment"]:
        if key in episode:
            episode[key] = episode[key][sl]

    for hand_key in ["left", "right"]:
        hd = episode[hand_key]
        temporal_keys = [
            "hand_pose", "global_orient_worldspace", "global_orient_camspace",
            "transl_worldspace", "transl_camspace", "kept_frames",
            "joints_camspace", "joints_worldspace", "wrist",
        ]
        for k in temporal_keys:
            if k in hd and hd[k] is not None and hasattr(hd[k], '__len__') and len(hd[k]) > last:
                hd[k] = hd[k][sl]

        if hand_key in active_keys:
            kept = hd["kept_frames"].astype(bool)
            hd["transl_worldspace"], hd["global_orient_worldspace"], hd["hand_pose"] = \
                interpolate_invalid(
                    hd["transl_worldspace"], hd["global_orient_worldspace"],
                    hd["hand_pose"], kept
                )
            hd["transl_worldspace"], hd["global_orient_worldspace"], hd["hand_pose"] = \
                clean_hand(
                    hd["transl_worldspace"], hd["global_orient_worldspace"],
                    hd["hand_pose"], fps, fast_thresh, smooth_sigma
                )
            hd["transl_camspace"] = hd["transl_worldspace"].copy()
            hd["global_orient_camspace"] = hd["global_orient_worldspace"].copy()

    return episode


def render_overlay_video(episode, all_frames, overlay_video_path, mano_model, device):
    """Render MANO mesh overlay on source frames and save as mp4."""
    from visualization.render_utils import Renderer
    from visualization.video_utils import save_to_video

    video_decode_frames = episode["video_decode_frame"]
    start_frame = int(video_decode_frames[0])
    end_frame = int(video_decode_frames[-1]) + 1

    frames = [
        cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB)
        for i in range(start_frame, min(end_frame, len(all_frames)))
    ]

    T = len(episode["left"]["kept_frames"])
    if len(frames) < T:
        print(f"  Warning: have {len(frames)} frames but episode has {T}")
        T = min(T, len(frames))

    intrinsics = episode["intrinsics"]
    H, W = frames[0].shape[:2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    renderer = Renderer(W, H, (fx, fy), device)

    faces_right = torch.from_numpy(mano_model.faces).float().to(device)
    faces_left = faces_right[:, [0, 2, 1]]

    LEFT_COLOR = np.array([0.6594, 0.6259, 0.7451])
    RIGHT_COLOR = np.array([0.4078, 0.4980, 0.7451])

    hand_verts = {}
    hand_masks = {}
    for hand_key in ["left", "right"]:
        hd = episode[hand_key]
        kept = hd["kept_frames"].astype(bool)
        hand_masks[hand_key] = kept

        if not kept.any():
            hand_verts[hand_key] = np.zeros((T, 778, 3), dtype=np.float32)
            continue

        beta = torch.from_numpy(hd["beta"]).float().to(device).unsqueeze(0).repeat(T, 1)
        pose = torch.from_numpy(hd["hand_pose"]).float().to(device)
        global_rot_placeholder = torch.eye(3).float().to(device).unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)

        with torch.no_grad():
            mano_out = mano_model(betas=beta, hand_pose=pose, global_orient=global_rot_placeholder)

        verts = mano_out.vertices.cpu().numpy()
        joints = mano_out.joints.cpu().numpy()

        is_left = (hand_key == "left")
        if is_left:
            verts[:, :, 0] *= -1
            joints[:, :, 0] *= -1

        wrist_orient = hd["global_orient_worldspace"]
        wrist_transl = hd["transl_worldspace"].reshape(-1, 1, 3)
        verts_world = (
            wrist_orient @
            (verts - joints[:, 0][:, None]).transpose(0, 2, 1)
        ).transpose(0, 2, 1) + wrist_transl

        hand_verts[hand_key] = verts_world.astype(np.float32)

    R_w2c = episode["extrinsics"][:T, :3, :3]
    t_w2c = episode["extrinsics"][:T, :3, 3:]

    overlay_frames = []
    for i in range(T):
        img = frames[i].copy().astype(np.float32) / 255.0

        verts_list = []
        faces_list = []
        colors_list = []

        for hand_key in ["left", "right"]:
            if not hand_masks[hand_key][i]:
                continue

            v = hand_verts[hand_key][i]
            v_cam = (R_w2c[i] @ v.T + t_w2c[i]).T

            color = LEFT_COLOR if hand_key == "left" else RIGHT_COLOR
            faces = faces_left if hand_key == "left" else faces_right

            verts_list.append(torch.from_numpy(v_cam).float().to(device))
            colors_list.append(torch.from_numpy(color).float().unsqueeze(0).repeat(778, 1).to(device))
            faces_list.append(faces)

        if verts_list:
            rend, mask = renderer.render(verts_list, faces_list, colors_list)
            rend_float = rend.astype(np.float32) / 255.0
            img[mask] = 0.5 * img[mask] + 0.5 * rend_float[mask]

        overlay_frames.append((img * 255).astype(np.uint8))

        if (i + 1) % 50 == 0 or i == T - 1:
            print(f"  Overlay: {i+1}/{T}", end="\r")

    print()

    os.makedirs(dirname(overlay_video_path), exist_ok=True)
    save_to_video(overlay_frames, overlay_video_path, fps=30)
    print(f"  Overlay saved: {overlay_video_path}")


def process_video(args, row, hawor_pipeline, mano_model, device):
    """Process a single video from pre-extracted frames and precomputed MoGe outputs."""
    filename = row["filename"]
    fps = float(row["fps"])

    print(f"Processing {filename} ...")

    # 1. Load JPEG frames
    rgb_dir = join(args.data_dir, filename, "rgb_frames")
    frames, frame_paths = load_frames(rgb_dir)
    N = len(frames)
    if N == 0:
        print(f"No frames found in {rgb_dir}")
        return

    H, W = frames[0].shape[:2]
    print(f"  Frames: {N}, Resolution: {W}x{H}, FPS: {fps}")

    # 2. Read precomputed FoV → compute intrinsics
    moge_dir = join(args.img_focal_dir, filename, "moge")
    fov_x = load_fov(moge_dir, frame_paths)
    fx = 0.5 * W / np.tan(0.5 * fov_x * np.pi / 180)
    intrinsics = np.array([
        [fx, 0, W / 2.0],
        [0, fx, H / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)
    print(f"  FoV: {fov_x:.2f}°, focal: {fx:.1f}px")

    # 3. Run HaWoR hand pose estimation
    print("  Running HaWoR ...")
    recon_results = hawor_pipeline.recon(frames, img_focal=fx, single_image=(N == 1))

    # 4. MANO wrist-alignment for translations
    recon_results = adjust_translations(recon_results, mano_model, device, N)

    # 5. Sparse → Dense + compute joints
    hand_dicts = {}
    anno_type = None
    for ht in ["left", "right"]:
        dense = sparse_to_dense(recon_results, ht, N)
        if dense is not None:
            joints, wrist = compute_joints(mano_model, dense, ht, device)
            hand_dicts[ht] = build_hand_dict(dense, joints, wrist, N)
            if anno_type is None:
                anno_type = ht
        else:
            hand_dicts[ht] = build_empty_hand(N)

    if anno_type is None:
        print(f"  No hands detected, skipping.")
        return

    # 6. Build episode dict (VITRA-1M format)
    episode = {
        "video_clip_id_segment": np.zeros(N, dtype=np.int64),
        "extrinsics": np.tile(np.eye(4), (N, 1, 1)).astype(np.float64),
        "intrinsics": intrinsics,
        "video_decode_frame": np.arange(N, dtype=np.int64),
        "video_name": str(filename),
        "avg_speed": 0.0,
        "total_rotvec_degree": 0.0,
        "total_transl_dist": 0.0,
        "anno_type": anno_type,
        "text": {"left": [], "right": []},
        "text_rephrase": {"left": [], "right": []},
        "left": hand_dicts["left"],
        "right": hand_dicts["right"],
    }

    # 7. Optional post-processing
    if args.postprocess:
        print(f"  Post-processing ...")
        episode = postprocess_episode(episode, fps, args.slow_thresh, args.fast_thresh, args.smooth_sigma)
        if episode is None:
            print(f"  Skipping {filename}: no active frames after trimming")
            return
        N_new = len(episode["video_decode_frame"])
        print(f"  Trimmed: {N} → {N_new} frames")

    # 8. Save episode .npy
    hawor_dir = join(args.output_dir, filename, "hawor")
    os.makedirs(hawor_dir, exist_ok=True)
    save_path = join(hawor_dir, "annotation.npy")
    np.save(save_path, episode)
    print(f"  Saved: {save_path}")

    # 9. Optional overlay video
    if args.visualize:
        overlay_path = join(hawor_dir, "overlay.mp4")
        print(f"  Rendering overlay video ...")
        render_overlay_video(episode, frames, overlay_path, mano_model, device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load index.csv
    df = pd.read_csv(args.index_path)
    row = df.iloc[args.index]
    print(f"Index {args.index}: filename={row['filename']}, hand_side={row['hand_side']}, fps={row['fps']}")

    # Check if already done
    save_path = join(args.output_dir, f"{row['filename']}.npy")
    if exists(save_path):
        print(f"Already exists: {save_path}, skipping.")
        return

    # Initialize models
    config = Config()
    config.MANO_PATH = join(args.mano_model_dir, "mano")

    hawor_pipeline = HaworPipeline(
        model_path=config.HAWOR_MODEL_PATH,
        detector_path=config.DETECTOR_PATH,
        device=device,
    )

    mano_model = MANO(model_path=config.MANO_PATH).to(device)
    mano_model.eval()

    process_video(args, row, hawor_pipeline, mano_model, device)


if __name__ == "__main__":
    main()
