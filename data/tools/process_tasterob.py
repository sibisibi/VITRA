import argparse
import os
from os.path import join, dirname, exists

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from data.tools.hand_recon_core import Config, HandReconstructor
from data.tools.postprocess import find_active_range, interpolate_invalid, clean_hand
from libs.models.mano_wrapper import MANO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--index_path", required=True,
                        help="Path to index.csv")
    parser.add_argument("--mano_model_dir", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply post-processing (trim, interpolate, outlier removal, smooth)")
    parser.add_argument("--visualize", action="store_true",
                        help="Render MANO mesh overlay on source video")
    parser.add_argument("--slow_thresh", type=float, default=0.01,
                        help="Stationary speed threshold (m/s) for trimming")
    parser.add_argument("--fast_thresh", type=float, default=3.0,
                        help="Outlier speed threshold (m/s) for removal")
    parser.add_argument("--smooth_sigma", type=float, default=2.0,
                        help="Gaussian smoothing sigma (frames)")
    parser.add_argument("--moge_model_path", type=str, default=None,
                        help="Local path to MoGe model.pt (skip HF download)")
    parser.add_argument("--save_all", action="store_true",
                        help="Save all MoGe outputs (depth, points, mask, normal) as .npz")
    return parser.parse_args()


def extract_frames(video_path):
    """Extract all frames from a video file as BGR numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


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

    # Set identity for rotation matrices on missing frames
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

    # Use median beta across valid frames
    beta = np.median(np.stack(beta_list), axis=0).astype(np.float64)

    return {
        "beta": beta,
        "hand_pose": hand_pose,
        "global_orient": global_orient,
        "transl": transl,
        "kept_frames": kept_frames,
    }


def compute_joints(mano_model, dense, hand_type, device):
    """Run MANO forward pass to get joint positions in camera space.

    Args:
        mano_model: MANO model instance (right hand).
        dense: dict from sparse_to_dense.
        hand_type: 'left' or 'right'.
        device: torch device.

    Returns:
        joints: (N, 21, 3) float32 joint positions in camera space.
        wrist: (N, 1, 3) float32 wrist offset from MANO canonical.
    """
    N = len(dense["kept_frames"])
    hand_pose = dense["hand_pose"]          # (N, 15, 3, 3)
    beta = dense["beta"]                    # (10,)
    global_orient = dense["global_orient"]  # (N, 3, 3)
    transl = dense["transl"]                # (N, 3)

    all_joints = []
    all_wrist = []

    with torch.no_grad():
        for i in range(N):
            hp = torch.tensor(hand_pose[i:i+1], dtype=torch.float32).to(device)
            b = torch.tensor(beta, dtype=torch.float32).unsqueeze(0).to(device)

            out = mano_model(betas=b, hand_pose=hp)
            joints_m = out.joints[0].cpu()  # (21, 3)

            # X-flip for left hand (VITRA convention: right MANO for both)
            if hand_type == "left":
                joints_m[:, 0] *= -1

            wrist_m = joints_m[0:1].clone()  # (1, 3)

            # World transform: R @ (j - j0) + T
            R = torch.tensor(global_orient[i], dtype=torch.float32)  # (3, 3)
            T = torch.tensor(transl[i], dtype=torch.float32)         # (3,)

            joints_world = (R @ (joints_m - wrist_m).T).T + T  # (21, 3)

            all_joints.append(joints_world.numpy())
            all_wrist.append(wrist_m.numpy())

    joints = np.stack(all_joints).astype(np.float32)  # (N, 21, 3)
    wrist = np.stack(all_wrist).astype(np.float32)     # (N, 1, 3)
    return joints, wrist


def build_empty_hand(N):
    """Build an empty hand dict for a hand with no detections."""
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
    """Build a VITRA-1M-format hand dict from dense arrays and computed joints."""
    # For TasteRob: single camera, no extrinsics → camspace == worldspace
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



# ---------------------------------------------------------------------------
#  Post-processing: trim, interpolate invalid, outlier removal, smooth
# ---------------------------------------------------------------------------

def postprocess_episode(episode, fps, slow_thresh, fast_thresh, smooth_sigma):
    """Apply post-processing to an episode dict in-place.

    1. Trim leading/trailing stationary frames (if ANY hand is slow).
    2. Interpolate invalid (kept_frames=False) interior frames.
    3. Remove speed outliers + Gaussian smoothing.

    Also trims per-frame metadata (video_decode_frame, extrinsics, etc.).

    Returns the episode dict, or None if no active frames remain.
    """
    # Collect active hands
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

    # Trim per-frame metadata
    for key in ["video_decode_frame", "extrinsics", "video_clip_id_segment"]:
        if key in episode:
            episode[key] = episode[key][sl]

    # Trim + clean each hand
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
            # Keep camspace in sync (TasteRob: cam == world)
            hd["transl_camspace"] = hd["transl_worldspace"].copy()
            hd["global_orient_camspace"] = hd["global_orient_worldspace"].copy()

    return episode


# ---------------------------------------------------------------------------
#  Visualization: MANO mesh overlay on source video
# ---------------------------------------------------------------------------

def render_overlay_video(episode, video_path, overlay_video_path, mano_model, device):
    """Render MANO mesh overlay on source video frames and save as mp4.

    Uses VITRA's PyTorch3D-based Renderer and the same MANO forward pass
    logic as visualize_core.process_single_hand_labels.
    """
    from visualization.render_utils import Renderer
    from visualization.video_utils import save_to_video
    import imageio

    # Load video frames for the episode's frame range
    video_decode_frames = episode["video_decode_frame"]
    start_frame = int(video_decode_frames[0])
    end_frame = int(video_decode_frames[-1]) + 1

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    T = len(episode["left"]["kept_frames"])
    if len(frames) < T:
        print(f"  Warning: video has {len(frames)} frames but episode has {T}")
        T = min(T, len(frames))

    # Setup renderer
    intrinsics = episode["intrinsics"]
    H, W = frames[0].shape[:2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    renderer = Renderer(W, H, (fx, fy), device)

    # Get MANO faces
    faces_right = torch.from_numpy(mano_model.faces).float().to(device)
    faces_left = faces_right[:, [0, 2, 1]]

    # Colors
    LEFT_COLOR = np.array([0.6594, 0.6259, 0.7451])
    RIGHT_COLOR = np.array([0.4078, 0.4980, 0.7451])

    # Compute world-space vertices for each hand
    hand_verts = {}
    hand_masks = {}
    for hand_key in ["left", "right"]:
        hd = episode[hand_key]
        kept = hd["kept_frames"].astype(bool)
        hand_masks[hand_key] = kept

        if not kept.any():
            hand_verts[hand_key] = np.zeros((T, 778, 3), dtype=np.float32)
            continue

        # MANO forward pass (batched)
        beta = torch.from_numpy(hd["beta"]).float().to(device).unsqueeze(0).repeat(T, 1)
        pose = torch.from_numpy(hd["hand_pose"]).float().to(device)
        global_rot_placeholder = torch.eye(3).float().to(device).unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)

        with torch.no_grad():
            mano_out = mano_model(betas=beta, hand_pose=pose, global_orient=global_rot_placeholder)

        verts = mano_out.vertices.cpu().numpy()  # (T, 778, 3)
        joints = mano_out.joints.cpu().numpy()    # (T, 21, 3)

        is_left = (hand_key == "left")
        if is_left:
            verts[:, :, 0] *= -1
            joints[:, :, 0] *= -1

        # World transform: R @ (V - J0) + T
        wrist_orient = hd["global_orient_worldspace"]  # (T, 3, 3)
        wrist_transl = hd["transl_worldspace"].reshape(-1, 1, 3)  # (T, 1, 3)
        verts_world = (
            wrist_orient @
            (verts - joints[:, 0][:, None]).transpose(0, 2, 1)
        ).transpose(0, 2, 1) + wrist_transl

        hand_verts[hand_key] = verts_world.astype(np.float32)

    # Camera transform (TasteRob: identity extrinsics, so camspace == worldspace)
    R_w2c = episode["extrinsics"][:T, :3, :3]  # (T, 3, 3)
    t_w2c = episode["extrinsics"][:T, :3, 3:]  # (T, 3, 1)

    # Render each frame
    overlay_frames = []
    for i in range(T):
        img = frames[i].copy().astype(np.float32) / 255.0

        verts_list = []
        faces_list = []
        colors_list = []

        for hand_key in ["left", "right"]:
            if not hand_masks[hand_key][i]:
                continue

            v = hand_verts[hand_key][i]  # (778, 3)
            # World → camera
            v_cam = (R_w2c[i] @ v.T + t_w2c[i]).T  # (778, 3)

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

    # Save
    os.makedirs(dirname(overlay_video_path), exist_ok=True)
    save_to_video(overlay_frames, overlay_video_path, fps=30)
    print(f"  Overlay saved: {overlay_video_path}")


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def process_video(args, row, idx, reconstructor, mano_model, device):
    """Process a single TasteRob video and save VITRA-1M episode .npy."""
    dataset = row["dataset"]
    filename = row["filename"]
    hand_side = row["hand_side"]

    video_path = join(args.data_dir, f"{filename}.mp4")
    if not exists(video_path):
        print(f"Video not found: {video_path}")
        return

    print(f"Processing {filename} ...")

    # 1. Extract frames
    frames, fps = extract_frames(video_path)
    N = len(frames)
    if N == 0:
        print(f"No frames extracted from {video_path}")
        return

    H, W = frames[0].shape[:2]
    print(f"  Frames: {N}, Resolution: {W}x{H}, FPS: {fps}")


    # 2. Run HandReconstructor (MoGe FoV + HaWoR hand pose)
    recon_results = reconstructor.recon(frames)
    fov_x = recon_results["fov_x"]

    # Optional: save MoGe perceptual outputs as stacked .npz
    if args.save_all:
        moge_outputs = recon_results["moge"]  # list of N dicts: points, depth, mask, normal
        moge_save_path = join(args.data_dir, "moge", f"{filename}.npz")
        os.makedirs(dirname(moge_save_path), exist_ok=True)
        np.savez_compressed(
            moge_save_path,
            depth=np.stack([m["depth"]  for m in moge_outputs]),   # (N, H, W)
            points=np.stack([m["points"] for m in moge_outputs]),   # (N, H, W, 3)
            mask=np.stack([m["mask"]   for m in moge_outputs]),     # (N, H, W)
        )
        print(f"  MoGe saved: {moge_save_path}")

    # Compute intrinsics from FoV
    fx = 0.5 * W / np.tan(0.5 * fov_x * np.pi / 180)
    intrinsics = np.array([
        [fx, 0, W / 2.0],
        [0, fx, H / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)

    # 3. Convert sparse results to dense arrays + compute joints
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

    # 4. Build text annotations (empty — captions live in the CSV, not in the .npy)
    text = {"left": [], "right": []}
    text_rephrase = {"left": [], "right": []}

    # 5. Build episode dict (VITRA-1M format)
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
        "text": text,
        "text_rephrase": text_rephrase,
        "left": hand_dicts["left"],
        "right": hand_dicts["right"],
    }

    # 6. Optional post-processing
    if args.postprocess:
        print(f"  Post-processing (slow={args.slow_thresh}, fast={args.fast_thresh}, sigma={args.smooth_sigma})")
        episode = postprocess_episode(episode, fps, args.slow_thresh, args.fast_thresh, args.smooth_sigma)
        if episode is None:
            print(f"  Skipping {filename}: no active frames after trimming")
            return
        N_new = len(episode["video_decode_frame"])
        print(f"  Trimmed: {N} → {N_new} frames")

    # 7. Save
    save_path = join(args.data_dir, "episodic_annotations", f"{filename}.npy")
    os.makedirs(dirname(save_path), exist_ok=True)
    np.save(save_path, episode)
    print(f"  Saved: {save_path}")

    # 8. Optional visualization
    if args.visualize:
        overlay_video_path = join(args.data_dir, "overlay", f"{filename}.mp4")
        if args.postprocess:
            overlay_video_path = overlay_video_path.replace(".mp4", "_postprocessed.mp4")
        print(f"  Rendering overlay video...")
        render_overlay_video(episode, video_path, overlay_video_path, mano_model, device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load index
    df = pd.read_csv(args.index_path)
    row = df.iloc[args.index]
    print(f"Index {args.index}: dataset={row['dataset']}, filename={row['filename']}, hand_side={row['hand_side']}")


    # Initialize HandReconstructor
    config = Config(args)
    config.MANO_PATH = join(args.mano_model_dir, "mano")
    reconstructor = HandReconstructor(config, device=device)

    # Initialize MANO model for joint computation
    mano_model = MANO(model_path=config.MANO_PATH).to(device)
    mano_model.eval()

    process_video(args, row, args.index, reconstructor, mano_model, device)


if __name__ == "__main__":
    main()
