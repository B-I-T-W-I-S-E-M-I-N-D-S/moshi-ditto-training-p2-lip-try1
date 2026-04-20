"""
preprocess_source_features.py — Extract Source Appearance & Keypoint Features
==============================================================================
Preprocessing step that extracts per-video:
  - f_s: Source appearance features (32, 16, 64, 64) from first frame
  - x_s_info: Source keypoint info (scale, pitch, yaw, roll, t, exp, kp)

These features are stored as .npy files and used during training for the
lip-sync loss (which requires rendering predicted motion to images).

Usage:
    python preprocess_source_features.py \
        -i /workspace/HDTF/data_info.json \
        --ditto_pytorch_path ditto-train/checkpoints/ditto_pytorch \
        --device cuda

This script reads the 'video_list' from data_info.json, extracts features
from the first frame of each cropped video, and saves them to:
  - {save_dir}/f_s_npy/{name}.npy          — (32, 16, 64, 64) float16
  - {save_dir}/x_s_info_npy/{name}.npy     — (328,) float32 [full kp_info flattened]
  - {save_dir}/x_s_kp_npy/{name}.npy       — (21, 3) float32 [transformed source kp]

The paths are added to data_info.json under:
  - f_s_npy_list
  - x_s_info_npy_list
  - x_s_kp_npy_list
"""

import os
import sys
import json
import traceback
import numpy as np
import cv2
from dataclasses import dataclass
from typing_extensions import Annotated
from tqdm import tqdm

import torch

# Ensure project imports work
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def dump_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def extract_first_frame(video_path):
    """Extract first frame from a video file as RGB uint8 numpy array."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read video: {video_path}")
    # BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def frame_to_256_bchw(frame):
    """Convert RGB uint8 frame to (1, 3, 256, 256) float32 normalized [0,1]."""
    h, w = frame.shape[:2]
    # Resize to 256x256
    frame_256 = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
    # Normalize and convert to BCHW
    frame_bchw = (frame_256.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2)
    return frame_bchw  # (1, 3, 256, 256)


def main():
    import tyro

    @dataclass
    class Options:
        input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""
        ditto_pytorch_path: str = ""
        device: str = "cuda"
        skip_existing: bool = True

    tyro.extras.set_accent_color("bright_cyan")
    opt = tyro.cli(Options)
    assert opt.input_data_json, "Must provide --input_data_json / -i"
    assert opt.ditto_pytorch_path, "Must provide --ditto_pytorch_path"

    data_info = load_json(opt.input_data_json)
    video_list = data_info['video_list']
    save_dir = os.path.dirname(opt.input_data_json)

    # Import model classes
    ditto_inference_dir = os.path.join(
        os.path.dirname(CUR_DIR), "ditto-inference"
    )
    sys.path.insert(0, ditto_inference_dir)

    from core.models.modules.appearance_feature_extractor import AppearanceFeatureExtractor
    from core.models.modules.motion_extractor import MotionExtractor as MEModule

    # Load models
    print("[preprocess_source_features] Loading AppearanceFeatureExtractor...")
    app_path = os.path.join(opt.ditto_pytorch_path, "appearance_feature_extractor.pt")
    app_model = AppearanceFeatureExtractor()
    app_model.load_model(app_path).to(opt.device)
    app_model.eval()

    print("[preprocess_source_features] Loading MotionExtractor...")
    me_path = os.path.join(opt.ditto_pytorch_path, "motion_extractor.pt")
    me_model = MEModule()
    me_model.load_model(me_path).to(opt.device)
    me_model.eval()

    # Output lists
    f_s_npy_list = []
    x_s_info_npy_list = []
    x_s_kp_npy_list = []

    # Create output directories
    f_s_dir = os.path.join(save_dir, "f_s_npy")
    x_s_info_dir = os.path.join(save_dir, "x_s_info_npy")
    x_s_kp_dir = os.path.join(save_dir, "x_s_kp_npy")
    os.makedirs(f_s_dir, exist_ok=True)
    os.makedirs(x_s_info_dir, exist_ok=True)
    os.makedirs(x_s_kp_dir, exist_ok=True)

    print(f"[preprocess_source_features] Processing {len(video_list)} videos...")

    for i, video_path in enumerate(tqdm(video_list)):
        name = os.path.basename(video_path).rsplit('.', 1)[0]

        f_s_path = os.path.join(f_s_dir, f"{name}.npy")
        x_s_info_path = os.path.join(x_s_info_dir, f"{name}.npy")
        x_s_kp_path = os.path.join(x_s_kp_dir, f"{name}.npy")

        f_s_npy_list.append(f_s_path)
        x_s_info_npy_list.append(x_s_info_path)
        x_s_kp_npy_list.append(x_s_kp_path)

        # Skip if already exists
        if opt.skip_existing and all(os.path.isfile(p) for p in
                                     [f_s_path, x_s_info_path, x_s_kp_path]):
            continue

        try:
            # Extract first frame
            frame = extract_first_frame(video_path)
            frame_bchw = frame_to_256_bchw(frame)  # (1, 3, 256, 256)
            frame_tensor = torch.from_numpy(frame_bchw).to(opt.device)

            with torch.no_grad(), torch.autocast(device_type=opt.device[:4],
                                                  dtype=torch.float16,
                                                  enabled=True):
                # Extract appearance features
                f_s = app_model(frame_tensor)  # (1, 32, 16, 64, 64)
                f_s_np = f_s.float().cpu().numpy().squeeze(0)  # (32, 16, 64, 64)

                # Extract motion/keypoint info
                kp_out = me_model(frame_tensor)
                # kp_out is a list: [pitch, yaw, roll, t, exp, scale, kp]

            output_names = ["pitch", "yaw", "roll", "t", "exp", "scale", "kp"]
            kp_info = {}
            for j, name_k in enumerate(output_names):
                kp_info[name_k] = kp_out[j].float().cpu().numpy()
            kp_info['exp'] = kp_info['exp'].reshape(1, -1)
            kp_info['kp'] = kp_info['kp'].reshape(1, -1)

            # Build full kp_info array (same format as LP_npy)
            # Layout: scale(1) + pitch(66) + yaw(66) + roll(66) + t(3) + exp(63) + kp(63) = 328
            x_s_info_arr = np.concatenate([
                (kp_info['scale'].reshape(1) - 1.0),  # store as scale-1
                kp_info['pitch'].reshape(66),
                kp_info['yaw'].reshape(66),
                kp_info['roll'].reshape(66),
                kp_info['t'].reshape(3),
                kp_info['exp'].reshape(63),
                kp_info['kp'].reshape(63),
            ]).astype(np.float32)  # (328,)

            # Compute transformed source keypoints
            from MotionDiT.src.models.frozen_renderer import (
                motion_arr_to_kp_info, transform_keypoint_torch
            )
            motion_265 = torch.from_numpy(x_s_info_arr[:265]).unsqueeze(0).to(opt.device)
            source_kp = torch.from_numpy(x_s_info_arr[265:]).unsqueeze(0).to(opt.device)
            kp_dict = motion_arr_to_kp_info(motion_265, source_kp)
            x_s_kp = transform_keypoint_torch(kp_dict)  # (1, 21, 3)
            x_s_kp_np = x_s_kp.float().cpu().numpy().squeeze(0)  # (21, 3)

            # Save
            np.save(f_s_path, f_s_np.astype(np.float16))
            np.save(x_s_info_path, x_s_info_arr)
            np.save(x_s_kp_path, x_s_kp_np)

        except Exception:
            traceback.print_exc()
            print(f"  ⚠️  Failed for video {i}: {video_path}")

    # Update data_info.json
    data_info['f_s_npy_list'] = f_s_npy_list
    data_info['x_s_info_npy_list'] = x_s_info_npy_list
    data_info['x_s_kp_npy_list'] = x_s_kp_npy_list

    dump_json(data_info, opt.input_data_json)
    print(f"✅ Source features extracted and saved. "
          f"Updated {opt.input_data_json}")


if __name__ == "__main__":
    main()
