"""
frozen_renderer.py — Frozen LivePortrait Renderer for Training
==============================================================
Wraps the LivePortrait WarpingNetwork + SPADEDecoder for use during training.
All weights are frozen — no gradients flow through the renderer itself.
However, gradients DO flow through the input keypoints, allowing the
diffusion model to learn from pixel-level losses.

Pipeline:
    motion_vector (265-dim) → kp_info dict → transform_keypoint → x_d (21,3)
    f_s + x_s + x_d → WarpingNetwork → warped feature
    warped feature → SPADEDecoder → image (3, H, W)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def motion_arr_to_kp_info(motion, source_kp):
    """
    Convert 265-dim motion array to LivePortrait kp_info dict.

    The motion vector layout (from extract_motion_feat_by_LP.py):
        [0:1]     scale (stored as scale-1, so we add 1 back)
        [1:67]    pitch (66 bins)
        [67:133]  yaw (66 bins)
        [133:199] roll (66 bins)
        [199:202] t (tx, ty, tz)
        [202:265] exp (63 = 21 keypoints × 3)

    Args:
        motion: (B, 265) or (B, L, 265) tensor
        source_kp: (B, 63) canonical keypoints from source

    Returns:
        dict with {scale, pitch, yaw, roll, t, exp, kp} — all (B, dim) tensors
    """
    if motion.dim() == 3:
        # (B, L, 265) → take a single frame
        raise ValueError("Pass single-frame motion (B, 265), not sequence")

    kp_info = {
        'scale': motion[:, 0:1] + 1.0,       # (B, 1) — undo the scale-1 storage
        'pitch': motion[:, 1:67],              # (B, 66)
        'yaw': motion[:, 67:133],              # (B, 66)
        'roll': motion[:, 133:199],            # (B, 66)
        't': motion[:, 199:202],               # (B, 3)
        'exp': motion[:, 202:265],             # (B, 63)
        'kp': source_kp,                       # (B, 63)
    }
    return kp_info


def bin66_to_degree(pred):
    """Convert 66-bin classification to degree value (differentiable)."""
    if pred.dim() > 1 and pred.shape[-1] == 66:
        idx = torch.arange(66, device=pred.device, dtype=pred.dtype)
        pred_soft = torch.softmax(pred, dim=-1)
        degree = (pred_soft * idx).sum(dim=-1) * 3 - 97.5
        return degree
    return pred


def get_rotation_matrix_torch(pitch_, yaw_, roll_):
    """
    Compute rotation matrix from pitch, yaw, roll in degrees.
    Fully differentiable PyTorch implementation.

    Args:
        pitch_, yaw_, roll_: (B,) tensors in degrees

    Returns:
        (B, 3, 3) rotation matrix
    """
    pitch = pitch_ / 180.0 * np.pi
    yaw = yaw_ / 180.0 * np.pi
    roll = roll_ / 180.0 * np.pi

    if pitch.dim() == 0:
        pitch = pitch.unsqueeze(0)
    if yaw.dim() == 0:
        yaw = yaw.unsqueeze(0)
    if roll.dim() == 0:
        roll = roll.unsqueeze(0)

    bs = pitch.shape[0]
    ones = torch.ones(bs, 1, device=pitch.device, dtype=pitch.dtype)
    zeros = torch.zeros(bs, 1, device=pitch.device, dtype=pitch.dtype)

    x, y, z = pitch.unsqueeze(1), yaw.unsqueeze(1), roll.unsqueeze(1)

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape(bs, 3, 3)

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape(bs, 3, 3)

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape(bs, 3, 3)

    rot = torch.bmm(torch.bmm(rot_z, rot_y), rot_x)
    return rot.permute(0, 2, 1)  # transpose


def transform_keypoint_torch(kp_info):
    """
    Transform implicit keypoints with pose, shift, and expression deformation.
    Fully differentiable PyTorch implementation.

    Args:
        kp_info: dict with {kp, pitch, yaw, roll, t, exp, scale}

    Returns:
        (B, 21, 3) transformed keypoints
    """
    kp = kp_info['kp']  # (B, 63) or (B, 21, 3)
    pitch = bin66_to_degree(kp_info['pitch'])
    yaw = bin66_to_degree(kp_info['yaw'])
    roll = bin66_to_degree(kp_info['roll'])
    t = kp_info['t']       # (B, 3)
    exp = kp_info['exp']   # (B, 63)
    scale = kp_info['scale']  # (B, 1)

    bs = kp.shape[0]
    num_kp = 21

    if kp.dim() == 2:
        kp = kp.reshape(bs, num_kp, 3)
    exp = exp.reshape(bs, num_kp, 3)

    rot_mat = get_rotation_matrix_torch(pitch, yaw, roll)  # (B, 3, 3)

    # s * (R * x_c + exp) + t
    kp_transformed = torch.bmm(kp, rot_mat) + exp  # (B, 21, 3)
    kp_transformed = kp_transformed * scale.unsqueeze(-1)  # (B, 21, 3)
    kp_transformed[:, :, 0:2] = kp_transformed[:, :, 0:2] + t[:, None, 0:2]

    return kp_transformed


class FrozenRenderer(nn.Module):
    """
    Frozen LivePortrait renderer for training-time image generation.

    Loads WarpingNetwork and SPADEDecoder from PyTorch checkpoints.
    All weights are frozen. Gradients flow through input keypoints only.

    Usage:
        renderer = FrozenRenderer(ditto_pytorch_path, device="cuda")
        images = renderer.render(f_s, x_s, x_d)
        # images: (B, 3, 512, 512)
    """

    def __init__(self, ditto_pytorch_path, device="cuda"):
        super().__init__()
        self.device = device

        # Import model classes
        import sys
        ditto_inference_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ))),
            "ditto-inference"
        )
        if ditto_inference_dir not in sys.path:
            sys.path.insert(0, ditto_inference_dir)

        from core.models.modules.warping_network import WarpingNetwork
        from core.models.modules.spade_generator import SPADEDecoder

        # ── Load WarpingNetwork ───────────────────────────────────────────
        warp_path = os.path.join(ditto_pytorch_path, "warping_module.pt")
        self.warp_net = WarpingNetwork()
        self.warp_net.load_model(warp_path).to(device)
        self.warp_net.eval()
        for p in self.warp_net.parameters():
            p.requires_grad = False

        # ── Load SPADEDecoder ─────────────────────────────────────────────
        decoder_path = os.path.join(ditto_pytorch_path, "spade_generator.pt")
        self.decoder = SPADEDecoder()
        self.decoder.load_model(decoder_path).to(device)
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

        total_params = (
            sum(p.numel() for p in self.warp_net.parameters()) +
            sum(p.numel() for p in self.decoder.parameters())
        )
        print(f"[FrozenRenderer] Loaded WarpingNetwork + SPADEDecoder "
              f"({total_params:,} params, all frozen)")

    def forward(self, f_s, x_s, x_d):
        """
        Render image from appearance features and keypoints.

        Args:
            f_s: (B, 32, 16, 64, 64) source appearance features
            x_s: (B, 21, 3) source keypoints (transformed)
            x_d: (B, 21, 3) driving keypoints (transformed)

        Returns:
            (B, 3, H, W) rendered image, values in [0, 1]
        """
        with torch.no_grad():
            # Warp features — no grad through renderer weights
            warped = self.warp_net(f_s, x_s, x_d)   # (B, 256, 64, 64)
            # Decode to image
            image = self.decoder(warped)              # (B, 3, H, W), sigmoid output
        return image

    def render_from_motion(self, f_s, x_s_kp, motion, source_kp):
        """
        Full pipeline: motion vector → keypoints → render.

        Args:
            f_s: (B, 32, 16, 64, 64) source appearance features
            x_s_kp: (B, 21, 3) source keypoints (pre-transformed)
            motion: (B, 265) motion vector
            source_kp: (B, 63) source canonical keypoints

        Returns:
            (B, 3, H, W) rendered image
        """
        # Convert motion to kp_info (this part IS differentiable)
        kp_info = motion_arr_to_kp_info(motion, source_kp)

        # Transform keypoints (differentiable)
        x_d = transform_keypoint_torch(kp_info)  # (B, 21, 3)

        # Render (frozen, no grad through renderer)
        return self.forward(f_s, x_s_kp, x_d)
