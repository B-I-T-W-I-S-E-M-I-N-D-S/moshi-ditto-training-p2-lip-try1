"""
lip_sync_loss.py — Lip-Sync Loss Module for Ditto Training
============================================================
Encapsulates the full lip-sync loss computation pipeline:
  1. Mel spectrogram extraction from audio waveforms
  2. Lip region extraction from rendered face images
  3. Frozen SyncNet forward pass (audio + visual encoders)
  4. Loss computation (L_sync + L_stable)

All SyncNet weights are frozen — no gradients flow through the pretrained model.

Usage:
    lip_sync = LipSyncLoss(
        syncnet_path="checkpoints/lipsync_expert.pth",
        device="cuda",
    )
    loss_dict = lip_sync(
        pred_images,   # (B, 5, 3, 256, 256) — 5 consecutive rendered frames
        gt_images,     # (B, 5, 3, 256, 256) — 5 consecutive GT rendered frames
        audio_mel,     # (B, 1, 80, 16) — mel spectrogram for the window
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .syncnet import SyncNet, load_syncnet


class MelSpectrogramExtractor:
    """
    Compute mel spectrograms compatible with Wav2Lip SyncNet.

    SyncNet expects: (B, 1, 80, 16) mel spectrogram
      - 80 mel bins
      - 16 time steps ≈ 0.2s of audio at 16kHz
      - Matches 5 video frames at 25 FPS

    Parameters match Wav2Lip's audio preprocessing:
      - sample_rate=16000
      - n_fft=800 (50ms window)
      - hop_length=200 (12.5ms hop → 80 fps mel rate)
      - n_mels=80
      - For 5 frames at 25fps = 0.2s → 16 mel frames at 80 fps
    """

    def __init__(self, sample_rate=16000, n_fft=800, hop_length=200,
                 n_mels=80, device="cuda"):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device = device

        # Number of audio samples per video frame (at 25 fps)
        self.samples_per_frame = sample_rate // 25  # 640

        # Number of mel frames per video frame
        self.mel_frames_per_video_frame = self.samples_per_frame // hop_length  # ~3.2

        # Pre-build mel filter bank
        import torchaudio
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)

    def extract_mel_from_wav(self, wav_path, frame_idx, num_frames=5):
        """
        Extract mel spectrogram for a window of video frames.

        Args:
            wav_path: Path to 16kHz wav file
            frame_idx: Start frame index (0-based)
            num_frames: Number of video frames in window (default 5)

        Returns:
            (1, 80, 16) mel spectrogram tensor
        """
        import torchaudio

        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        # Calculate audio sample range for the frame window
        start_sample = frame_idx * self.samples_per_frame
        end_sample = (frame_idx + num_frames) * self.samples_per_frame

        # Ensure within bounds
        total_samples = waveform.shape[1]
        start_sample = max(0, min(start_sample, total_samples - 1))
        end_sample = min(end_sample, total_samples)

        audio_segment = waveform[:, start_sample:end_sample].to(self.device)

        # Pad if too short
        expected_length = num_frames * self.samples_per_frame
        if audio_segment.shape[1] < expected_length:
            audio_segment = F.pad(audio_segment,
                                  (0, expected_length - audio_segment.shape[1]))

        mel = self.mel_transform(audio_segment)  # (1, 80, T_mel)

        # Take exactly 16 mel frames (matching SyncNet input)
        if mel.shape[2] >= 16:
            mel = mel[:, :, :16]
        else:
            mel = F.pad(mel, (0, 16 - mel.shape[2]))

        # Log mel
        mel = torch.log(mel.clamp(min=1e-5))

        return mel  # (1, 80, 16)

    @torch.no_grad()
    def precompute_full_mel(self, wav_path):
        """
        Precompute mel spectrogram for an entire audio file.
        Returns the full mel that can be sliced per-window during training.

        Args:
            wav_path: Path to wav file

        Returns:
            (80, T_mel) full mel spectrogram tensor on CPU
        """
        import torchaudio

        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        waveform = waveform.to(self.device)
        mel = self.mel_transform(waveform)  # (1, 80, T_mel)
        mel = torch.log(mel.clamp(min=1e-5))

        return mel.squeeze(0).cpu()  # (80, T_mel)

    def slice_mel_for_window(self, full_mel, frame_idx, num_frames=5):
        """
        Slice a pre-computed mel spectrogram for a video frame window.

        Args:
            full_mel: (80, T_mel) full mel spectrogram
            frame_idx: Start frame index (0-based)
            num_frames: Number of video frames in window

        Returns:
            (1, 80, 16) mel spectrogram tensor on self.device
        """
        # Map video frame indices to mel frame indices
        # At 25fps video, 80fps mel → 3.2 mel frames per video frame
        mel_fps = self.sample_rate / self.hop_length  # 80
        video_fps = 25.0
        mel_per_video = mel_fps / video_fps  # 3.2

        mel_start = int(frame_idx * mel_per_video)
        mel_end = mel_start + 16

        T_mel = full_mel.shape[1]
        mel_start = max(0, min(mel_start, T_mel - 1))
        mel_end = min(mel_end, T_mel)

        mel_slice = full_mel[:, mel_start:mel_end]

        if mel_slice.shape[1] < 16:
            mel_slice = F.pad(mel_slice, (0, 16 - mel_slice.shape[1]))

        return mel_slice.unsqueeze(0).to(self.device)  # (1, 80, 16)


def extract_lip_region(images, lip_size=(48, 96)):
    """
    Extract lip (lower-half face) region from rendered face images.

    The renderer outputs 256×256 or 512×512 centered face crops.
    The lower half contains the mouth region.

    Args:
        images: (B, 3, H, W) rendered face images, values in [0, 1]
        lip_size: (height, width) of output lip crop — (48, 96) for SyncNet

    Returns:
        (B, 3, 48, 96) lip region crops
    """
    H, W = images.shape[2], images.shape[3]

    # Use lower 40% of face as mouth region (empirical for centered LivePortrait crops)
    top = int(H * 0.55)
    bottom = int(H * 0.95)
    left = int(W * 0.15)
    right = int(W * 0.85)

    lip_crop = images[:, :, top:bottom, left:right]

    # Resize to SyncNet expected size
    lip_crop = F.interpolate(lip_crop, size=lip_size, mode='bilinear',
                             align_corners=False)

    return lip_crop


def prepare_syncnet_visual_input(lip_frames):
    """
    Stack consecutive lip frames for SyncNet visual input.

    SyncNet expects 5 consecutive lip frames stacked in the channel dimension.

    Args:
        lip_frames: (B, 5, 3, 48, 96) — 5 consecutive lip crops

    Returns:
        (B, 15, 48, 96) — channel-stacked lip frames
    """
    B, T, C, H, W = lip_frames.shape
    assert T == 5, f"Expected 5 frames, got {T}"
    assert C == 3, f"Expected 3 channels, got {C}"

    # Stack along channel dimension: (B, 5, 3, H, W) → (B, 15, H, W)
    return lip_frames.reshape(B, T * C, H, W)


class LipSyncLoss(nn.Module):
    """
    Complete lip-sync loss module.

    Computes two losses:
      - L_sync = (1 - cosine_sim(A, V_pred)).mean()
      - L_stable = |cosine_sim(A, V_gt) - cosine_sim(A, V_pred)|.mean()

    All SyncNet weights are frozen. The mel extractor is deterministic.
    """

    def __init__(self, syncnet_path, device="cuda",
                 sync_weight=1.0, stable_weight=0.5):
        super().__init__()
        self.device = device
        self.sync_weight = sync_weight
        self.stable_weight = stable_weight

        # Load frozen SyncNet
        self.syncnet = load_syncnet(syncnet_path, device)

        # Mel spectrogram extractor
        self.mel_extractor = MelSpectrogramExtractor(device=device)

        # Small epsilon for numerical stability
        self.eps = 1e-8

    def compute_sync_loss(self, pred_lip_frames, gt_lip_frames, mel_specs):
        """
        Compute lip-sync and stabilized sync losses.

        Args:
            pred_lip_frames: (B, 5, 3, 48, 96) predicted lip frames
            gt_lip_frames:   (B, 5, 3, 48, 96) ground-truth lip frames
            mel_specs:       (B, 1, 80, 16) mel spectrograms

        Returns:
            loss_dict with: lip_sync, lip_stable, sim_pred, sim_gt
        """
        # Prepare SyncNet inputs
        pred_visual = prepare_syncnet_visual_input(pred_lip_frames)  # (B, 15, 48, 96)
        gt_visual = prepare_syncnet_visual_input(gt_lip_frames)      # (B, 15, 48, 96)

        with torch.no_grad():
            # Forward through frozen SyncNet
            A = self.syncnet.forward_audio(mel_specs)         # (B, 512)
            V_pred = self.syncnet.forward_visual(pred_visual) # (B, 512)
            V_gt = self.syncnet.forward_visual(gt_visual)     # (B, 512)

        # Cosine similarity (already L2-normalized by SyncNet)
        sim_pred = F.cosine_similarity(A, V_pred, dim=1, eps=self.eps)  # (B,)
        sim_gt = F.cosine_similarity(A, V_gt, dim=1, eps=self.eps)      # (B,)

        # L_sync = 1 - sim_pred
        loss_sync = (1.0 - sim_pred).mean()

        # L_stable = |sim_gt - sim_pred|
        loss_stable = torch.abs(sim_gt - sim_pred).mean()

        loss_dict = {
            'lip_sync': loss_sync,
            'lip_stable': loss_stable,
            'sim_pred': sim_pred.mean().detach(),
            'sim_gt': sim_gt.mean().detach(),
        }

        return loss_dict

    def forward(self, pred_lip_frames, gt_lip_frames, mel_specs):
        """
        Compute weighted total lip-sync loss.

        Returns:
            total_loss: scalar loss value
            loss_dict: detailed loss components for logging
        """
        loss_dict = self.compute_sync_loss(pred_lip_frames, gt_lip_frames, mel_specs)

        total_loss = (
            self.sync_weight * loss_dict['lip_sync'] +
            self.stable_weight * loss_dict['lip_stable']
        )

        loss_dict['lip_sync_total'] = total_loss

        return total_loss, loss_dict
