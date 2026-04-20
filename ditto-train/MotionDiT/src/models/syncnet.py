"""
syncnet.py — Wav2Lip SyncNet Architecture for Lip-Sync Loss
=============================================================
Pretrained SyncNet with audio and visual encoders.
All weights are FROZEN — used only for computing lip-sync loss during training.

Architecture matches the Wav2Lip SyncNet_color model:
  - Visual encoder: (B, 15, 48, 96) → 512-d embedding
    Input: 5 consecutive lower-half face frames stacked in channel dim (5*3=15)
  - Audio encoder: (B, 1, 80, 16) → 512-d embedding
    Input: mel spectrogram (80 mel bins, 16 time steps ≈ 0.2s)
  - Cosine similarity between embeddings

Reference: https://github.com/Rudrabha/Wav2Lip
"""

import torch
import torch.nn as nn


class Conv2dBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU block used in SyncNet."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class SyncNet(nn.Module):
    """
    Wav2Lip SyncNet_color architecture.

    Visual input:  (B, 15, 48, 96) — 5 frames × 3 channels, lower half of 96×96
    Audio input:   (B, 1, 80, 16)  — mel spectrogram
    Output:        cosine similarity between audio and visual embeddings
    """

    def __init__(self):
        super().__init__()

        # ── Visual Encoder ────────────────────────────────────────────────
        self.face_encoder = nn.Sequential(
            Conv2dBlock(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2dBlock(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2dBlock(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBlock(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBlock(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2dBlock(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBlock(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBlock(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBlock(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2dBlock(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBlock(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBlock(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2dBlock(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBlock(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBlock(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2dBlock(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2dBlock(512, 512, kernel_size=1, stride=1, padding=0),
        )

        # ── Audio Encoder ─────────────────────────────────────────────────
        self.audio_encoder = nn.Sequential(
            Conv2dBlock(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2dBlock(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBlock(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBlock(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2dBlock(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBlock(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBlock(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2dBlock(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBlock(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBlock(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2dBlock(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBlock(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBlock(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2dBlock(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward_visual(self, face_sequences):
        """
        Encode visual (lip) input.

        Args:
            face_sequences: (B, 15, 48, 96) — 5 frames × 3 channels, lower half

        Returns:
            (B, 512) L2-normalized visual embedding
        """
        out = self.face_encoder(face_sequences)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out

    def forward_audio(self, audio_sequences):
        """
        Encode audio (mel spectrogram) input.

        Args:
            audio_sequences: (B, 1, 80, 16) — mel spectrogram

        Returns:
            (B, 512) L2-normalized audio embedding
        """
        out = self.audio_encoder(audio_sequences)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out

    def forward(self, audio_sequences, face_sequences):
        """
        Compute audio and visual embeddings.

        Returns:
            audio_embedding: (B, 512)
            visual_embedding: (B, 512)
        """
        audio_embedding = self.forward_audio(audio_sequences)
        face_embedding = self.forward_visual(face_sequences)
        return audio_embedding, face_embedding


def load_syncnet(checkpoint_path, device="cuda"):
    """
    Load pretrained SyncNet and freeze all weights.

    Args:
        checkpoint_path: Path to lipsync_expert.pth (Wav2Lip checkpoint)
        device: Target device

    Returns:
        Frozen SyncNet model in eval mode
    """
    model = SyncNet()

    # Load checkpoint — Wav2Lip saves as {"state_dict": ..., "global_step": ..., ...}
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model = model.to(device)

    # Freeze everything
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"[SyncNet] Loaded pretrained model from {checkpoint_path} "
          f"({sum(p.numel() for p in model.parameters())} params, all frozen)")

    return model
