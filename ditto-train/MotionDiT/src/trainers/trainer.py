import torch
import os
import time
from tqdm import trange, tqdm
import traceback
import numpy as np

from ..utils.utils import load_json, DictAverageMeter, dump_pkl
from ..models.modules.adan import Adan
from ..models.LMDM import LMDM
from ..datasets.s2_dataset_v2 import Stage2Dataset as Stage2DatasetV2
from ..options.option import TrainOptions


class Trainer:
    def __init__(self, opt: TrainOptions):
        self.opt = opt

        print(time.asctime(), '_init_accelerate')
        self._init_accelerate()

        print(time.asctime(), '_init_LMDM')
        self.LMDM = self._init_LMDM()

        print(time.asctime(), '_init_dataset')
        self.data_loader = self._init_dataset()

        print(time.asctime(), '_init_optim')
        self.optim = self._init_optim()

        print(time.asctime(), '_set_accelerate')
        self._set_accelerate()

        # Lip sync loss (after accelerate setup, on same device)
        if opt.use_lip_sync_loss:
            print(time.asctime(), '_init_lip_sync')
            self._init_lip_sync()

        print(time.asctime(), '_init_log')
        self._init_log()

    def _init_accelerate(self):
        opt = self.opt
        if opt.use_accelerate:
            from accelerate import Accelerator
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.is_main_process = self.accelerator.is_main_process
            self.process_index = self.accelerator.process_index
        else:
            self.accelerator = None
            self.device = 'cuda'
            self.is_main_process = True
            self.process_index = 0

    def _set_accelerate(self):
        if self.accelerator is None:
            return
        
        self.LMDM.use_accelerator(self.accelerator)
        self.optim = self.accelerator.prepare(self.optim)
        self.data_loader = self.accelerator.prepare(self.data_loader)

        self.accelerator.wait_for_everyone()

    def _init_LMDM(self):
        opt = self.opt

        part_w_dict = None
        if opt.part_w_dict_json:
            part_w_dict = load_json(opt.part_w_dict_json)
        dim_ws = None
        if opt.dim_ws_npy:
            dim_ws = np.load(opt.dim_ws_npy)

        lmdm = LMDM(
            motion_feat_dim=opt.motion_feat_dim,
            audio_feat_dim=opt.audio_feat_dim,
            seq_frames=opt.seq_frames,
            part_w_dict=part_w_dict,   # only for train
            checkpoint=opt.checkpoint,
            device=self.device,
            use_last_frame_loss=opt.use_last_frame_loss,
            use_reg_loss=opt.use_reg_loss,
            dim_ws=dim_ws,
        )

        return lmdm

    def _init_lip_sync(self):
        """Initialize lip sync loss components: FrozenRenderer + LipSyncLoss."""
        opt = self.opt
        device = self.device

        from ..models.frozen_renderer import FrozenRenderer
        from ..models.lip_sync_loss import LipSyncLoss

        # Frozen renderer (WarpingNetwork + SPADEDecoder)
        assert opt.ditto_pytorch_path, (
            "--ditto_pytorch_path is required when --use_lip_sync_loss is set"
        )
        self.frozen_renderer = FrozenRenderer(
            ditto_pytorch_path=opt.ditto_pytorch_path,
            device=device,
        )

        # Lip sync loss (frozen SyncNet + mel extractor)
        assert opt.syncnet_checkpoint, (
            "--syncnet_checkpoint is required when --use_lip_sync_loss is set"
        )
        self.lip_sync_loss = LipSyncLoss(
            syncnet_path=opt.syncnet_checkpoint,
            device=device,
            sync_weight=opt.lip_sync_weight,
            stable_weight=opt.lip_sync_stable_weight,
        )

        # Cache for precomputed mel spectrograms (wav_path → full mel tensor)
        self._mel_cache = {}

        print(f"[Trainer] Lip sync loss initialized: "
              f"λ_sync={opt.lip_sync_weight}, λ_stable={opt.lip_sync_stable_weight}, "
              f"every {opt.lip_sync_every_n_steps} steps, "
              f"{opt.lip_sync_num_samples} samples/step")

    def _init_dataset(self):
        opt = self.opt

        if opt.dataset_version in ['v2']:
            Stage2Dataset = Stage2DatasetV2
        else:
            raise NotImplementedError()

        dataset = Stage2Dataset(
            data_list_json=opt.data_list_json, 
            seq_len=opt.seq_frames,
            preload=opt.data_preload, 
            cache=opt.data_cache, 
            preload_pkl=opt.data_preload_pkl, 
            motion_feat_dim=opt.motion_feat_dim, 
            motion_feat_start=opt.motion_feat_start,
            motion_feat_offset_dim_se=opt.motion_feat_offset_dim_se,
            use_eye_open=opt.use_eye_open,
            use_eye_ball=opt.use_eye_ball,
            use_emo=opt.use_emo,
            use_sc=opt.use_sc,
            use_last_frame=opt.use_last_frame,
            use_lmk=opt.use_lmk,
            use_cond_end=opt.use_cond_end,
            mtn_mean_var_npy=opt.mtn_mean_var_npy,
            reprepare_idx_map=opt.reprepare_idx_map,
            use_lip_sync=opt.use_lip_sync_loss,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        return data_loader
    
    def _init_optim(self):
        opt = self.opt
        optim = Adan(self.LMDM.model.parameters(), lr=opt.lr, weight_decay=0.02)
        return optim

    def _init_log(self):
        opt = self.opt
        
        experiment_path = os.path.join(opt.experiment_dir, opt.experiment_name)
        self.error_log_path = os.path.join(experiment_path, 'error')
        
        if not self.is_main_process:
            return

        # ckpt
        self.ckpt_path = os.path.join(experiment_path, 'ckpts')
        os.makedirs(self.ckpt_path, exist_ok=True)

        # save opt
        opt_pkl = os.path.join(experiment_path, 'opt.pkl')
        dump_pkl(vars(opt), opt_pkl)

        # loss log
        loss_log = os.path.join(experiment_path, 'loss.log')
        self.loss_logger = open(loss_log, 'a')

        self.ckpt_file_list_for_clear = []

    def _loss_backward(self, loss):
        self.optim.zero_grad()

        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        self.optim.step()

    def _compute_lip_sync_loss(self, data_dict, x_pred, x_gt):
        """
        Compute lip sync loss on a subset of the batch.

        Args:
            data_dict: batch data containing f_s, x_s_kp, source_kp, wav_path, etc.
            x_pred: (B, L, 265) predicted motion from diffusion model
            x_gt: (B, L, 265) ground-truth motion

        Returns:
            lip_loss: scalar loss
            lip_loss_dict: dict of loss components for logging
        """
        import random as _random
        from ..models.frozen_renderer import (
            motion_arr_to_kp_info, transform_keypoint_torch
        )
        from ..models.lip_sync_loss import extract_lip_region

        opt = self.opt
        device = self.device
        B = x_pred.shape[0]
        L = x_pred.shape[1]  # seq_frames (80)
        num_samples = min(opt.lip_sync_num_samples, B)

        # Check that lip sync data is available
        if 'f_s' not in data_dict or 'x_s_kp' not in data_dict:
            return torch.tensor(0.0, device=device), {}
        if 'source_kp' not in data_dict:
            return torch.tensor(0.0, device=device), {}

        # Select random subset of batch
        indices = _random.sample(range(B), num_samples)

        f_s = data_dict['f_s'][indices].to(device)           # (N, 32, 16, 64, 64)
        x_s_kp = data_dict['x_s_kp'][indices].to(device)     # (N, 21, 3)
        source_kp = data_dict['source_kp'][indices].to(device)  # (N, 63)

        # Select 5 consecutive frames from sequence for SyncNet
        max_start = L - 5
        if max_start <= 0:
            frame_start = 0
        else:
            frame_start = _random.randint(0, max_start)

        # Get predicted and GT motion for 5-frame window
        pred_motion_5 = x_pred[indices, frame_start:frame_start+5, :265]  # (N, 5, 265)
        if 'gt_motion_seq' in data_dict:
            gt_motion_5 = data_dict['gt_motion_seq'][indices, frame_start:frame_start+5].to(device)  # (N, 5, 265)
        else:
            gt_motion_5 = x_gt[indices, frame_start:frame_start+5, :265]  # (N, 5, 265)

        # Render 5 frames each for pred and GT
        pred_lip_frames = []
        gt_lip_frames = []

        for t_idx in range(5):
            # Predicted frame
            pred_frame_motion = pred_motion_5[:, t_idx]  # (N, 265)
            with torch.no_grad():
                pred_img = self.frozen_renderer.render_from_motion(
                    f_s, x_s_kp, pred_frame_motion, source_kp
                )  # (N, 3, H, W)
            pred_lip = extract_lip_region(pred_img)  # (N, 3, 48, 96)
            pred_lip_frames.append(pred_lip)

            # GT frame
            gt_frame_motion = gt_motion_5[:, t_idx]  # (N, 265)
            with torch.no_grad():
                gt_img = self.frozen_renderer.render_from_motion(
                    f_s, x_s_kp, gt_frame_motion, source_kp
                )  # (N, 3, H, W)
            gt_lip = extract_lip_region(gt_img)  # (N, 3, 48, 96)
            gt_lip_frames.append(gt_lip)

        # Stack: (N, 5, 3, 48, 96)
        pred_lip_stack = torch.stack(pred_lip_frames, dim=1)
        gt_lip_stack = torch.stack(gt_lip_frames, dim=1)

        # Get mel spectrogram for the 5-frame audio window
        # Use the frame index from data_dict to compute mel window
        mel_specs = []
        if 'wav_path' in data_dict:
            f_idx_base = data_dict.get('f_idx', None)
            for b_idx in range(num_samples):
                orig_idx = indices[b_idx]
                wav_path = data_dict['wav_path'][orig_idx]

                # Get or compute full mel
                if wav_path not in self._mel_cache:
                    try:
                        full_mel = self.lip_sync_loss.mel_extractor.precompute_full_mel(wav_path)
                        self._mel_cache[wav_path] = full_mel
                        # Keep cache bounded
                        if len(self._mel_cache) > 200:
                            oldest_key = next(iter(self._mel_cache))
                            del self._mel_cache[oldest_key]
                    except Exception:
                        # Fallback: zero mel
                        full_mel = torch.zeros(80, 1600)
                        self._mel_cache[wav_path] = full_mel

                full_mel = self._mel_cache[wav_path]

                # Slice for the 5-frame window
                if f_idx_base is not None:
                    if isinstance(f_idx_base, (int, float)):
                        abs_frame = int(f_idx_base) + frame_start
                    else:
                        abs_frame = int(f_idx_base[orig_idx]) + frame_start
                else:
                    abs_frame = frame_start

                mel_slice = self.lip_sync_loss.mel_extractor.slice_mel_for_window(
                    full_mel, abs_frame, num_frames=5
                )  # (1, 80, 16)
                mel_specs.append(mel_slice.squeeze(0))  # (80, 16)

            mel_batch = torch.stack(mel_specs, dim=0).unsqueeze(1).to(device)  # (N, 1, 80, 16)
        else:
            # No wav data — use zeros (loss will be noisy but won't crash)
            mel_batch = torch.zeros(num_samples, 1, 80, 16, device=device)

        # Compute lip sync loss
        lip_loss, lip_loss_dict = self.lip_sync_loss(
            pred_lip_stack, gt_lip_stack, mel_batch
        )

        return lip_loss, lip_loss_dict

    def _train_one_step(self, data_dict):
        x = data_dict["kp_seq"]             # (B, L, kp_dim)
        cond_frame = data_dict["kp_cond"]   # (B, kp_dim)
        cond = data_dict["aud_cond"]        # (B, L, aud_dim)

        if not self.opt.use_accelerate:
            x = x.to(self.device)
            cond_frame = cond_frame.to(self.device)
            cond = cond.to(self.device)

        # ── Standard diffusion loss ───────────────────────────────────────
        use_lip = (self.opt.use_lip_sync_loss and
                   self.global_step % self.opt.lip_sync_every_n_steps == 0)

        if use_lip:
            # Use diffusion_with_pred to also get predicted motion
            loss, loss_dict, x_pred = self.LMDM.diffusion_with_pred(
                x, cond_frame, cond, t_override=None
            )
        else:
            loss, loss_dict = self.LMDM.diffusion(
                x, cond_frame, cond, t_override=None
            )

        # ── Lip sync loss (every N steps) ─────────────────────────────────
        if use_lip:
            try:
                lip_loss, lip_loss_dict = self._compute_lip_sync_loss(
                    data_dict, x_pred.detach(), x
                )
                loss = loss + lip_loss
                loss_dict.update({k: float(v) for k, v in lip_loss_dict.items()})
            except Exception:
                if self.is_main_process:
                    traceback.print_exc()
                    print("[Trainer] ⚠️  Lip sync loss failed, skipping this step")

        return loss, loss_dict

    def _train_one_epoch(self):
        data_loader = self.data_loader

        DAM = DictAverageMeter()

        self.LMDM.train()
        self.local_step = 0
        for data_dict in tqdm(data_loader, disable=not self.is_main_process):
            self.global_step += 1
            self.local_step += 1

            loss, loss_dict = self._train_one_step(data_dict)
            self._loss_backward(loss)

            if self.is_main_process:
                loss_dict['total_loss'] = loss
                loss_dict_val = {k: float(v) for k, v in loss_dict.items()}
                DAM.update(loss_dict_val)

        return DAM

    def _show_and_save(self, DAM: DictAverageMeter):
        if not self.is_main_process:
            return
        
        self.LMDM.eval()

        epoch = self.epoch

        # show all loss
        avg_loss_msg = "|"
        for k, v in DAM.average().items():
            avg_loss_msg += " %s: %.6f |" % (k, v)
        msg = f'Epoch: {epoch}, Global_Steps: {self.global_step}, {avg_loss_msg}'
        print(msg, file=self.loss_logger)
        self.loss_logger.flush()

        # save model
        if self.accelerator is not None:
            state_dict = self.accelerator.unwrap_model(self.LMDM.model).state_dict()
        else:
            state_dict = self.LMDM.model.state_dict()

        ckpt = {
            "model_state_dict": state_dict,
        }
        ckpt_p = os.path.join(self.ckpt_path, f"train_{epoch}.pt")
        torch.save(ckpt, ckpt_p)
        tqdm.write(f"[MODEL SAVED at Epoch {epoch}] ({len(self.ckpt_file_list_for_clear)})")
        
        # clear model
        if epoch % self.opt.save_ckpt_freq != 0:
            self.ckpt_file_list_for_clear.append(ckpt_p)

        if len(self.ckpt_file_list_for_clear) > 5:
            _ckpt = self.ckpt_file_list_for_clear.pop(0)
            try:
                os.remove(_ckpt)
            except:
                traceback.print_exc()
                self.ckpt_file_list_for_clear.insert(0, _ckpt)

    def _train_loop(self):
        print(time.asctime(), 'start ...')

        opt = self.opt

        start_epoch = 1
        self.global_step = 0
        self.local_step = 0
        for epoch in trange(start_epoch, opt.epochs + 1, disable=not self.is_main_process):
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            self.epoch = epoch
            DAM = self._train_one_epoch()

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            if self.is_main_process:
                self.LMDM.eval()
                self._show_and_save(DAM)

        print(time.asctime(), 'done.')

    def train_loop(self):
        try:
            self._train_loop()
        except:
            msg = traceback.format_exc()
            error_msg = f'{time.asctime()} \n {msg} \n'
            print(error_msg)
            t = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            logname = f'{t}_rank{self.process_index}_error.log'
            os.makedirs(self.error_log_path, exist_ok=True)
            errorfile = os.path.join(self.error_log_path, logname)
            with open(errorfile, 'a') as f:
                f.write(error_msg)
            print(f'error msg write into {errorfile}')