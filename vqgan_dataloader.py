# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import os

import numpy.typing as npt
import torch
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf
from taming.models.vqgan import VQModel
from teco.metrics.utils import get_lpips, get_psnr, get_ssim
import random
import time

class vqganDataloader:
    """HeliosPredictor processes test batches concurrently on multiple GPUs,
    performs inference on them and computes test_loss, metrics, ground_truth videos
    and predicted videos for visualization.
    """

    def __init__(self):
        self.dtype = None

        print("Loading VQGAN...")
        vqgan_config = OmegaConf.load("/workspaces/bdai/projects/helios/configs/vq_f4.yaml")
        self.vqgan_model = VQModel(**vqgan_config.model.params).eval()

        self.vqgan_model.requires_grad_(False)
        self.vqgan_model = self.vqgan_model.bfloat16()
        self.vqgan_model = self.vqgan_model.to("cuda:0")
        print("Loaded VQGAN")

        # Metrics
        self.ssim = get_ssim()
        self.psnr = get_psnr()
        self.lpips = get_lpips(device="cuda:0")

    def _decode_batch_in_chunks(self, batch: torch.Tensor, seq_len) -> torch.Tensor:
        # Gets the dtype of the model.  Assumes all the parameters of the model have the same dtype
        # and that the dtype doesn't change after initialization
        if self.dtype is None:
            self.dtype = next(self.vqgan_model.parameters()).dtype

        videos = []
        for start_time in range(0, seq_len, 9):
            end_time = min(start_time + 9, seq_len)
            chunk = batch[start_time:end_time]
            video_chunk = self._decode_one_chunk(chunk)
            videos.append(video_chunk)
        videos = torch.concatenate(videos, dim=0)
        return videos

    def _decode_one_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.dtype is None:
            self.dtype = next(self.vqgan_model.parameters()).dtype

        chunk = chunk.to(self.dtype)
        chunk = self.vqgan_model.decode(chunk).detach()
        video_chunk = 255 * (chunk * 0.5 + 0.5).clip(0, 1)
        video_chunk = video_chunk.to(torch.uint8)
        return video_chunk

    def __call__(self, batch: dict[str, npt.NDArray], seq_len):
        # Convert numpy batches to torch Tensors
        with torch.no_grad():
            pred_video = self._decode_batch_in_chunks(batch, seq_len)
        return pred_video

def main():
    hp = vqganDataloader()
    path = "/workspaces/bdai/train/dataset_splits/rtx_enc_emb_train_len100_frac25_npz.txt"
    train_dataset_sequences = open(path,"r").readlines()
    for i, train_dataset_sequence in enumerate(train_dataset_sequences):
        start_time = time.time()
        filepath, start_idx = train_dataset_sequence.strip().split(",")
        data = np.load(filepath,allow_pickle=True)
        seq_len = data["embedding"].shape[0] - int(start_idx)
        batch = data["embedding"][int(start_idx):]
        batch = torch.from_numpy(batch).to("cuda:0")
        pred_video = hp(batch, seq_len)
        episode = pred_video.permute(0,2,3,1).cpu().detach().numpy()
        save_path = filepath.replace("nvme/rtx_enc_emb","nfs/jpatel/rtx_enc_emb")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, video=episode)
        print(f"Processing {i}/{len(train_dataset_sequences)} : time : {time.time()- start_time} s")
        
if __name__=="__main__":
    main()
    

