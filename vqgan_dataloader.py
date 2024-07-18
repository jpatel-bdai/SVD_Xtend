# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import os

import numpy.typing as npt
import torch
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf
from taming.models.vqgan import VQModel
from teco.metrics.utils import get_lpips, get_psnr, get_ssim

class Dataloader:
    """HeliosPredictor processes test batches concurrently on multiple GPUs,
    performs inference on them and computes test_loss, metrics, ground_truth videos
    and predicted videos for visualization.
    """

    def __init__(self, test_config: dict):
        """Load the model for inference

        Args:
            test_config (dict): Config file to initialize the model
            test_batch (tuple[torch.Tensor, torch.Tensor]): Temporary batch for forward pass
        """
        self.test_config = DictConfig(test_config)
        self.dtype = None
        
        print("Loading VQGAN...")
        vqgan_config = OmegaConf.load(self.test_config.vqgan_config)
        vqgan_model = VQModel(**vqgan_config.model.params).eval()

        vqgan_model.requires_grad_(False)
        if self.test_config.precision in ["bf16-true", "bf16-mixed", "bf16"]:
            print(f"casting to {self.test_config.precision}")
            vqgan_model = vqgan_model.bfloat16()
        elif self.test_config.precision in ["16-true"]:
            print(f"{self.test_config.precision} is not supported")
        self.vqgan_model = vqgan_model.to("cuda:0")
        print("Loaded VQGAN")

        # Metrics
        self.ssim = get_ssim()
        self.psnr = get_psnr()
        self.lpips = get_lpips(device="cuda:0")

    def _decode_batch_in_chunks(self, batch: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        """Decode a single batch of data by dividing them into small temporal chunks, decoding them
        and finally concatenating them into one video, with the intention of saving memory.

        Args:
            batch (torch.Tensor): data in TCHW format.
            start_idx (int): starting index (in dim T) when "chunking". Default: 0.

        Returns:
            torch.Tensor: the decoded video, of data type torch.uint8.
        """
        # Gets the dtype of the model.  Assumes all the parameters of the model have the same dtype
        # and that the dtype doesn't change after initialization
        if self.dtype is None:
            self.dtype = next(self.vqgan_model.parameters()).dtype

        videos = []
        for start_time in range(start_idx, 90, 9):
            end_time = min(start_time + 9, 90)
            chunk = batch[start_time:end_time]
            video_chunk = self._decode_one_chunk(chunk)
            videos.append(video_chunk)
        videos = torch.concatenate(videos, dim=0)
        return videos

    def _decode_one_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Decode one chunk of data and return the corresponding video.

        Args:
            chunk (torch.Tensor): data in TCHW format, typically with a small size in dim T.

        Returns:
            torch.Tensor: the decoded video, of data type torch.uint8.
        """
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

import hydra
@hydra.main(config_path="/workspaces/bdai/projects/helios/configs", config_name="config.yaml")
def main(config: DictConfig):
    config_as_dict: dict = OmegaConf.to_container(cfg=config, resolve=True)
    
    path = "/workspaces/bdai/train/dataset_splits/rtx_enc_emb_test_len100_frac25_npz.txt"
    train_dataset_sequences = open(path,"r").readlines()
    hp = Dataloader(config_as_dict)
    filepath, start_idx = train_dataset_sequences[0].strip().split(",")
    data = np.load(filepath,allow_pickle=True)
    seq_len = data["embedding"].shape[0]
    batch = data["embedding"][int(start_idx):]
    breakpoint()
    batch = torch.from_numpy(batch).to("cuda:0")
    pred_video = hp(batch, seq_len)

if __name__=="__main__":
    main()
    

