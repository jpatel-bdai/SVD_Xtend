#  Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from pathlib import Path
from typing import Any, Union

import torch
import torch.nn
import torchvision.transforms
import wandb
from torchmetrics.image.fid import _compute_fid
import numpy as np
from typing import Callable
import torch
from torchmetrics.functional.image.lpips import _lpips_update
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class FVD(torch.nn.Module):
    """Implementation of Frechet Video Distance (FVD) using I3D-based embeddings."""

    def __init__(self, device: str, chunk_size_threshold: int = 8):
        super().__init__()
        self.device = device
        # The source of the PyTorch version of the I3D model is the StyleGAN-V paper
        # https://github.com/universome/stylegan-v
        self.model_name = "i3d_torchscript.pt"
        self.model_path = Path("artifacts/i3d_torchscript:v0")
        self.model = self.init_model_()
        self.model_kwargs = dict(rescale=False, resize=False, return_features=True)
        self.chunk_size_threshold = chunk_size_threshold

    def init_model_(self) -> Any:
        """Isolate model loading implementation details from general behaviors (to help testing)."""
        model_filepath = self.model_path / self.model_name
        if not model_filepath.exists():
            self.model_path = Path(wandb.Api().artifact("bdaii/model-registry/i3d_torchscript:v0").download())
            model_filepath = self.model_path / self.model_name
        return torch.jit.load(model_filepath.as_posix()).eval().to(self.device)

    def set_device(self, device: str) -> None:
        """Set the device of the FVD object.

        Args:
            device (str): target device.
        """
        self.device = device
        self.model = self.model.to(device)

    def fvd_preprocess(self, videos: torch.Tensor, target_resolution: tuple[int, int]) -> torch.Tensor:
        """Preprocess video for embedding generation using I3D model, where the output video:
        - is floating point data type, in range [-1, 1];
        - has height and width as specified in `target_resolution`;
        - has shape (Batch, Channel, Time, Height, Width), i.e. BCTHW, as expected by I3D model.

        Args:
            video (torch.Tensor): original video of shape ...CHW, e.g. (Batch, Time, Channel, Height, Width).
            target_resolution (tuple[int, int]): target resolution, i.e. new (Height, Width) values.

        Returns:
            torch.Tensor: processed video.
        """
        scale_factor = 1.0
        if not videos.dtype == torch.uint8:
            scale_factor = 255.0
        videos = (videos * scale_factor).to(self.device).to(torch.float32)

        videos_shape = list(videos.size())
        all_frames = torch.reshape(videos, [-1] + videos_shape[-3:])
        resized_videos = torchvision.transforms.Resize(target_resolution)(all_frames)

        # Reshape to [batch, time, channel, height, width]
        target_shape = (*videos_shape[:3], *target_resolution)
        output_videos = torch.reshape(resized_videos, target_shape)

        # Permute to [batch, channel, time, height, width]
        output_videos = torch.permute(output_videos, [0, 2, 1, 3, 4])

        scaled_videos = 2.0 * output_videos / 255.0 - 1
        return scaled_videos

    def calculate_fvd(self, real_activations: torch.Tensor, generated_activations: torch.Tensor) -> torch.Tensor:
        """Calculate the Frechet Video Distance (FVD) between real and generated activations.

        Args:
            real_activations (torch.Tensor): real video activations;
            generated_activations (torch.Tensor): generated video activations.

        Returns:
            torch.Tensor: computed FVD.
        """
        # Calculate the unbiased covariance matrix of first activations.
        num_examples_real = torch.tensor(real_activations.size()[1], dtype=torch.float64)
        unbiased_correction = 1 if num_examples_real > 1 else 0
        sigma_real = torch.cov(input=real_activations, correction=unbiased_correction)

        # Calculate the unbiased covariance matrix of second activations.
        num_examples_generated = torch.tensor(generated_activations.size()[1], dtype=torch.float64)
        unbiased_correction = 1 if num_examples_generated > 1 else 0
        sigma_gen = torch.cov(input=generated_activations, correction=unbiased_correction)

        # FID
        mu_real = real_activations.mean(dim=1)
        mu_gen = generated_activations.mean(dim=1)
        result = _compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)

        return result

    def embed_videos(
        self, videos: list[torch.Tensor], image_shape: tuple[int, int] = (224, 224), chunk_size: int = 9
    ) -> torch.Tensor:
        """Compute embeddings for videos, using I3D model.  Input videos will be resized to `image_shape`
        if shaped differently, then fed to I3D model to get embeddings.

        Args:
            videos (list[torch.Tensor]): video sequence.
            image_shape (tuple[int, int], optional): shape of video fed into I3D model. Defaults to (224, 224).
            chunk_size (int): size of chunks to split video sequence into. Defaults to 9 (mininum length).

        Returns:
            torch.Tensor: embedding sequence.
        """
        if chunk_size > self.chunk_size_threshold:
            embeddings_list = []
            for v in videos:
                if not (v.shape[0] < chunk_size):  # Skip videos with sequence length less than chunk size.
                    # Preprocess one video and get frames in shape CTHW.
                    frames = self.fvd_preprocess(v[None, ...], image_shape).squeeze()
                    num_frames = frames.shape[1]
                    rounded_num_frames = (num_frames // chunk_size) * chunk_size
                    # Split to B' tensors, each with shape (C, T', H, W)
                    # where T' is chunk_size and B' is number of chunks.
                    chunks = torch.split(frames[:, :rounded_num_frames], chunk_size, dim=1)
                    # Stack the list of B' tensors to a single tensor, with shape (B', C, T', H, W).
                    frames = torch.stack(chunks)
                    embeddings_list.append(self.model(frames, **self.model_kwargs))

            if not embeddings_list:
                embeddings = torch.empty((0, 0)).to(videos[0].device)
            else:
                embeddings = torch.concatenate(embeddings_list)
        else:
            videos = torch.stack(videos)  # Convert list of videos to tensor of shape BTCHW.
            v = self.fvd_preprocess(videos, image_shape)
            embeddings = self.model(v, **self.model_kwargs)

        return embeddings.T

    def forward(
        self,
        real_videos: list[torch.Tensor],
        generated_videos: list[torch.Tensor],
        chunk_size: int = 9,
        return_embed: bool = False,
    ) -> Union[tuple[torch.Tensor], torch.Tensor]:
        # real_videos : List with 4 values where each value is of shape - (50 * 3 * 128 * 128)
        """Forward pass: compute embeddings for real and generated videos and calculate FVD."""
        for real_video, generated_video in zip(real_videos, generated_videos, strict=True):
            if real_video.shape != generated_video.shape:
                raise RuntimeError(
                    f"The size of the real videos {real_video.shape} should be the same as the generated videos"
                    f" {generated_video.shape}"
                )

        embed_real = self.embed_videos(real_videos, chunk_size=chunk_size).detach()
        embed_generated = self.embed_videos(generated_videos, chunk_size=chunk_size).detach()
        if return_embed:
            return embed_real, embed_generated

        result = self.calculate_fvd(embed_real, embed_generated)
        return result

lpips_eval: LPIPS = None

def compute_metric(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    metric_fn: Callable,
    average_dim: int = 1,
) -> torch.Tensor:
    # BTCHW in [0, 1]
    if prediction.shape != ground_truth.shape:
        raise ValueError("Predicted shape is not equal to ground truth")
    batch_size, frame_count = prediction.shape[0], prediction.shape[1]
    prediction = prediction.reshape(-1, *prediction.shape[2:])
    ground_truth = ground_truth.reshape(-1, *ground_truth.shape[2:])

    metrics = metric_fn(prediction, ground_truth)
    metrics = torch.reshape(metrics, (batch_size, frame_count))

    metrics = metrics.mean(axis=average_dim)  # B or T depending on dim

    return metrics


# all methods below take as input pairs of images
# of shape BTCHW. They DO NOT reduce batch dimension


def get_ssim(average_dim: int = 1) -> Callable:
    def fn(imgs1: torch.Tensor, imgs2: torch.Tensor) -> torch.Tensor:
        ssim = SSIM(reduction=None)
        return ssim(imgs1, imgs2)

    return lambda imgs1, imgs2: compute_metric(imgs1, imgs2, fn, average_dim=average_dim)


def get_psnr(average_dim: int = 1) -> Callable:
    def fn(imgs1: torch.Tensor, imgs2: torch.Tensor) -> torch.Tensor:
        psnr = PSNR(data_range=(0, 1), reduction=None, dim=[1, 2, 3])
        return psnr(imgs1, imgs2)

    return lambda imgs1, imgs2: compute_metric(imgs1, imgs2, fn, average_dim=average_dim)


# NOTE: lpips Assumes that images are in [0, 1]
def get_lpips(average_dim: int = 1, device: str = "cpu") -> Callable:
    global lpips_eval
    if lpips_eval is None:
        lpips_eval = LPIPS(net_type="alex", normalize=True).to(device)

    def fn(imgs1: torch.Tensor, imgs2: torch.Tensor) -> torch.Tensor:
        lpips, _ = _lpips_update(imgs1, imgs2, net=lpips_eval.net, normalize=True)
        return lpips

    return lambda imgs1, imgs2: compute_metric(imgs1, imgs2, fn, average_dim=average_dim)
