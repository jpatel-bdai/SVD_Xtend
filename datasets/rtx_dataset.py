import numpy as np
from torch.utils.data import Dataset
import cv2
from pathlib import Path
import torch
import random
import glob
from tqdm import tqdm
import os
import time
from PIL import Image


class RTXDataset(Dataset):
    def __init__(self, num_samples=100000, width=1024, height=576, sample_frames=25, original_fps=True):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples
        # Define the path to the folder containing video frames
        self.rtx_root_dir = '/storage/nfs/jpatel/rtx_dataset_with_language'
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

        self._load_video_paths()

    def _load_video_paths(self):
        root_dir = Path(self.rtx_root_dir)
        # Mpdified to use only a single dataset
        self.datasets = [d for d in root_dir.iterdir() if d.is_dir()]
        print(f"Found {len(self.datasets)} datasets.")
        # found_episode = False
        # while not found_episode:
        chosen_dataset = random.choice(self.datasets)

        image_dir = chosen_dataset / "data" / "images"
        image_key = next(image_dir.iterdir()).name
        image_dir = image_dir / image_key
        image_key = "images/" + image_key
        lang_dir = chosen_dataset / "data" / "language_instruction"
        video_npz = sorted(image_dir.glob('*.npz'))
        episodes = np.load(random.choice(video_npz),
                           allow_pickle=True)[image_key]

        # Randomly select an episode from episodes
        episode = episodes[np.random.randint(episodes.shape[0])]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a dataset
        found_episode = False
        while not found_episode:
            chosen_dataset = random.choice(self.datasets)

            image_dir = chosen_dataset / "data" / "images"
            image_key = next(image_dir.iterdir()).name
            image_dir = image_dir / image_key
            image_key = "images/" + image_key
            lang_dir = chosen_dataset / "data" / "language_instruction"
            video_npz = sorted(image_dir.glob('*.npz'))
            episodes = np.load(random.choice(video_npz),
                               allow_pickle=True)[image_key]

            # Randomly select an episode from episodes
            episode = episodes[np.random.randint(episodes.shape[0])]

            if len(episode) > self.sample_frames:
                found_episode = True
            else:
                print(
                    f"Finding new episode, Length of current episode : {len(episode)}")

        # Randomly select a start index for frame sequence
        start_idx = random.randint(0, episode.shape[0] - self.sample_frames)
        start_idx = 0
        original_episode = episode
        episode = episode[start_idx:start_idx + self.sample_frames]

        # Initialize a tensor to store the pixel values
        pixel_values = torch.empty(
            (self.sample_frames, self.channels, self.height, self.width))

        # Load and process each frame
        for i, frame in enumerate(episode):
            if i > (self.sample_frames-1):
                break
            # Resize the image and convert it to a tensor
            img_resized = cv2.resize(frame, (self.width, self.height))

            # img_resized = np.resize(frame, (self.height, self.width, 3))
            img_tensor = torch.from_numpy(img_resized).float()

            # Normalize the image by scaling pixel values to [-1, 1]
            img_normalized = img_tensor / 127.5 - 1

            # Rearrange channels if necessary
            if self.channels == 3:
                img_normalized = img_normalized.permute(
                    2, 0, 1)  # For RGB images
            elif self.channels == 1:
                img_normalized = img_normalized.mean(
                    dim=2, keepdim=True)  # For grayscale images

            pixel_values[i] = img_normalized

        task = "robot doing a task"
        return {'pixel_values': pixel_values, "task_text": task, "original_episode": original_episode}
