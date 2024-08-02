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


class BDAIDataset(Dataset):
    def __init__(self, num_samples=172, width=1024, height=576, sample_frames=25):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        # self.num_samples = num_samples
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        sequence_dirs = glob.glob(os.path.join(
            "/storage/nfs/bdai_download_local/bridge_bdai_no_berkeley_numpy/new_datasets/bdai/stacking_cups_logitech/", "**/out.npy"), recursive=True)
        # stacking_cups_logitech
        self.sequences = []
        self.tasks = []
        self.sample_per_seq = sample_frames
        obss, tasks = [], []
        val_obss, val_tasks = [], []
        for seq_dir in tqdm(sequence_dirs):
            obs, task = self.extract_seq(seq_dir)
            if 'val' in seq_dir:
                val_tasks.extend(task)
                val_obss.extend(obs)
            else:
                tasks.extend(task)
                obss.extend(obs)

        self.sequences = obss
        self.tasks = tasks
        self.val_sequences = val_obss
        self.val_tasks = val_tasks
        self.num_samples = len(self.sequences) + len(self.val_sequences)
        self.sample_validation = False
        print("training_samples: ", len(self.sequences))
        print("validation_samples: ", len(self.val_sequences))
        print("Done")
        # self.get_video(5)
        # breakpoint()

    def __len__(self):
        return self.num_samples

    def get_video(self, idx):
        print(f"idx : {idx}")
        if self.sample_validation:
            idx = random.randrange(0, len(self.val_sequences))
            samples = self.val_sequences[idx]
            task = self.val_tasks[idx]
        else:
            idx = random.randrange(0, len(self.sequences))
            samples = self.sequences[idx]
            task = self.tasks[idx]

        # List of images where each image is (128,128,3) and np.uint8
        episode = [s for s in samples]

    def get_samples(self, seq):
        N = len(seq)
        # uniformly sample {self.sample_per_seq} frames, including the first and last frame
        samples = []
        for i in range(self.sample_per_seq-1):
            samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
        samples.append(N-1)
        return [seq[i] for i in samples]

    def extract_seq(self, seqs_path):
        seqs = np.load(seqs_path, allow_pickle=True)
        extract_language_from_npy = "language" in seqs[0].keys()
        task = [] if extract_language_from_npy else seqs_path.split(
            '/')[-3].replace('_', ' ')
        outputs = []

        for seq in seqs:
            observations = seq["observations"]
            viewpoints = [v for v in observations[0].keys() if "image" in v]
            N = len(observations)
            for viewpoint in viewpoints:
                full_obs = [observations[i][viewpoint] for i in range(N)]
                print(f"Episode length : {len(full_obs)}")
                sampled_obs = self.get_samples(full_obs)
                outputs.append(sampled_obs)
                if extract_language_from_npy:
                    if seq['language'][0] == "":
                        task.append("Robot doing a task")
                    else:
                        task.append(seq['language'][0])
        tasks = task if extract_language_from_npy else [task] * len(outputs)

        return outputs, tasks

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a dataset
        if self.sample_validation:
            idx = random.randrange(0, len(self.val_sequences))
            samples = self.val_sequences[idx]
            task = self.val_tasks[idx]
        else:
            idx = random.randrange(0, len(self.sequences))
            samples = self.sequences[idx]
            task = self.tasks[idx]

        episode = [s for s in samples]

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
        return {'pixel_values': pixel_values, "task_text": task}

class BDAIHighResJPGDataset(Dataset):
    def __init__(self, num_samples=50, width=1024, height=576, sample_frames=25, original_fps=False):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
            original_fps (bool): If it is required to have original FPS
        """
        # self.num_samples = num_samples
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.sample_per_seq = self.sample_frames
        data_path = "/storage/nfs/jpatel/high_res_in_house_datasets/"
        # data_path = "/storage/nfs/jpatel/franka_data/franka_jpg/place_cube_in_tray_20240510_185930"
        self.dirs_with_jpgs = list(self.get_dirs_with_jpgs(data_path))
        self.num_samples = len(self.dirs_with_jpgs)
        print(f"Training samples: {self.num_samples}")
        print("Done")
        self.original_fps = original_fps
        self.get_video(5)

    def __len__(self):
        return self.num_samples

    def get_video(self, idx):
        dir_name = self.dirs_with_jpgs[idx]
        image_files = [f for f in os.listdir(
            dir_name) if f.lower().endswith(('.jpg'))]
        sorted_image_files = sorted(
            image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

        # image_files = [f for f in os.listdir(dir_name) if f.lower().endswith(('.png'))]
        # sorted_image_files = image_files

        episode = []
        for image_file in sorted_image_files:
            image_path = os.path.join(dir_name, image_file)
            with Image.open(image_path) as img:
                # Convert image to numpy array with type np.uint8
                img_array = np.array(img, dtype=np.uint8)
                # Ensure the image has 3 channels (RGB)
                episode.append(img_array)
        if not self.original_fps:
            episode = self.get_samples(episode)
        else:
            self.original_seq_len = len(episode)
            max_start_idx = self.original_seq_len - self.sample_frames
            start_idx = random.randint(0, max_start_idx)
            episode = episode[start_idx: start_idx+self.sample_frames]

    def get_samples(self, seq):
        N = len(seq)
        # uniformly sample {self.sample_per_seq} frames, including the first and last frame
        samples = []
        for i in range(self.sample_per_seq-1):
            samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
        samples.append(N-1)
        return [seq[i] for i in samples]

    def get_dirs_with_jpgs(self, base_path):
        # Initialize a set to avoid duplicates
        dirs_with_jpgs = set()

        # Walk through the directory tree
        for root, dirs, files in os.walk(base_path):
            # Check if any .jpg files are in the current directory
            if any(file.lower().endswith('.jpg') for file in files):
                dirs_with_jpgs.add(root)
            # if any(file.lower().endswith('.png') for file in files):
            #     dirs_with_jpgs.add(root)
        return dirs_with_jpgs

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        dir_name = self.dirs_with_jpgs[idx]
        image_files = [f for f in os.listdir(
            dir_name) if f.lower().endswith(('.jpg'))]
        sorted_image_files = sorted(
            image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

        # image_files = [f for f in os.listdir(dir_name) if f.lower().endswith(('.png'))]
        # sorted_image_files = image_files

        episode = []
        for image_file in sorted_image_files:
            image_path = os.path.join(dir_name, image_file)
            with Image.open(image_path) as img:
                # Convert image to numpy array with type np.uint8
                img_array = np.array(img, dtype=np.uint8)
                # Ensure the image has 3 channels (RGB)
                episode.append(img_array)

        if not self.original_fps:
            episode = self.get_samples(episode)
            original_episode = episode
        else:
            self.original_seq_len = len(episode)
            max_start_idx = self.original_seq_len - self.sample_frames
            start_idx = random.randint(0, max_start_idx)
            original_episode = episode
            episode = episode[start_idx: start_idx+self.sample_frames]

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
