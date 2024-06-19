import numpy as np
from torch.utils.data import Dataset
import cv2
from pathlib import Path
import torch
import random
import glob 
from tqdm import tqdm
import os

class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, width=1024, height=576, sample_frames=15):
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

    def calculate_flow_between_frames(self, image1, image2):
        prvs = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(image2)
        hsv[..., 1] = 255
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitude = np.linalg.norm(mag) / (np.prod(mag.shape) / 10000)
        return magnitude

    def estimate_flow(self, video):
        magnitudes = []
        for idx in range(video.shape[0]-1):
            current_img = video[idx]
            next_img = video[idx+1]
            magnitude = self.calculate_flow_between_frames(current_img, next_img)
            magnitudes.append(magnitude)
        return magnitudes

    def trim_video_with_flow_threshold(self, video, flow_threshold, num_frames):
        trimmed_video = []
        trimmed_video.append(video[0])
        idx = 0
        skipped_frames = 0
        while len(trimmed_video) < num_frames:
            current_img = video[idx]
            while idx < (video.shape[0]-2):
                next_img = video[idx+1]
                magnitude = self.calculate_flow_between_frames(current_img, next_img)
                if(magnitude>flow_threshold):
                    trimmed_video.append(next_img)
                    idx += 1
                    break
                else:
                    skipped_frames += 1
                    idx += 1
            if(idx==(video.shape[0]-2)):
                break
                
        if len(trimmed_video) < num_frames:
            print(f"Did not find {num_frames} frames, only found {len(trimmed_video)} frames :( !!")
            return None, -1
        
        trimmed_video = np.array(trimmed_video)
        
        return trimmed_video, skipped_frames

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
            episodes = np.load(random.choice(video_npz), allow_pickle=True)[image_key]

            # Randomly select an episode from episodes
            episode = episodes[np.random.randint(episodes.shape[0])]

            # if len(episode) <= self.sample_frames:
            #     continue
            # elif len(episode) > self.sample_frames and len(episode) <= (2*self.sample_frames):
            #     N = len(episode)
            #     samples = []
            #     for i in range(self.sample_frames-1):
            #         samples.append(int(i*(N-1)/(self.sample_frames-1)))
            #     samples.append(N-1)
            #     final_video = [episode[i] for i in samples]
            #     episode = final_video
            #     found_episode = True
            # else:
            #     original_len = len(episode)
            #     if len(episode) > 250:
            #         episode = episode[:250]
            #     magnitudes = self.estimate_flow(episode)
            #     rem_frames = len(magnitudes) - (2 * self.sample_frames)
            #     flow_threshold = sorted(magnitudes)[rem_frames]
            #     episode, skipped_frames = self.trim_video_with_flow_threshold(episode, flow_threshold, self.sample_frames)
            #     if skipped_frames == -1:
            #         print("Trimmed video not found!!")
            #         continue
            #     else:
            #         found_episode = True
            
            if len(episode)>self.sample_frames:
                found_episode = True
            else:
                print(f"Finding new episode, Length of current episode : {len(episode)}")
        
        # folder_path = chosen_dataset
        # frames = os.listdir(folder_path)
        # # Sort the frames by name
        # frames.sort()

        # Ensure the selected folder has at least `sample_frames`` frames
        # if len(frames) < self.sample_frames:
        #     raise ValueError(
        #         f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames.")

        # Randomly select a start index for frame sequence
        # start_idx = random.randint(0, len(frames) - self.sample_frames)
        # start_idx = 0
        # selected_frames = frames[start_idx:start_idx + self.sample_frames]

        # Initialize a tensor to store the pixel values
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))

        # Load and process each frame
        for i, frame in enumerate(episode):
            if i > (self.sample_frames-1): break
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
        return {'pixel_values': pixel_values}


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
        # sequence_dirs = glob.glob(os.path.join("/storage/nfs/bdai_download_local", "**/out.npy"), recursive=True)
        sequence_dirs = glob.glob(os.path.join("/storage/nfs/bdai_download_local/bridge_bdai_no_berkeley_numpy/new_datasets/bdai", "**/out.npy"), recursive=True)
        self.sequences = []
        self.tasks = []
        self.sample_per_seq = sample_frames
        obss, tasks = [], []
        for seq_dir in tqdm(sequence_dirs[:len(sequence_dirs)//2]):
            obs, task = self.extract_seq(seq_dir)
            tasks.extend(task)
            obss.extend(obs)
            
        self.sequences = obss
        self.tasks = tasks
        self.num_samples = len(self.sequences)
        print("training_samples: ", len(self.sequences))
        print("Done")
        
    def __len__(self):
        return self.num_samples
    
    def get_samples(self, seq):
        N = len(seq)
        ### uniformly sample {self.sample_per_seq} frames, including the first and last frame
        samples = []
        for i in range(self.sample_per_seq-1):
            samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
        samples.append(N-1)
        return [seq[i] for i in samples]
    
    def extract_seq(self, seqs_path):
        seqs = np.load(seqs_path, allow_pickle=True)
        extract_language_from_npy = "language" in seqs[0].keys()
        task = [] if extract_language_from_npy else seqs_path.split('/')[-3].replace('_', ' ')
        outputs = []
        for seq in seqs:
            observations = seq["observations"]
            viewpoints = [v for v in observations[0].keys() if "image" in v]
            N = len(observations)
            for viewpoint in viewpoints:
                full_obs = [observations[i][viewpoint] for i in range(N)]
                sampled_obs = self.get_samples(full_obs)
                outputs.append(sampled_obs)
                if extract_language_from_npy:
                    if seq['language'][0]=="":
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
        samples = self.sequences[idx]
        task = self.tasks[idx]
        episode = [s for s in samples]
        
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))

        # Load and process each frame
        for i, frame in enumerate(episode):
            if i > (self.sample_frames-1): break
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
        return {'pixel_values': pixel_values, "task_text" : task}



class BDAIZoomInOutDataset(Dataset):
    def __init__(self, num_samples=336, width=1024, height=576, sample_frames=25):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        # sequence_dirs = glob.glob(os.path.join("/storage/nfs/bdai_download_local", "**/out.npy"), recursive=True)
        sequence_dirs = glob.glob(os.path.join("/storage/nfs/jpatel/bridge_data_bdai_mix_04_07_2024/bridge_data_bdai_mix_numpy/bdai/", "**/out.npy"), recursive=True)
        self.sequences = []
        self.tasks = []
        self.sample_per_seq = sample_frames
        obss, tasks = [], []
        for seq_dir in tqdm(sequence_dirs[:len(sequence_dirs)//2]):
            obs, task = self.extract_seq(seq_dir)
            tasks.extend(task)
            obss.extend(obs)
            
        self.sequences = obss
        self.tasks = tasks
        print("training_samples: ", len(self.sequences))
        print("Done")
        idx = 5
        # To test __getitem__ function
        # self.load_image(idx)
        self.random_choice = 'zoom_in'
        
    def __len__(self):
        return self.num_samples
    
    def load_image(self, idx):
        samples = self.sequences[idx]
        task = self.tasks[idx]
        episode = []
        if self.random_choice == "zoom_in":
            scale_factors = np.linspace(2, 3, 25)
            task = random.choice(["Zooming in", 'Enlarging an image'])
            self.random_choice = 'zoom_out'
        else:
            scale_factors = np.linspace(2, 1, 25)
            task = random.choice(["Zooming out", 'Shrinking an image'])
            self.random_choice = 'zoom_in'
        
        for scale_factor in scale_factors:
            zoomed_image = self.zoom_in(samples[0], scale_factor)
            episode.append(zoomed_image)

        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        
        
    def get_samples(self, seq):
        N = len(seq)
        ### uniformly sample {self.sample_per_seq} frames, including the first and last frame
        samples = []
        for i in range(self.sample_per_seq-1):
            samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
        samples.append(N-1)
        return [seq[i] for i in samples]
    
    def extract_seq(self, seqs_path):
        seqs = np.load(seqs_path, allow_pickle=True)
        extract_language_from_npy = "language" in seqs[0].keys()
        task = [] if extract_language_from_npy else seqs_path.split('/')[-3].replace('_', ' ')
        outputs = []
        for seq in seqs:
            observations = seq["observations"]
            viewpoints = [v for v in observations[0].keys() if "image" in v]
            N = len(observations)
            for viewpoint in viewpoints:
                full_obs = [observations[i][viewpoint] for i in range(N)]
                sampled_obs = self.get_samples(full_obs)
                outputs.append(sampled_obs)
                if extract_language_from_npy:
                    if seq['language'][0]=="":
                        task.append("Robot doing a task")
                    else:
                        task.append(seq['language'][0])
        tasks = task if extract_language_from_npy else [task] * len(outputs)

        return outputs, tasks

    def zoom_in(self, image, scale_factor):
        # Get the center of the image
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        
        # Calculate the new dimensions
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)

        
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Calculate the coordinates to crop the image to original size
        start_x = new_width // 2 - center_x
        start_y = new_height // 2 - center_y
        end_x = start_x + image.shape[1]
        end_y = start_y + image.shape[0]
        
        # Crop the resized image to the original size
        zoomed_image = resized_image[start_y:end_y, start_x:end_x]
        
        return zoomed_image

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a dataset
        samples = self.sequences[idx]
        task = self.tasks[idx]
        
        # frame_width, frame_height = image.shape[1], image.shape[0]
        self.random_choice = random.choice(["zoom_in", "zoom_out"])
        episode = []
        if self.random_choice == "zoom_in":
            scale_factors = np.linspace(2, 3, 25)
            task = random.choice(["Zooming in", 'Enlarging an image'])
            # self.random_choice = "zoom_out"
        else:
            scale_factors = np.linspace(2, 1, 25)
            task = random.choice(["Zooming out", 'Shrinking an image'])
            # self.random_choice = "zoom_in"
        
        for scale_factor in scale_factors:
            zoomed_image = self.zoom_in(samples[0], scale_factor)
            episode.append(zoomed_image)

        
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))

        # Load and process each frame
        for i, frame in enumerate(episode):
            if i > (self.sample_frames-1): break
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
            
        return {'pixel_values': pixel_values, "task_text" : task}
