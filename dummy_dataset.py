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
        sequence_dirs = glob.glob(os.path.join("/storage/nfs/bdai_download_local/bridge_bdai_no_berkeley_numpy/new_datasets/bdai/stacking_cups_logitech/", "**/out.npy"), recursive=True)
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
    
    def get_video(self,idx):
        print(f"idx : {idx}")
        if self.sample_validation:
            idx = random.randrange(0,len(self.val_sequences))
            samples = self.val_sequences[idx]
            task = self.val_tasks[idx]
        else:
            idx = random.randrange(0,len(self.sequences))
            samples = self.sequences[idx]
            task = self.tasks[idx]

        episode = [s for s in samples] # List of images where each image is (128,128,3) and np.uint8
    
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
                print(f"Episode length : {len(full_obs)}")
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
        if self.sample_validation:
            idx = random.randrange(0,len(self.val_sequences))
            samples = self.val_sequences[idx]
            task = self.val_tasks[idx]
        else:
            idx = random.randrange(0,len(self.sequences))
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

class BDAIHighResJPGDataset(Dataset):
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
        self.sample_per_seq = self.sample_frames
        data_path = "/storage/nfs/jpatel/high_res_in_house_datasets/"
        self.dirs_with_jpgs = list(self.get_dirs_with_jpgs(data_path))
        self.num_samples = len(self.dirs_with_jpgs)
        print(f"Training samples: {self.num_samples}")
        print("Done")
        self.get_video(5)
        
    def __len__(self):
        return self.num_samples
    
    def get_video(self, idx):
        dir_name = self.dirs_with_jpgs[idx]
        image_files = [f for f in os.listdir(dir_name) if f.lower().endswith(('.jpg'))]
        sorted_image_files = sorted(image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        episode = []
        for image_file in sorted_image_files:
            image_path = os.path.join(dir_name, image_file)
            with Image.open(image_path) as img:
                # Convert image to numpy array with type np.uint8
                img_array = np.array(img, dtype=np.uint8)
                # Ensure the image has 3 channels (RGB)
                episode.append(img_array)

        episode = self.get_samples(episode)

    def get_samples(self, seq):
        N = len(seq)
        ### uniformly sample {self.sample_per_seq} frames, including the first and last frame
        samples = []
        for i in range(self.sample_per_seq-1):
            samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
        samples.append(N-1)
        return [seq[i] for i in samples]
    
    def get_dirs_with_jpgs(self,base_path):
        # Initialize a set to avoid duplicates
        dirs_with_jpgs = set()

        # Walk through the directory tree
        for root, dirs, files in os.walk(base_path):
            # Check if any .jpg files are in the current directory
            if any(file.lower().endswith('.jpg') for file in files):
                dirs_with_jpgs.add(root)

        return dirs_with_jpgs
    
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
                print(f"Episode length : {len(full_obs)}")
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
        dir_name = self.dirs_with_jpgs[idx]
        image_files = [f for f in os.listdir(dir_name) if f.lower().endswith(('.jpg'))]
        sorted_image_files = sorted(image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        episode = []
        for image_file in sorted_image_files:
            image_path = os.path.join(dir_name, image_file)
            with Image.open(image_path) as img:
                # Convert image to numpy array with type np.uint8
                img_array = np.array(img, dtype=np.uint8)
                # Ensure the image has 3 channels (RGB)
                episode.append(img_array)
        
        episode = self.get_samples(episode)
        
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
            
        task = "robot doing a task"
        return {'pixel_values': pixel_values, "task_text" : task}

class BDAIZoomInOutDataset(Dataset):
    def __init__(self, num_samples=10, width=1024, height=576, sample_frames=25):
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
        self.random_choice = 'zoom_in'
        self.load_image(idx)
        
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
        breakpoint()
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

class ShapeDataset(Dataset):
    def __init__(self, num_samples=2, width=1024, height=576, sample_frames=25):
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
        # self.get_images(5)
        
    def draw_circle(self, circle_center, image):
        # Draw a red circle
        radius = 30
        color_red = (0, 0, 255)  # BGR format for red
        thickness = -1  # Solid fill
        cv2.circle(image, circle_center, radius, color_red, thickness)
        return image
        
    def draw_square(self, square_center, image):
        # Draw a red circle
        side_length = 100
        thickness = -1  # Solid fill
        top_left_vertex = (square_center[0] - side_length // 2, square_center[1] - side_length // 2)
        color_blue = (255, 0, 0)  # BGR format for blue
        cv2.rectangle(image, top_left_vertex, (top_left_vertex[0] + side_length, top_left_vertex[1] + side_length), color_blue, thickness)
        return image

    def draw_triangle(self, triangle_center, image):
        # Draw a red circle
        side_length = 100
        triangle_height = int(np.sqrt(3) / 2 * side_length)  # Height of the equilateral triangle
        pts = np.array([ 
            [triangle_center[0], triangle_center[1] - triangle_height // 2],  # Top vertex
            [triangle_center[0] - side_length // 2, triangle_center[1] + triangle_height // 2],  # Bottom left vertex
            [triangle_center[0] + side_length // 2, triangle_center[1] + triangle_height // 2]  # Bottom right vertex
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        color_green = (0, 255, 0)  # BGR format for green
        cv2.fillPoly(image, [pts], color_green)
        return image
        
    def __len__(self):
        return self.num_samples
    
    def interpolate_points(self, p1, p2, n):
        x1, y1 = p1
        x2, y2 = p2
        x_values = np.linspace(x1, x2, n+2).astype(int)  # n+2 because we include the endpoints
        y_values = np.linspace(y1, y2, n+2).astype(int)
        interpolated_points = list(zip(x_values, y_values))
        return interpolated_points
    
    def get_images(self, idx):
        images = []
        
        instructions = ["move the red square to blue circle.",
        "move the red square to green triangle.",
        "move the blue circle to green triangle."]
        
        square_center_start = (100, 270)
        square_center_end = (100, 270)
        
        triangle_center_start = (70, 70)
        triangle_center_end = (70, 70)
        
        circle_center_start = (400, 230)
        circle_center_end = (400, 230)
        
        instruction = instructions[idx%3]
        
        if instruction=="move the red square to blue circle.":
            square_center_end = circle_center_end
        elif instruction=="move the red square to green triangle.":
            square_center_end = triangle_center_end
        else:
            circle_center_end = triangle_center_end

        square_centers = self.interpolate_points(square_center_start, square_center_end, self.sample_frames)
        circle_centers = self.interpolate_points(circle_center_start, circle_center_end, self.sample_frames)
        triangle_centers = self.interpolate_points(triangle_center_start, triangle_center_end, self.sample_frames)
        
        for i in range(self.sample_frames):
            # Create a blank image
            image = np.zeros((320, 512, 3), dtype=np.uint8)

            # Draw shapes
            image = self.draw_circle(circle_centers[i], image)
            image = self.draw_square(square_centers[i], image)
            if instruction=="move the red square to blue circle.":
                image = self.draw_triangle(triangle_centers[i], image)
            cv2.imwrite('shapes.png', image)
            time.sleep(0.3)
        breakpoint()
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Save the image
        episode = []
        
        instructions = ["move the red square to blue circle.",
        "move the red square to green triangle."]
        # "move the blue circle to green triangle."]
        
        square_center_start = (100, 270)
        square_center_end = (100, 270)
        
        triangle_center_start = (70, 70)
        triangle_center_end = (70, 70)
        
        circle_center_start = (400, 230)
        circle_center_end = (400, 230)
        
        instruction = instructions[idx%2]
        
        if instruction=="move the red square to blue circle.":
            # print(f"picking 0 : {instruction}")
            square_center_end = circle_center_end
        elif instruction=="move the red square to green triangle.":
            square_center_end = triangle_center_end
            # print(f"picking 1 : {instruction}")
        else:
            print("Dataloader is not working as expected")
        # else:
        #     circle_center_end = triangle_center_end

        square_centers = self.interpolate_points(square_center_start, square_center_end, self.sample_frames)
        circle_centers = self.interpolate_points(circle_center_start, circle_center_end, self.sample_frames)
        triangle_centers = self.interpolate_points(triangle_center_start, triangle_center_end, self.sample_frames)
        
        for i in range(self.sample_frames):
            # Create a blank image
            image = np.zeros((320, 512, 3), dtype=np.uint8)

            # Draw shapes
            image = self.draw_circle(circle_centers[i], image)
            image = self.draw_square(square_centers[i], image)
            image = self.draw_triangle(triangle_centers[i], image)
            cv2.imwrite('shapes.png', image)
            episode.append(image)

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
            
        return {'pixel_values': pixel_values, "task_text" : instruction}
