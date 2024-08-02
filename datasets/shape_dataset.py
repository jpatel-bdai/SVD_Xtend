from itertools import combinations
from enum import Enum
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


class ShapeDataset(Dataset):
    """
    This dataset is only compatible with train_svd_lang.py
    """

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

    def draw_circle(self, circle_center, image, radius):
        # Draw a red circle
        radius = radius
        color_red = (0, 0, 255)  # BGR format for red
        thickness = -1  # Solid fill
        cv2.circle(image, circle_center, radius, color_red, thickness)
        return image

    def draw_square(self, square_center, image):
        # Draw a red circle
        side_length = 100
        thickness = -1  # Solid fill
        top_left_vertex = (
            square_center[0] - side_length // 2, square_center[1] - side_length // 2)
        color_blue = (255, 0, 0)  # BGR format for blue
        cv2.rectangle(image, top_left_vertex, (
            top_left_vertex[0] + side_length, top_left_vertex[1] + side_length), color_blue, thickness)
        return image

    def draw_triangle(self, triangle_center, image):
        # Draw a red circle
        side_length = 100
        # Height of the equilateral triangle
        triangle_height = int(np.sqrt(3) / 2 * side_length)
        pts = np.array([
            [triangle_center[0], triangle_center[1] -
                triangle_height // 2],  # Top vertex
            [triangle_center[0] - side_length // 2, triangle_center[1] + \
                triangle_height // 2],  # Bottom left vertex
            [triangle_center[0] + side_length // 2, triangle_center[1] + \
                triangle_height // 2]  # Bottom right vertex
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
        # n+2 because we include the endpoints
        x_values = np.linspace(x1, x2, n+2).astype(int)
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

        instruction = instructions[idx % 3]

        if instruction == "move the red square to blue circle.":
            square_center_end = circle_center_end
        elif instruction == "move the red square to green triangle.":
            square_center_end = triangle_center_end
        else:
            circle_center_end = triangle_center_end

        square_centers = self.interpolate_points(
            square_center_start, square_center_end, self.sample_frames)
        circle_centers = self.interpolate_points(
            circle_center_start, circle_center_end, self.sample_frames)
        triangle_centers = self.interpolate_points(
            triangle_center_start, triangle_center_end, self.sample_frames)

        for i in range(self.sample_frames):
            # Create a blank image
            image = np.zeros((320, 512, 3), dtype=np.uint8)

            # Draw shapes
            image = self.draw_circle(circle_centers[i], image, 30)
            image = self.draw_square(square_centers[i], image, 30)
            if instruction == "move the red square to blue circle.":
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

        instructions = ["move the red square to blue circle",
                        "move the red square to green triangle"]
        # "move the blue circle to green triangle."]
        # instructions = ["blue circle",
        #                 "green triangle"]

        square_center_start = (100, 270)
        square_center_end = (100, 270)

        triangle_center_start = (70, 70)
        triangle_center_end = (70, 70)

        circle_center_start = (400, 230)
        circle_center_end = (400, 230)

        instruction = instructions[idx % 2]

        if instruction == "move the red square to blue circle":
            # print(f"picking 0 : {instruction}")
            square_center_end = circle_center_end
        elif instruction == "move the red square to green triangle":
            square_center_end = triangle_center_end
            # print(f"picking 1 : {instruction}")
        else:
            print("Dataloader is not working as expected")
        # else:
        #     circle_center_end = triangle_center_end

        square_centers = self.interpolate_points(
            square_center_start, square_center_end, self.sample_frames)
        circle_centers = self.interpolate_points(
            circle_center_start, circle_center_end, self.sample_frames)
        triangle_centers = self.interpolate_points(
            triangle_center_start, triangle_center_end, self.sample_frames)

        for i in range(self.sample_frames):
            # Create a blank image
            image = np.zeros((320, 512, 3), dtype=np.uint8)

            # Draw shapes
            image = self.draw_circle(circle_centers[i], image, 30)
            image = self.draw_square(square_centers[i], image)
            image = self.draw_triangle(triangle_centers[i], image)

            # Save conditioning images
            cond_image = np.zeros((320, 512, 3), dtype=np.uint8)
            if i == 0:
                cond_image_1 = self.draw_circle(
                    square_centers[i], cond_image, 10)
                # cv2.imwrite(f"./{idx%2}/cond_image_{i}.png", cond_image_1)
            if i == 24:
                cond_image_2 = self.draw_circle(
                    square_centers[i], cond_image, 10)
                # cv2.imwrite(f"./{idx%2}/cond_image_{i}.png", cond_image_2)

            cv2.imwrite(f"./{idx%2}/image_{i}.png", image)
            episode.append(image)

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

        condition_frames = []
        for i, frame in enumerate([cond_image_1, cond_image_2]):
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

            condition_frames.append(img_normalized)

        return {'pixel_values': pixel_values, "task_text": instruction, "original_episode": episode}


# from vqgan_dataloader import vqganDataloader


# Create a colour class
class Color(Enum):
    RED = (0, 0, 255, "Red")
    GREEN = (0, 255, 0, "Green")
    BLUE = (255, 0, 0, "Blue")
    YELLOW = (0, 255, 255, "Yellow")
    CYAN = (255, 255, 0, "Cyan")
    MAGENTA = (255, 0, 255, "Magenta")
    ORANGE = (0, 165, 255, "Orange")
    PURPLE = (128, 0, 128, "Purple")
    PINK = (203, 192, 255, "Pink")
    BROWN = (42, 42, 165, "Brown")

    def bgr(self):
        return self.value[:3]

    def name(self):
        return self.value[3]


class FiveShapesDataset(Dataset):
    """
    This dataset is only compatible with train_svd_lang.py
    """

    def __init__(self, num_samples=8, width=1024, height=576, sample_frames=25):
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
        self.shapes = ["red circle", "green square",
                       "blue triangle", "cyan pentagon", "magenta ellipse"]
        self.shapes_start_end = list(combinations(self.shapes, 2))
        # self.get_images(5)

    def draw_circle(self, circle_center, image, radius, color):
        thickness = -1  # Solid fill
        cv2.circle(image, circle_center, radius, color.bgr(), thickness)
        return image

    def draw_square(self, square_center, side_length, image, color):
        thickness = -1  # Solid fill
        top_left_vertex = (
            square_center[0] - side_length // 2, square_center[1] - side_length // 2)
        cv2.rectangle(image, top_left_vertex, (
            top_left_vertex[0] + side_length, top_left_vertex[1] + side_length), color.bgr(), thickness)
        return image

    def draw_triangle(self, triangle_center, side_length, image, color):
        triangle_height = int(np.sqrt(3) / 2 * side_length)
        pts = np.array([
            [triangle_center[0], triangle_center[1] -
                triangle_height // 2],  # Top vertex
            [triangle_center[0] - side_length // 2, triangle_center[1] + \
                triangle_height // 2],  # Bottom left vertex
            [triangle_center[0] + side_length // 2, triangle_center[1] + \
                triangle_height // 2]  # Bottom right vertex
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], color.bgr())
        return image

    def draw_ellipse(self, ellipse_center, image, axes, angle, start_angle, end_angle, color):
        thickness = -1  # Solid fill
        cv2.ellipse(image, ellipse_center, axes, angle,
                    start_angle, end_angle, color.bgr(), thickness)
        return image

    def draw_pentagon(self, pentagon_center, side_length, image, color):
        angle = 72  # Each internal angle of a regular pentagon
        # Radius of the circumscribed circle
        radius = side_length / (2 * np.sin(np.pi / 5))
        pts = []
        for i in range(5):
            theta = np.deg2rad(angle * i)
            x = pentagon_center[0] + int(radius * np.cos(theta))
            y = pentagon_center[1] + int(radius * np.sin(theta))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], color.bgr())
        return image

    def __len__(self):
        return self.num_samples

    def interpolate_points(self, p1, p2, n):
        x1, y1 = p1
        x2, y2 = p2
        # n+2 because we include the endpoints
        x_values = np.linspace(x1, x2, n+2).astype(int)
        y_values = np.linspace(y1, y2, n+2).astype(int)
        interpolated_points = list(zip(x_values, y_values))
        return interpolated_points

    def get_images(self, idx):
        start_shape = self.shapes_start_end[idx][0]
        end_shape = self.shapes_start_end[idx][1]
        instruction = f"move the {start_shape} to {end_shape}"

        circle_center_start = (30, 30)
        circle_center_end = (30, 30)

        square_center_start = (70, 30)
        square_center_end = (70, 30)

        triangle_center_start = (110, 40)
        triangle_center_end = (110, 40)

        pentagon_center_start = (40, 90)
        pentagon_center_end = (40, 90)

        ellipse_center_start = (100, 100)
        ellipse_center_end = (100, 100)

        start_shape_name = start_shape.split(" ")[1]
        end_shape_name = end_shape.split(" ")[1]

        if end_shape_name == "circle":
            end_shape_location = circle_center_end
        elif end_shape_name == "square":
            end_shape_location = square_center_end
        elif end_shape_name == "triangle":
            end_shape_location = triangle_center_end
        elif end_shape_name == "pentagon":
            end_shape_location = pentagon_center_end
        elif end_shape_name == "ellipse":
            end_shape_location = ellipse_center_end

        if start_shape_name == "circle":
            circle_center_end = end_shape_location
        elif start_shape_name == "square":
            square_center_end = end_shape_location
        elif start_shape_name == "triangle":
            triangle_center_end = end_shape_location
        elif start_shape_name == "pentagon":
            pentagon_center_end = end_shape_location
        elif start_shape_name == "ellipse":
            ellipse_center_end = end_shape_location

        # Modify shape end locations based on the instructions

        circle_centers = self.interpolate_points(
            circle_center_start, circle_center_end, self.sample_frames)
        square_centers = self.interpolate_points(
            square_center_start, square_center_end, self.sample_frames)
        triangle_centers = self.interpolate_points(
            triangle_center_start, triangle_center_end, self.sample_frames)
        pentagon_centers = self.interpolate_points(
            pentagon_center_start, pentagon_center_end, self.sample_frames)
        ellipse_centers = self.interpolate_points(
            ellipse_center_start, ellipse_center_end, self.sample_frames)
        image = np.zeros((128, 128, 3), dtype=np.uint8)

        print("Instruction : ", instruction)
        for i in range(self.sample_frames):
            # Create a blank image
            image = np.zeros((128, 128, 3), dtype=np.uint8)

            # Draw shapes
            image = self.draw_circle(circle_centers[i], image, 15, Color.RED)
            image = self.draw_square(square_centers[i], 20, image, Color.GREEN)
            image = self.draw_triangle(
                triangle_centers[i], 30, image, Color.BLUE)
            image = self.draw_pentagon(
                pentagon_centers[i], 20, image, Color.CYAN)
            image = self.draw_ellipse(
                ellipse_centers[i], image, (10, 20), 45, 0, 360, Color.MAGENTA)
            cv2.imwrite(f'shapes{i}.png', image)

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

        start_shape = self.shapes_start_end[idx][0]
        end_shape = self.shapes_start_end[idx][1]
        instruction = f"move the {start_shape} to {end_shape}"

        circle_center_start = (30, 30)
        circle_center_end = (30, 30)

        square_center_start = (70, 30)
        square_center_end = (70, 30)

        triangle_center_start = (110, 40)
        triangle_center_end = (110, 40)

        pentagon_center_start = (40, 90)
        pentagon_center_end = (40, 80)

        ellipse_center_start = (100, 100)
        ellipse_center_end = (100, 100)

        start_shape_name = start_shape.split(" ")[1]
        end_shape_name = end_shape.split(" ")[1]

        if end_shape_name == "circle":
            end_shape_location = circle_center_end
        elif end_shape_name == "square":
            end_shape_location = square_center_end
        elif end_shape_name == "triangle":
            end_shape_location = triangle_center_end
        elif end_shape_name == "pentagon":
            end_shape_location = pentagon_center_end
        elif end_shape_name == "ellipse":
            end_shape_location = ellipse_center_end

        if start_shape_name == "circle":
            circle_center_end = end_shape_location
        elif start_shape_name == "square":
            square_center_end = end_shape_location
        elif start_shape_name == "triangle":
            triangle_center_end = end_shape_location
        elif start_shape_name == "pentagon":
            pentagon_center_end = end_shape_location
        elif start_shape_name == "ellipse":
            ellipse_center_end = end_shape_location

        # Modify shape end locations based on the instructions

        circle_centers = self.interpolate_points(
            circle_center_start, circle_center_end, self.sample_frames)
        square_centers = self.interpolate_points(
            square_center_start, square_center_end, self.sample_frames)
        triangle_centers = self.interpolate_points(
            triangle_center_start, triangle_center_end, self.sample_frames)
        pentagon_centers = self.interpolate_points(
            pentagon_center_start, pentagon_center_end, self.sample_frames)
        ellipse_centers = self.interpolate_points(
            ellipse_center_start, ellipse_center_end, self.sample_frames)
        image = np.zeros((128, 128, 3), dtype=np.uint8)

        for i in range(self.sample_frames):
            # Create a blank image
            image = np.zeros((128, 128, 3), dtype=np.uint8)

            # Draw shapes
            image = self.draw_circle(circle_centers[i], image, 15, Color.RED)
            image = self.draw_square(square_centers[i], 20, image, Color.GREEN)
            image = self.draw_triangle(
                triangle_centers[i], 30, image, Color.BLUE)
            image = self.draw_pentagon(
                pentagon_centers[i], 20, image, Color.CYAN)
            image = self.draw_ellipse(
                ellipse_centers[i], image, (10, 20), 45, 0, 360, Color.MAGENTA)
            # cv2.imwrite(f'shapes{i}.png', image)
            episode.append(image)

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

        # condition_frames = []
        # for i, frame in enumerate([cond_image_1, cond_image_2]):
        #     # Resize the image and convert it to a tensor
        #     img_resized = cv2.resize(frame, (self.width, self.height))

        #     # img_resized = np.resize(frame, (self.height, self.width, 3))
        #     img_tensor = torch.from_numpy(img_resized).float()

        #     # Normalize the image by scaling pixel values to [-1, 1]
        #     img_normalized = img_tensor / 127.5 - 1

        #     # Rearrange channels if necessary
        #     if self.channels == 3:
        #         img_normalized = img_normalized.permute(
        #             2, 0, 1)  # For RGB images
        #     elif self.channels == 1:
        #         img_normalized = img_normalized.mean(
        #             dim=2, keepdim=True)  # For grayscale images

        #     condition_frames.append(img_normalized)

        return {'pixel_values': pixel_values, "task_text": instruction, "original_episode": episode}
