import torch

from diffusers import UNetSpatioTemporalConditionModel, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
from PIL import Image
import io
import h5py
import wandb
import math
import time
import numpy as np

unet = UNetSpatioTemporalConditionModel.from_pretrained(
    "/storage/nfs/jpatel/svd_checkpoints/franka_aug_5_ckpt_bkp/checkpoint-293000/unet",
    subfolder="unet",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    unet=unet,
    low_cpu_mem_usage=False,
    torch_dtype=torch.float16, variant="fp16", local_files_only=False,
)
pipe.to("cuda:0")

timestr = time.strftime("%Y%m%d-%H%M%S")
wandb_run = wandb.init(
        project="teco_inference",
        id=timestr,
        job_type="inference",
        group="inference_franka",
    )

for video_idx in range(30):
    file = f"/workspaces/bdai/pick_and_place_20240806_185358/demo_{video_idx}/world_state.hdf5"
    hdf  = h5py.File(file, 'r')
    camera_data = hdf['camera0_rgb']
    image = Image.open(io.BytesIO(camera_data[0]))
    gt_image = image.resize((512, 320))
    generator = torch.manual_seed(-1)
    num_pred_frames = (camera_data.shape[0] * 2)
    with torch.inference_mode():
        all_frames = []
        num_frames = 25
        start_time = time.time()
        for idx in range(math.ceil(num_pred_frames/num_frames)):
            pred_frames = pipe(gt_image,
                        num_frames=25,
                        width=512,
                        height=320,
                        decode_chunk_size=8, generator=generator, motion_bucket_id=127, fps=8, num_inference_steps=30).frames[0]
                        
            all_frames.extend(pred_frames)
            gt_image = pred_frames[-1]

    video_frames = [np.array(frame) for frame in all_frames]
    
    # Log SVD Predictions
    pred_key = f"visualize/franka/{video_idx}/pred_franka_293"
    video_frames = [np.array(frame) for frame in all_frames]
    pred_svd_video = wandb.Video(
                np.stack(video_frames).transpose(0,3,1,2),
                fps=30,
            )

    wandb.log({pred_key: pred_svd_video}, commit=False)
    

