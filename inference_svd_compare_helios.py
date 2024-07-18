import torch
import wandb
from pathlib import Path
import imageio
import numpy as np
from PIL import Image
import time
import json
import matplotlib.pyplot as plt
import math
from diffusers import UNetSpatioTemporalConditionModel, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
import cv2
import metrics


helios_dir = Path("/storage/nfs/jpatel/helios_rtx_baseline_results/run-20240716_213553-20240716-213402/files/media")
plotly_key = "plotly/visualize/test/"
videos_key = "videos/visualize/test/"
unet = UNetSpatioTemporalConditionModel.from_pretrained(
    # "/storage/nfs/jpatel/svd_checkpoints/bdai_datasets_highresfps_ckpt_bkp/checkpoint-500000/unet",
    "/storage/nfs/jpatel/svd_checkpoints/rtx_ckpt_bkp/checkpoint-142000/unet",
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
        job_type="testing",
        group="inference_comparison",
    )

def gif_file_to_np(gif_file):
    gif = imageio.mimread(gif_file)
    gif_np = np.array(gif)
    gif_np = np.transpose(gif_np, (0, 3, 1, 2))
    return gif_np

helios_lpips_all = []
helios_psnr_all = []
helios_ssim_all = []
svd_lpips_all = []
svd_psnr_all = []
svd_ssim_all = []

for video_idx in range(100):
    # READ DATA FROM HELIOS RESULTS
    plots_dir = helios_dir / plotly_key / str(video_idx)
    lpips_file = next(plots_dir.glob("lpips*"),None).open('r')
    psnr_file = next(plots_dir.glob("psnr*"),None).open('r')
    ssim_file = next(plots_dir.glob("ssim*"),None).open('r')
    
    lpips_data = json.load(lpips_file)
    psnr_data = json.load(psnr_file)
    ssim_data = json.load(ssim_file)
    
    lpips_x, lpips_y = lpips_data['data'][0]['x'], lpips_data['data'][0]['y']
    psnr_x, psnr_y = psnr_data['data'][0]['x'], psnr_data['data'][0]['y']
    ssim_x, ssim_y = ssim_data['data'][0]['x'], ssim_data['data'][0]['y']
    helios_lpips_all.append(lpips_y)
    helios_psnr_all.append(psnr_y)
    helios_ssim_all.append(ssim_y)
    
    videos_dir = helios_dir / videos_key / str(video_idx)
    gt_file = next(videos_dir.glob("gt*"),None)
    pred_file = next(videos_dir.glob("pred*"),None)
    pred_ar_file = next(videos_dir.glob("pred_ar*"),None)
    
    gt_np = gif_file_to_np(gt_file)
    pred_ar_np = gif_file_to_np(pred_ar_file)
    
    # GENERATE FRAMES FOR SVD
    first_frame = Image.fromarray(gt_np[0].transpose(1, 2, 0))    
    # image = load_image('dalle3_cat.jpg')
    gt_image = first_frame.resize((512, 320))
    
    generator = torch.manual_seed(-1)
    with torch.inference_mode():
        all_frames = []
        num_pred_frames = gt_np.shape[0]
        num_frames = 25
        for idx in range(math.ceil(num_pred_frames/num_frames)):
            pred_frames = pipe(gt_image,
                        num_frames=25,
                        width=512,
                        height=320,
                        decode_chunk_size=8, generator=generator, motion_bucket_id=127, fps=8, num_inference_steps=30).frames[0]
                        
            all_frames.extend(pred_frames)
            gt_image = pred_frames[-1]

    # CALCULATE METRICS FOR SVD
    result_video = []
    for i in range(num_pred_frames):
        img = all_frames[i]
        all_frames[i] = np.array(img)
        img_resized = cv2.resize(all_frames[i], (128, 128), interpolation=cv2.INTER_AREA)
        result_video.append(img_resized)
    
    pred_result_video = np.array(result_video).transpose(0, 3, 1, 2)
    # pred_video = np.array(all_frames).transpose(0, 3, 1, 2)

    # Compute only over the generated part
    gtt = torch.from_numpy(gt_np/ 255.0).unsqueeze(0).to("cuda:0")[:,1:,:]
    predt = torch.from_numpy(pred_result_video/ 255.0).unsqueeze(0).to("cuda:0")[:,1:,:]
    
    predt = predt.to(gtt.dtype)
    print(f"pred_video : {pred_result_video.shape}")
    
    # Compute metrics
    fvd = metrics.FVD(device="cuda:0")
    ssim = metrics.get_ssim()
    psnr = metrics.get_psnr()
    lpips = metrics.get_lpips(device="cuda:0")
    # breakpoint()
    predt = predt.to(torch.float32)
    gtt = gtt.to(torch.float32)
    
    svd_ssim_plot = [ssim(predt[:,i:i+1], gtt[:,i:i+1])[0].detach().cpu().numpy() for i in range(gtt.shape[1])]
    svd_ssim_plot = [float(value) for value in svd_ssim_plot]
    svd_lpips_plot = [lpips(predt[:,i:i+1], gtt[:,i:i+1])[0].detach().cpu().numpy() for i in range(gtt.shape[1])]
    svd_lpips_plot = [float(value) for value in svd_lpips_plot]
    svd_psnr_plot = [psnr(predt[:,i:i+1], gtt[:,i:i+1])[0].detach().cpu().numpy() for i in range(gtt.shape[1])]
    svd_psnr_plot = [float(value) for value in svd_psnr_plot]
    
    svd_lpips_all.append(svd_lpips_plot)
    svd_psnr_all.append(svd_psnr_plot)
    svd_ssim_all.append(svd_ssim_plot)
    
    # Log ground truth
    gt_key = f"visualize/test/{video_idx}/gt"
    gt_val = wandb.Video(
                gt_np,
                fps=30,
            )
    wandb.log({gt_key: gt_val}, commit=False)
    
    # Log SVD Predictions
    pred_key = f"visualize/test/{video_idx}/pred_svd"
    video_frames = [np.array(frame) for frame in all_frames]
    pred_svd_video = wandb.Video(
                np.stack(video_frames).transpose(0,3,1,2),
                fps=30,
            )
    wandb.log({pred_key: pred_svd_video}, commit=False)
    
    # Log Helios Predictions
    pred_ar_key = f"visualize/test/{video_idx}/pred_helios_ar"
    pred_ar_val = wandb.Video(
                pred_ar_np,
                fps=30,
            )
    wandb.log({pred_ar_key: pred_ar_val}, commit=False)


    # Log metrics
    fig, ax = plt.subplots()
    ax.plot(lpips_y, label='helios', color='green')
    ax.plot(svd_lpips_plot, label='svd', color='red')
    ax.set_ylabel("LPIPS")
    # Log the LPIPS plot
    wandb.log({f"visualize/test/{video_idx}/lpips": fig})

    fig, ax = plt.subplots()
    ax.plot(ssim_y, label='helios', color='green')
    ax.plot(svd_ssim_plot, label='svd', color='red')
    ax.set_ylabel("SSIM")
    # Log the SSIM plot
    wandb.log({f"visualize/test/{video_idx}/ssim": fig})
    
    fig, ax = plt.subplots()
    ax.plot(psnr_y, label='helios', color='green')
    ax.plot(svd_psnr_plot, label='svd', color='red')
    ax.set_ylabel("PSNR")
    # Log the PSNR plot
    wandb.log({f"visualize/test/{video_idx}/psnr": fig})
    
def get_average(metrics):
    max_seq_len = 0
    for idx in range(len(metrics)):
        max_seq_len = max(max_seq_len, len(metrics[idx]))
    
    avg_metrics = []
    for j in range(max_seq_len):
        metric_values = []
        for idx in range(len(metrics)):
            if j<len(metrics[idx]):
                metric_values.append(metrics[idx][j])
        avg_metrics.append(sum(metric_values)/len(metric_values))
    
    return avg_metrics

helios_lpips_avg = get_average(helios_lpips_all)
helios_psnr_avg = get_average(helios_psnr_all)
helios_ssim_avg = get_average(helios_ssim_all)
svd_lpips_avg = get_average(svd_lpips_all)
svd_psnr_avg = get_average(svd_psnr_all)
svd_ssim_avg = get_average(svd_ssim_all)

# Log metrics
fig, ax = plt.subplots()
ax.plot(helios_lpips_avg, label='helios', color='green')
ax.plot(svd_lpips_avg, label='svd', color='red')
ax.set_ylabel("LPIPS")
# Log the LPIPS plot
wandb.log({f"visualize/test/LPIPS": fig})

# Log metrics
fig, ax = plt.subplots()
ax.plot(helios_ssim_avg, label='helios', color='green')
ax.plot(svd_ssim_avg, label='svd', color='red')
ax.set_ylabel("SSIM")
# Log the SSIM plot
wandb.log({f"visualize/test/SSIM": fig})

# Log metrics
fig, ax = plt.subplots()
ax.plot(helios_psnr_avg, label='helios', color='green')
ax.plot(svd_psnr_avg, label='svd', color='red')
ax.set_ylabel("PSNR")
# Log the PSNR plot
wandb.log({f"visualize/test/PSNR": fig})
