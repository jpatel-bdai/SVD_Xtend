export CUDA_VISIBLE_DEVICES=1 && accelerate launch train_svd_three_frame_cond.py \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=500000 \
    --width=512 \
    --height=320 \
    --checkpointing_steps=1000 --checkpoints_total_limit=1 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=50
    
# --resume_from_checkpoint=./output_test/checkpoint-3000 \
