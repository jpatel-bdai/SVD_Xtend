# export TOKENIZERS_PARALLELISM=false && Set this false on error
accelerate launch train_svd_lang.py \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=5000000 \
    --width=128 \
    --height=128 \
    --checkpointing_steps=1000 --checkpoints_total_limit=1 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=1000 \
    --num_validation_images=10 \
    --output_dir=/storage/nfs/jpatel/svd_checkpoints/svd_lang_pos_emb \
    --conditioning_dropout_prob=0.1
