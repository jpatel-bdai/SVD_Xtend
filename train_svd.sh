accelerate launch train_svd.py \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1823728 \
    --width=512 \
    --height=320 \
    --checkpointing_steps=1000 --checkpoints_total_limit=1 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=1000 \
    --num_validation_images=5



# distributed_type: 'NO'
# gpu_ids: '0'
# num_processes: 1

# distributed_type: MULTI_GPU
# gpu_ids: all
# num_processes: 4


# compute_environment: LOCAL_MACHINE
# debug: false
# distributed_type: 'NO'
# downcast_bf16: 'no'
# gpu_ids: '0'
# machine_rank: 0
# main_training_function: main
# mixed_precision: fp16
# num_machines: 1
# num_processes: 1
# rdzv_backend: static
# same_network: true
# tpu_env: []
# tpu_use_cluster: false
# tpu_use_sudo: false
# use_cpu: false

# compute_environment: LOCAL_MACHINE
# debug: true
# distributed_type: MULTI_GPU
# downcast_bf16: 'no'
# gpu_ids: all
# machine_rank: 0
# main_training_function: main
# mixed_precision: fp16
# num_machines: 1
# num_processes: 2
# rdzv_backend: static
# same_network: true
# tpu_env: []
# tpu_use_cluster: false
# tpu_use_sudo: false
# use_cpu: false
