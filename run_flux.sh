torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    pipeline_distributed.py \
    --enable_wandb \
    --model flux \
    --height 512 \
    --width 512 \
    --guidance_scale 8.0 \
    --num_inference_steps 15 \
    --num_layers 57 \
    --num_heads 24 \
    --top_k 3 \
    --num_iterations 5 \
    --prompt_path_style ./prompts_style.txt \
    --prompt_path_content ./prompts_content.txt \
    --output_root ./output_flux_iterative \
  --wandb_api_key <your_wandb_api_key>

