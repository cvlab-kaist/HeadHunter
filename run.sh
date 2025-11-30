torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=6 \
  pipeline_distributed.py \
  --enable_wandb \
  --num_layers 24 \
  --num_heads 24 \
  --top_k 3 \
  --num_iterations 5 \
  --prompt_path_style ./prompts_style.txt \
  --prompt_path_content ./prompts_content.txt \
  --output_root output_iterative \
  --wandb_api_key <your_wandb_api_key>
