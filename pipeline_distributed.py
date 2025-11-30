import os
import torch
import pickle
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from itertools import product
import matplotlib.pyplot as plt
from diffusers import StableDiffusion3Pipeline
from imscore.preference.model import CLIPScore
from imscore.pickscore.model import PickScorer
from imscore.aesthetic.model import LAIONAestheticScorer
import wandb
import datetime

# -------------------- Coonfigs --------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # wandb
    parser.add_argument("--enable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="headhunter") 
    parser.add_argument('--wandb_api_key', type=str, default='')
    parser.add_argument('--wandb_entity', type=str, default='diffusion-guidance')
    # Model config
    parser.add_argument('--model', type=str, default='sd3', choices=['sd3', 'flux'])
    parser.add_argument('--num_layers', type=int, default=24)
    parser.add_argument('--num_heads', type=int, default=24)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    # HeadHunter config
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--prompt_path_style', type=str, default='prompts_style.txt')
    parser.add_argument('--prompt_path_content', type=str, default='prompts_content.txt')
    parser.add_argument('--output_root', type=str, default='output_iterative')
    parser.add_argument('--method', type=str, default='pickscore', choices=['pickscore', 'pickscore_thresholding_clip', 'harmonic_pick_clip'])
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    return parser.parse_args()

args = parse_args()
args.output_root = os.path.abspath(args.output_root)
output_root = args.output_root
args.prompt_path_style = os.path.abspath(args.prompt_path_style)
args.prompt_path_content = os.path.abspath(args.prompt_path_content)
NUM_LAYERS = args.num_layers
NUM_HEADS = args.num_heads
TOP_K = args.top_k
NUM_ITERATIONS = args.num_iterations
METHOD = args.method
GUIDANCE_SCALE = args.guidance_scale
NUM_INFERENCE_STEPS = args.num_inference_steps

with open(args.prompt_path_style, "r") as f:
    style_prompts = [line.strip() for line in f]

with open(args.prompt_path_content, "r") as f:
    content_prompts = [line.strip() for line in f]
NUM_PROMPTS = len(content_prompts)

all_heads = list(product(range(NUM_LAYERS), range(NUM_HEADS)))

# -------------------- Util --------------------
def shortname(name):
    return name.replace(" ", "_")[:20]

def get_output_dir(style_prompt, iter_name, content_prompt):
    return os.path.join(output_root, f"_{shortname(style_prompt)}", iter_name, shortname(content_prompt))

def save_prompt_txt(output_dir, full_prompt):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
        f.write(full_prompt)

def save_final_heads(head_list, save_dir, output_root):
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, "final_perturb_heads.txt")
    pkl_path = os.path.join(save_dir, "final_perturb_heads.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(head_list, f)
    with open(txt_path, "w") as f:
        for (l, h) in head_list:
            f.write(f"{l},{h}\n")

    if args.enable_wandb:
        run.save(pkl_path, base_path=output_root)
        run.save(txt_path, base_path=output_root)

# -------------------- Validation --------------------
def run_validation_images(pipe, style_prompt, perturb_heads_2, val_idx):
    val_dir = os.path.join(output_root, f"_{shortname(style_prompt)}", "validation", f"iter{val_idx}")
    os.makedirs(val_dir, exist_ok=True)
    image_grid = []
    for p_idx, content_prompt in enumerate(content_prompts):
        full_prompt = f"{style_prompt}, {content_prompt}"
        seed = p_idx
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perturb_type = "[PROB_PERTURB]attention_identity@scale=1.0"
        images = pipe(
            full_prompt,
            negative_prompt="",
            height=args.height,
            width=args.width,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=0.0,
            generator=gen,
            return_dict=False,
            # return_pred_x0=False,
            perturb_heads=perturb_heads_2,
            perturb_type=perturb_type,
            perturb_guidance_scale=GUIDANCE_SCALE,
        )[0]
        img = images[0].convert("RGB")
        save_path = os.path.join(val_dir, f"content{p_idx}.png")
        img.save(save_path)
        image_grid.append([img])
        
        if args.enable_wandb:
            run.save(save_path, base_path=output_root)
    return image_grid

def save_final_grid(all_iters, style_prompt):
    val_dir = os.path.join(output_root, f"_{shortname(style_prompt)}", "validation")
    num_iters = len(all_iters)
    num_contents = len(content_prompts)
    fig, axs = plt.subplots(num_iters, num_contents, figsize=(3*num_contents, 3*num_iters))
    for i in range(num_iters):
        for j in range(num_contents):
            axs[i][j].imshow(all_iters[i][j][0])
            axs[i][j].axis('off')
            if i == 0:
                axs[i][j].set_title(f"Prompt {j}")
            if j == 0:
                axs[i][j].set_ylabel(f"Iter {i}")
    fig.suptitle(f"Style: {style_prompt}")
    plt.tight_layout()
    save_path = os.path.join(val_dir, "final_val.png")
    plt.savefig(save_path)
    plt.close()

    if args.enable_wandb:
        run.log({f"{shortname(style_prompt)}": wandb.Image(save_path)})
        print(f"Logged image to wandb: {save_path}, base_path: {output_root}")
        run.save(save_path, base_path=output_root)

# -------------------- Top-K --------------------
def select_topk_heads(gathered_score_list, method="pickscore", exclude_heads=[]):
    avg_score = {}
    
    for g in gathered_score_list:
        avg_score.update(g)

    exclude_set = set(exclude_heads)

    if method == "pickscore_thresholding_clip":
        qualified = [
            (key, score["clip"])
            for key, score in avg_score.items()
            if key not in exclude_set and score.get("pick", 0) >= 20 and score.get("laion", 0) >= 6
        ]
        sorted_heads = sorted(qualified, key=lambda x: -x[1])

    elif method == "pickscore":
        sorted_heads = sorted(
            [(k, v["pick"]) for k, v in avg_score.items() if k not in exclude_set and "pick" in v],
            key=lambda x: -x[1]
        )

    elif method == "harmonic_pick_clip":
        def harmonic(p, c):
            return 2 * p * c / (p + c) if (p + c) > 0 else 0
        sorted_heads = sorted(
            [
                (k, harmonic(v.get("pick", 0), v.get("clip", 0)))
                for k, v in avg_score.items() if k not in exclude_set
            ],
            key=lambda x: -x[1]
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return [pair for pair, _ in sorted_heads[:TOP_K]]

from diffusers import FluxPipeline
def load_pipe(model, device):
    
    if model == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to(device)
    elif model == "flux":
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    pipe.set_progress_bar_config(leave=False, disable=True)
    return pipe

# -------------------- Image Generation and Scoring --------------------

def generate_images(pipe, style_prompt, perturb_heads_2, iter_idx, assigned_layers, device):
    for layer in tqdm(assigned_layers, desc=f"[GPU {device}] Gen Layers", leave=False, disable=(dist.get_rank() != 0)):
        for head in tqdm(range(NUM_HEADS), desc=f"[GPU {device}] Gen Heads", leave=False, disable=(dist.get_rank() != 0)):
            for p_idx, content_prompt in tqdm(enumerate(content_prompts), desc=f"[GPU {device}] Gen Prompts", total=len(content_prompts), leave=False, disable=(dist.get_rank() != 0)):
                seed = p_idx
                full_prompt = f"{style_prompt}, {content_prompt}"
                gen = torch.Generator(device="cpu").manual_seed(seed)
                perturb_type = f"[PROB_PERTURB]attention_identity@scale=1.0"
                current_heads = perturb_heads_2 + [(layer, head)]
                images = pipe(
                    full_prompt,
                    height=args.height,
                    width=args.width,
                    negative_prompt="",
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=0.0,
                    generator=gen,
                    return_dict=False,
                    # return_pred_x0=False,
                    perturb_heads=current_heads,
                    perturb_type=perturb_type,
                    perturb_guidance_scale=GUIDANCE_SCALE,
                    # perturb_heads_2=perturb_heads_2,
                    # perturb_type_2=perturb_type,
                    # perturb_guidance_scale_2=GUIDANCE_SCALE,
                )[0]
                save_dir = os.path.join(get_output_dir(style_prompt, f"iter{iter_idx}", content_prompt))
                os.makedirs(save_dir, exist_ok=True)
                images[0].save(os.path.join(save_dir, f"layer{layer}_head{head}.png"))
                prompt_path = os.path.join(get_output_dir(style_prompt, "final", content_prompt), "prompt.txt")
                if not os.path.exists(prompt_path):
                    save_prompt_txt(os.path.dirname(prompt_path), full_prompt)



def score_all_images(style_prompt, iter_idx, assigned_layers, device, clip_model, pick_model, laion_model):
    raw_score = {}
    avg_score = {}

    for layer in tqdm(assigned_layers, desc=f"[GPU {device}] Score Layers", leave=False, disable=(dist.get_rank() != 0)):
        for head in tqdm(range(NUM_HEADS), desc=f"[GPU {device}] Score Heads", leave=False, disable=(dist.get_rank() != 0)):
            key = (layer, head)
            raw_score[key] = {"clip": [], "pick": [], "laion": []}
            for p_idx, content_prompt in tqdm(enumerate(content_prompts), desc=f"[GPU {device}] Score Heads", leave=False, disable=(dist.get_rank() != 0)):
                full_prompt = f"{style_prompt}, {content_prompt}"
                clip_prompt = f"A photo depicts {style_prompt}, {content_prompt}"
                img_path = os.path.join(get_output_dir(style_prompt, f"iter{iter_idx}", content_prompt), f"layer{layer}_head{head}.png")
                if not os.path.exists(img_path):
                    continue
                img = Image.open(img_path).convert("RGB")
                pixels = rearrange(torch.tensor(np.array(img)), "h w c -> 1 c h w") / 255.0
                pixels = pixels.to(device)
                raw_score[key]['clip'].append(clip_model.score(pixels, clip_prompt).item())
                raw_score[key]['pick'].append(pick_model.score(pixels, full_prompt).item())
                raw_score[key]['laion'].append(laion_model.score(pixels, full_prompt).item())
            avg_score[key] = {k: np.mean(v) for k, v in raw_score[key].items() if v}
    return avg_score

# -------------------- Distributed --------------------
def distributed_worker(rank, world_size, layer_splits):
    import gc
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    group = dist.init_process_group("nccl",
                                    rank=rank,
                                    world_size=world_size,
                                    timeout=datetime.timedelta(seconds=3600 * 10), # 10 hour timeout
                                    )

    # Initialize wandb
    if args.enable_wandb and rank == 0:
        wandb.login(key=args.wandb_api_key)
        global run
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            config=vars(args),
        )

        # Save style/content prompts
        print(f"Saving prompts to {args.prompt_path_style} and {args.prompt_path_content})")
        run.save(args.prompt_path_style)
        run.save(args.prompt_path_content)

    assigned_layers = layer_splits[rank]

    for style_prompt in style_prompts:
        if rank == 0:
            root_dir = os.path.join(output_root, f"_{shortname(style_prompt)}")
            os.makedirs(root_dir, exist_ok=True)
            # Save the style prompt
            with open(os.path.join(root_dir, "prompt_style.txt"), "w") as f:
                f.write(style_prompt)
            if args.enable_wandb:
                run.save(os.path.join(root_dir, "prompt_style.txt"), base_path=output_root)
            # Save the content prompts
            with open(os.path.join(root_dir, "prompts_content.txt"), "w") as f:
                for content_prompt in content_prompts:
                    f.write(f"{content_prompt}\n")
            if args.enable_wandb:
                run.save(os.path.join(root_dir, "prompts_content.txt"), base_path=output_root)
        dist.barrier(group=group)

        perturb_heads_2 = []
        all_val_images = []

        # --- Resume Logic Start ---
        resume_iteration = 0
        if rank == 0:
            style_output_dir = os.path.join(output_root, f"_{shortname(style_prompt)}")
            for i in range(NUM_ITERATIONS):
                iter_save_dir = os.path.join(style_output_dir, f"iter{i}")
                iter_head_file = os.path.join(iter_save_dir, "selected_heads.pkl")
                if os.path.exists(iter_head_file):
                    with open(iter_head_file, "rb") as f:
                        loaded_heads = pickle.load(f)
                        # Ensure no duplicates if resuming from a partial state
                        for h in loaded_heads:
                            if h not in perturb_heads_2:
                                perturb_heads_2.append(h)
                    resume_iteration = i + 1
                    print(f"Resuming for style '{style_prompt}' from iteration {resume_iteration}. Current perturb_heads: {perturb_heads_2}")
                else:
                    break
        # Broadcast resume_iteration and perturb_heads_2
        resume_info = [resume_iteration, perturb_heads_2]
        dist.broadcast_object_list(resume_info, src=0)
        resume_iteration = resume_info[0]
        perturb_heads_2 = resume_info[1]
        dist.barrier(group=group)
        # --- Resume Logic End ---

        if rank == 0 and resume_iteration == 0: # Only run initial validation if not resuming
            pipe = load_pipe(args.model, device)
            val_imgs = run_validation_images(pipe, style_prompt, perturb_heads_2, val_idx=0)
            all_val_images.append(val_imgs)
            del pipe
            torch.cuda.empty_cache()
            gc.collect()
        dist.barrier(group=group)

        for iter_idx in range(resume_iteration, NUM_ITERATIONS): # Start from resume_iteration
            if rank == 0:
                print(f"Iteration {iter_idx} for Style '{style_prompt}'")
            # ⬇️ Load SD3
            pipe = load_pipe(args.model, device)
            generate_images(pipe, style_prompt, perturb_heads_2, iter_idx, assigned_layers, device)
            # ⬇️ Release SD3
            del pipe
            torch.cuda.empty_cache()
            gc.collect()
            dist.barrier(group=group)

            # ⬇️ Load Score models
            clip_model = CLIPScore("openai/clip-vit-large-patch14").to(device)
            pick_model = PickScorer("yuvalkirstain/PickScore_v1").to(device)
            laion_model = LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic").to(device)
            local_score = score_all_images(style_prompt, iter_idx, assigned_layers, device, clip_model, pick_model, laion_model)
            # ⬇️ Release Score models
            del clip_model, pick_model, laion_model
            torch.cuda.empty_cache()
            gc.collect()
            dist.barrier(group=group)

            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_score)
            dist.barrier(group=group)
            if rank == 0:
                iter_output_dir = os.path.join(output_root, f"_{shortname(style_prompt)}", f"iter{iter_idx}")
                os.makedirs(iter_output_dir, exist_ok=True)
                with open(os.path.join(iter_output_dir, "scores.pkl"), "wb") as f:
                    pickle.dump(gathered, f)
                if args.enable_wandb:
                    print(f"Saving scores to {os.path.join(iter_output_dir, 'scores.pkl')}, base_path: {output_root}")
                    run.save(os.path.join(iter_output_dir, "scores.pkl"), base_path=output_root)


            if rank == 0:
                topk_heads = select_topk_heads(gathered, method=METHOD, exclude_heads=perturb_heads_2)
            else:
                topk_heads = None
            topk_container = [topk_heads]
            dist.broadcast_object_list(topk_container, src=0)
            topk_heads = topk_container[0]
            dist.barrier(group=group)

            for h in topk_heads:
                if h not in perturb_heads_2:
                    perturb_heads_2.append(h)

            # --- Save Iterative Heads ---
            if rank == 0:
                iter_save_dir = os.path.join(output_root, f"_{shortname(style_prompt)}", f"iter{iter_idx}")
                save_final_heads(perturb_heads_2, iter_save_dir, output_root) # Reuse save_final_heads for each iteration
                print(f"Heads for iteration {iter_idx} saved to {os.path.join(iter_save_dir, 'final_perturb_heads.pkl')}")
            dist.barrier(group=group)
            # --- End Save Iterative Heads ---

            # Validation
            if rank == 0:
                # ⬇️ Load SD3
                pipe = load_pipe(args.model, device)
                val_imgs = run_validation_images(pipe, style_prompt, perturb_heads_2, val_idx=iter_idx+1)
                all_val_images.append(val_imgs)
                # ⬇️ Release SD3
                del pipe
                torch.cuda.empty_cache()
                gc.collect()

            dist.barrier(group=group)

        if rank == 0:
            save_final_grid(all_val_images, style_prompt)
            print(f"\n🎯 Final Heads for Style '{style_prompt}': {perturb_heads_2}")
            save_dir = os.path.join(output_root, f"_{shortname(style_prompt)}", "final")
            save_final_heads(perturb_heads_2, save_dir, output_root)

    dist.destroy_process_group()


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    layer_splits = [list(range(NUM_LAYERS))[i::world_size] for i in range(world_size)]
    if rank == 0:
        print(f"[INFO] Using MODEL: {args.model}")
        print(f"[INFO] Using Height and Width: {args.height}x{args.width}")
        print(f"[INFO] World size: {world_size}")
        print(f"[INFO] Layer splits: {layer_splits}")

    print(f"[Rank {rank}] Starting distributed worker...")
    distributed_worker(rank, world_size, layer_splits)

if __name__ == "__main__":
    main()
