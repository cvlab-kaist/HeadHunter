# Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models (NeurIPS 2025)

<a href="https://cvlab-kaist.github.io/HeadHunter/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2506.1097"><img src="https://img.shields.io/badge/arXiv-2506.1097-b31b1b.svg" alt="arXiv"></a>

**Official PyTorch implementation of "Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models".**

**HeadHunter** is a framework designed to steer diffusion models more effectively by selecting and combining individual attention heads. It addresses the observation that when perturbation guidance (like PAG) is applied at the head level, effects vary significantly, with some heads exhibiting interpretable characteristics related to specific visual concepts.

This repository provides a **multi-GPU distributed inference pipeline** for HeadHunter. The pipeline performs iterative attention head perturbation and ranking using image scoring models, with per-iteration validation and memory-efficient model handling.

---

## 🚀 Key Features

### Method
- 🧠 **Head-level Specialization:** Exploits the distinct visual effects (structure, lighting, texture, style) captured by individual attention heads.
- 🔀 **Compositional Power:** Combines multiple heads to compose complex visual effects.
- 🎯 **Objective-Aware Guidance:** Iteratively finds the best set of heads to maximize arbitrary objectives (e.g., PickScore, CLIPScore).
- 🌫️ **SoftPAG:** Implements interpolation-based perturbation to control strength and prevent oversaturation.

### Pipeline
- ⚡ **Distributed Generation/Inference:** Efficiently distributes attention head perturbation tasks across layers and GPUs (Supports SD3 & FLUX.1-Dev).
- 📊 **Iterative Top-K Selection:** Automatically selects the best heads using:
  - PickScore (Used in paper)
  - CLIPScore
  - Harmonic mean of Pick + CLIP
- 🧪 **Validation:** Validation performed before and after every iteration.

---

## 📦 Requirements

```bash
conda env create -f headhunter.yaml
conda activate headhunter
cd diffusers
pip install -e ".[torch]"
```

---

## 🧾 Usage: HeadHunter Pipeline

We provide scripts for both Stable Diffusion 3 (SD3) and FLUX.1-Dev.

```bash
bash run.sh      # For SD3
bash run_flux.sh # For FLUX.1-Dev (512x512)

# Please modify 'prompts_style.txt' as needed.
# Please add  <your_wandb_api_key> in shell script.
```

### Arguments

| Argument                  | Description                                            | Default               |
|---------------------------|--------------------------------------------------------|------------------------|
| `--num_layers`            | Number of transformer layers                           | `24` (SD3)             |
| `--num_heads`             | Number of attention heads per layer                   | `24` (SD3)             |
| `--top_k`                 | Number of top heads selected per iteration            | `3`                    |
| `--num_iterations`        | Total number of iterative steps                       | `5`                    |
| `--prompt_path_style`     | Path to the file containing style prompts             | `prompts_style.txt`    |
| `--prompt_path_content`   | Path to the file containing content prompts           | `prompts_content.txt`  |
| `--output_root`           | Directory to store all generated outputs              | `output_iterative`     |
| `--method`                | Method to rank attention heads                        | `pickscore`            |
|                           | Options: `pickscore`, `pickscore_thresholding_clip`, `harmonic_pick_clip` |                        |
| `--guidance_scale`        | Perturbation guidance scale for attention heads       | `3.0`                  |

---

## 📁 Output Directory Structure

```text
output_iterative/
  _style_prompt/
    iter0/
      content_prompt/
        layerL_headH.png
    iter1/
      ...
    validation/
      iter0/
        content0.png
      ...
      final_val.png   # Grid image of all validation results
    final/
      content_prompt/
        final_perturb_heads.pkl
        final_perturb_heads.txt
```

---

## 🔧 Supported Perturbations

The perturbation logic is implemented in `diffusers/src/diffusers/models/attention_processor.py`. You can specify the behavior using the `perturb_type` string.

### 1. Perturbation Types (`perturb_type`)

These define the target attention map ($A'$) that acts as the negative guidance or perturbation target.

#### Probabilistic Perturbations (Post-Softmax) (`[PROB_PERTURB]`)
Modifies the attention map **after** the softmax operation.

| Syntax | Description | Method Reference |
| :--- | :--- | :--- |
| `[PROB_PERTURB]attention_identity` | Replaces the attention map with the **Identity Matrix** ($I$). | **PAG** [1] |
| `[PROB_PERTURB]uniform` | Replaces the attention map with a **Uniform Matrix** ($U$). | **Uniform Guidance** (Appendix C.1) |
| `[PROB_PERTURB]II_identity_IT_mask` | Sets Image-Image attention to Identity, and masks Image-Text attention. | |
| `[PROB_PERTURB]zeroout+II_mask` | Zeros out the **Image-Image** block of the attention map. | |
| `[PROB_PERTURB]zeroout+IT_mask` | Zeros out the **Image-Text** block of the attention map. | |
| `[PROB_PERTURB]zeroout+II_identity` | Masks Image-Image block to be diagonal (Identity), zeroes off-diagonals. | |

#### Logit Perturbations (Pre-Softmax) (`[LOGIT_PERTURB]` / Others)
Modifies the attention scores (logits) **before** the softmax operation.

| Syntax | Description | Method Reference |
| :--- | :--- | :--- |
| `[LOGIT_PERTURB]II_identity` | Forces Image-Image attention to Identity by setting off-diagonal logits to $-\infty$. | |
| `[LOGIT_PERTURB]II_mask` | Masks Image-Image attention (logits $-\infty$). | |
| `smoothed_energy@all` | Replaces query with mean query over **all tokens** (Global averaging). | **SEG** [22] (Appendix C.2) |
| `smoothed_energy@img` | Replaces query with mean query over **image tokens**. | SoftSEG (Appendix C.2) |
| `smoothed_energy@txt` | Replaces query with mean query over **text tokens**. | SoftSEG (Appendix C.2) |
| `temp_control@temperature={T}` | Scales logits by temperature $T$. (e.g., `temp_control@temperature=0.5`). | **Max Guidance** (Appendix C.3) |

---

### 2. SoftPAG: Continuous Perturbation Strength

You can control the strength of the perturbation by **interpolating** between the original attention map ($A$) and the perturbed target ($A'$).

**Syntax:** Append `@scale={u}` to the `perturb_type`, where `u` is the interpolation factor ($0.0 \le u \le 1.0$).
*   $u=0.0$: Original Attention ($A$).
*   $u=1.0$: Full Perturbation ($A'$).

#### Interpolation Methods
By default, **Linear Interpolation** is used. You can specify other geodesic paths by appending keywords:

| Interpolation Type | Syntax Example | Formula |
| :--- | :--- | :--- |
| **Linear** (Default) | `@scale=0.5` | $A_{final} = (1-u)A + uA'$ |
| **Fisher-Rao** | `@scale=0.5+fisher_rao` | Geodesic on the statistical manifold (Appendix B). |
| **Slerp** | `@scale=0.5+fisher_rao_slerp` | Spherical Linear Interpolation. |
| **Log-Linear** | `@scale=0.5+log_linear` | Interpolation in log-space (e-connection). |

**Example:**
`[PROB_PERTURB]attention_identity@scale=0.5` implements **SoftPAG** with 50% strength.

---

### 3. Head-Level Perturbation

You can target specific attention heads for perturbation, as introduced in **HeadHunter**.

*   **Via Pipeline Argument:** Pass `perturb_heads=[0, 1, 5]` to the pipeline call.

---

## References
*   **PAG**: [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance](https://arxiv.org/abs/2312.00000)
*   **SEG**: [Smoothed Energy Guidance](https://arxiv.org/abs/2312.00000)
*   **HeadHunter / SoftPAG**: [Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models](https://cvlab-kaist.github.io/HeadHunter/)

---

## 📜 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ahn2025headhunter,
  title={Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models},
  author={Ahn, Donghoon and Kang, Jiwon and Lee, Sanghyun and Kim, Minjae and Jang, Wooseok and Min, Jaewon and Lee, Sangwu and Paul, Sayak and Kim, Seungryong},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```