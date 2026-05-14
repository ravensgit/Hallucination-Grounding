# Grounding Image Captions: Reducing Hallucination in Vision-Language Models

**Authors:** Naveen Manikandan (manikan2@buffalo.edu) | Varun Teja Goud Madhagouni (varuntej@buffalo.edu)  
**Course:** CSE 676, University at Buffalo  
**Date:** May 2026

---

## Overview

This project investigates hallucination in image captioning models and proposes a training-free caption selection method to reduce hallucinated object mentions. We treat hallucination as a **selection problem** rather than a generation problem: given multiple candidate captions from BLIP, we select the most visually grounded one using a hybrid scoring function.

**Key Result:** On 5,000 COCO val2017 images, our method reduces CHAIRs (Caption Hallucination Assessment with Image Relevance) from 4.08% to 0.74%, an **82% relative reduction** in hallucination, without retraining any model weights.

---

## Project Structure

```
hallucination-grounding/
├── milestone_2.py              # M2 pipeline (500 images, baseline implementation)
├── milestone_2.ipynb           # M2 Jupyter notebook (original Colab version)
├── milestone_3.py              # M3 pipeline (5000 images, ablation study)
│
├── Milestone_3_Report.pdf      # Final NeurIPS-style report (4 pages)
├── Milestone_3_Report.tex      # LaTeX source for report
│
├── data/
│   └── coco/
│       ├── val2017/            # COCO val2017 images (auto-downloaded)
│       └── annotations/        # COCO annotations (auto-downloaded)
│
├── results_m3/                 # Milestone 3 outputs
│   ├── coco_baseline_captions.csv          # Greedy BLIP captions (5000 images)
│   ├── coco_sampled_captions.csv           # 5 nucleus-sampled alternatives per image
│   ├── coco_sample_variation.csv           # CHAIR across sample positions
│   ├── coco_ablation_results.csv           # 7 ablation configurations
│   │
│   ├── coco_baseline_chair.csv             # Detailed hallucination analysis (baseline)
│   ├── coco_clip_chair.csv                 # Detailed hallucination analysis (CLIP-only)
│   ├── coco_hybrid_chair.csv               # Detailed hallucination analysis (hybrid)
│   │
│   ├── coco_main_comparison.png            # Figure 1: Baseline vs CLIP-only vs Hybrid
│   ├── coco_ablation_chairs.png            # Figure 2: Ablation sensitivity analysis
│   └── coco_qualitative_fixed.png          # Figure 3: Qualitative examples
│
└── chair_official/             # CHAIR synonyms and grounding tools (auto-cloned)
```

---

## Method

### Caption Generation & Selection Pipeline

For each image:

1. **Generate 6 candidate captions** using BLIP:
   - 1 greedy caption (beam search, `sample=False`)
   - 5 nucleus-sampled captions (`top_p=0.9`, `sample=True`)

2. **Score each candidate** using:
   ```
   Score = α · CLIP(img, cap) − β · ObjectPenalty(cap) − γ · LengthPenalty(cap)
   ```
   - **α = 1.0:** CLIP visual alignment (cosine similarity, normalized to [0,1])
   - **β = 0.3:** Object mention penalty (unique COCO categories / 10)
   - **γ = 0.2:** Length penalty (max(0, words − 10) / 20)

3. **Select** the caption with the highest score.

### Why This Works

- **Object penalty** is the dominant factor (ablation shows 0.90% CHAIRs with β=0.3, γ=0 vs 5.10% for CLIP-only)
- **CLIP-only selection hurts performance** — visual similarity alone rewards plausible but hallucinated captions
- **Penalty combination stabilizes weight choices** — ablation shows robustness to modest variations (0.64%–1.26% over [weak, default, strong])

---

## Key Results

### Main Findings (5,000 COCO val2017 images)

| Method | CHAIRs | CHAIRi | Avg. Length | Avg. Objects |
|--------|--------|--------|-------------|--------------|
| Baseline (greedy) | 4.08% | 2.63% | 9.69 | 1.59 |
| CLIP-only selection | 5.10% | 3.48% | 10.24 | 1.51 |
| **Hybrid (ours)** | **0.74%** | **0.74%** | 9.25 | 1.02 |

**Interpretation:** 82% relative reduction in CHAIRs. The 82% improvement dwarfs the 36% object reduction, proving the hybrid **selects grounded captions**, not just short ones.

### Ablation Study

| Configuration | α | β | γ | CHAIRs |
|---------------|---|---|---|--------|
| baseline_greedy | – | – | – | 4.08% |
| clip_only | 1.0 | 0.0 | 0.0 | 5.10% |
| length_penalty_only | 1.0 | 0.0 | 0.2 | 4.28% |
| object_penalty_only | 1.0 | 0.3 | 0.0 | **0.90%** |
| **hybrid_default** | 1.0 | 0.3 | 0.2 | **0.74%** |
| weak_penalty | 1.0 | 0.15 | 0.1 | 1.26% |
| strong_penalty | 1.0 | 0.5 | 0.3 | 0.64% |

**Key insight:** Object penalty does ~90% of the work. The diminishing return (0.64% vs 0.74%) shows empirically-tuned defaults occupy a natural operating point where hallucination reduction plateaus.

---

## Setup & Installation

### Requirements

- **Python 3.8+**
- **GPU recommended** (NVIDIA CUDA 11.0+)

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.35.2 timm fairscale inflect pandas matplotlib pillow
```

### Clone BLIP

The script automatically clones BLIP from the official Salesforce repository and patches the import if needed.

### Download COCO

The script automatically downloads:
- **COCO val2017 images** (~800 MB)
- **COCO annotations** (~240 MB)

Set `USE_RANDOM_COCO_SUBSET = False` in the code to use the first 5,000 images (default).

---

## Running the Code

### Milestone 3 Pipeline (5,000 images)

```bash
python milestone_3.py
```

**What it does:**
1. Downloads COCO val2017 and annotations (if needed)
2. Clones BLIP and CHAIR repos (if needed)
3. Generates 6 captions per image (greedy + 5 sampled)
4. Evaluates baseline, CLIP-only, and hybrid selection
5. Runs 7-config ablation study
6. Saves results to `~/hallucination-grounding/results_m3/`
7. Generates Figures 1, 2, 3 as PNG files

**Runtime:** ~2-3 hours on GPU (V100 or better)

**Output files:**
- CSV files with captions and CHAIR scores
- PNG figures for the report
- Detailed CHAIR analysis with hallucinated objects

### Milestone 2 Pipeline (500 images)

```bash
python milestone_2.py
```

Similar structure but evaluates only the first 500 COCO images. Used for initial validation in M2.

---

## Reproducibility

### Random Seeds

All random states are fixed at **seed = 42**:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

### Model Versions

- **BLIP:** `model_base_caption_capfilt_large.pth` (auto-downloaded from Salesforce)
- **CLIP:** `openai/clip-vit-base-patch32` (auto-downloaded from Hugging Face)
- **Transformers:** `4.35.2` (pinned in requirements)

### Hyperparameters

All hyperparameters are hardcoded and documented at the top of each script:

```python
RANDOM_SEED        = 42
COCO_NUM_IMAGES    = 5000      
NUM_SAMPLES        = 5          
TOP_P              = 0.9
MAX_LENGTH         = 20
MIN_LENGTH         = 5
DEFAULT_ALPHA      = 1.0
DEFAULT_BETA       = 0.3
DEFAULT_GAMMA      = 0.2
```

To reproduce exact results, use these settings without modification.

---

## Evaluation Metrics

### CHAIR (Caption Hallucination Assessment with Image Relevance)

We implement a CHAIR-style evaluation:

1. **Extract objects** from captions using:
   - Lowercasing and word singularization
   - Synonym mapping to official COCO 80 categories (403-entry synonym list from CHAIR repo)
   - Bigram matching for compound objects

2. **Ground truth** is constructed by combining:
   - Segmentation annotations from `instances_val2017.json`
   - Object mentions from reference captions in `captions_val2017.json`
   - Mapped to COCO categories via synonym list

3. **Metrics:**
   - **CHAIRs (sentence-level):** % of captions with ≥1 hallucinated object
   - **CHAIRi (instance-level):** % of object mentions that are hallucinated

### Limitations

- CHAIR depends on completeness of ground-truth object set
- If a real object is missing from annotations, a correct mention is counted as hallucination
- We follow the CHAIR paper's methodology but do not claim exact replication of their official metric

---

## Findings & Insights

### What Worked

- **Caption selection at inference time significantly reduces hallucination** without retraining
- **Object-aware constraints are critical** — the object penalty does ~90% of the work
- **The method is stable** — ablation shows robustness across weight variations

### What Didn't Work

- **CLIP-only selection performs worse than baseline** (5.10% vs 4.08% CHAIRs)
- Visual similarity alone reflects plausibility, not factual grounding
- CLIP rewards "sound right" captions that confidently invent objects

### Important Discovery: CLIP Bias on Flickr30k

During Milestone 3, we attempted to validate on Flickr30k but discovered:

- CLIP was trained on a large internet corpus **believed to include** Flickr images
- This **likely introduces evaluation bias** where CLIP's preferences reflect its training distribution
- CLIP-based selection on Flickr30k likely does not provide reliable evidence for hallucination reduction

**Lesson:** Grounding metrics ideally require models trained on disjoint data, or datasets with independent hallucination annotations.

---

## Limitations & Future Work

### Current Limitations

1. **Hand-picked weights (β=0.3, γ=0.2):** Tuned empirically in M2; ablation shows stability but learning them end-to-end might improve results
2. **Object-count heuristic:** Simple count-based penalty; explicit object detection would be more direct
3. **COCO-only evaluation:** CHAIR requires COCO-style annotations; cross-dataset validation is challenging
4. **CLIP bias:** Our method uses CLIP, which may introduce bias on datasets seen during its pretraining

### Future Directions

1. **Replace object-count heuristic** with explicit object detection from models not trained on target dataset
2. **Learn penalty weights end-to-end** via held-out validation set
3. **Test on other captioning models** (BLIP-2, GIT, OFA) using COCO
4. **Acquire hallucination annotations for out-of-distribution datasets** (Flickr30k, Nocaps, etc.)
5. **Investigate alternative grounding signals** (spatial attention, bounding box alignment, etc.)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{manikandan2026grounding,
  title={Grounding Image Captions: Reducing Hallucination in Vision-Language Models},
  author={Manikandan, Naveen and Madhagouni, Varun Teja Goud},
  school={University at Buffalo},
  year={2026},
  howpublished={\url{https://github.com/naveen-m/hallucination-grounding}}
}
```

---

## References

- **BLIP:** Li et al. (2022). "Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation." arXiv:2201.12086
- **CHAIR:** Rohrbach et al. (2018). "Object Hallucination in Image Captioning." EMNLP. arXiv:1809.02156
- **CLIP:** Radford et al. (2021). "Learning Transferable Models for Unsupervised Domain Adaptation via Self-Supervision." ICML.
- **COCO Dataset:** Lin et al. (2014). "Microsoft COCO: Common Objects in Context." ECCV.

---

## Troubleshooting

### BLIP model download fails
- Check internet connection and available disk space (~2 GB)
- The script auto-retries; you can manually download from [Salesforce BLIP repo](https://github.com/salesforce/BLIP)

### COCO annotations not found
- Ensure `~/hallucination-grounding/data/coco/annotations/` exists
- Run the script again; it will re-download if needed

### CUDA out of memory
- Reduce `NUM_SAMPLES` from 5 to 3 (will affect ablation)
- Use CPU (slower): remove `.to(device)` calls or set `device = torch.device("cpu")`

### Results don't match report
- Verify seed 42 is set throughout
- Check that `transformers==4.35.2` is installed exactly
- Ensure BLIP and CLIP model URLs haven't changed

---

## Contact

For questions or issues, contact:
- **Naveen Manikandan:** manikan2@buffalo.edu
- **Varun Teja Goud Madhagouni:** varuntej@buffalo.edu

---

## License

This project is provided for educational purposes as part of CSE 676 at University at Buffalo.
