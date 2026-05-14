# Quick Start Guide

Get up and running with the hallucination reduction pipeline in 5 minutes.

---

## 1. Environment Setup (5 min)

### Option A: Create a Fresh Python Environment

```bash
# Using venv
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Using conda
conda create -n hallucination python=3.9
conda activate hallucination
```

### Option B: Use existing Python 3.8+

```bash
python3 --version  # Must be 3.8 or higher
```

---

## 2. Install Dependencies (3 min)

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ML libraries (pinned versions for reproducibility)
pip install transformers==4.35.2 timm fairscale inflect pandas matplotlib pillow

# Optional (for notebooks)
pip install jupyter notebook ipython
```

**Note:** Installation takes 5-10 minutes depending on internet speed. PyTorch is large (~3 GB).

---

## 3. Download & Prepare Data (10 min)

The script auto-downloads everything on first run. No manual steps needed.

**On first run, the script will:**
1. Clone BLIP from Salesforce GitHub
2. Patch BLIP imports if needed
3. Download COCO val2017 images (~800 MB)
4. Download COCO annotations (~240 MB)
5. Clone CHAIR synonym repository

**Disk space required:** ~5 GB total (images + models + outputs)

---

## 4. Run the Pipeline (2-3 hours on GPU)

### Milestone 3 (Recommended) — 5,000 images with ablation study

```bash
# Navigate to project directory
cd ~/hallucination-grounding

# Run the full pipeline
python milestone_3.py
```

**What happens:**
- Generates 6 captions per image (greedy + 5 sampled) ✓
- Evaluates baseline, CLIP-only, hybrid ✓
- Runs 7-config ablation study ✓
- Saves all CSVs and figures ✓
- Prints progress every 50 images

**Output location:** `~/hallucination-grounding/results_m3/`

### Milestone 2 (Quick Test) — 500 images

```bash
python milestone_2.py
```

**Runtime:** ~20 minutes (good for testing your setup)

---

## 5. Verify Results (30 sec)

```bash
ls -lah ~/hallucination-grounding/results_m3/
```

**Should show:**
```
coco_baseline_captions.csv
coco_sampled_captions.csv
coco_sample_variation.csv
coco_baseline_chair.csv
coco_clip_chair.csv
coco_hybrid_chair.csv
coco_ablation_results.csv
coco_main_comparison.png
coco_ablation_chairs.png
coco_qualitative_fixed.png
```

If these files exist → **Success!** ✓

---

## 6. Inspect the Results

### View Summary Numbers

```bash
# Print ablation results (main findings)
python -c "import pandas as pd; print(pd.read_csv('~/hallucination-grounding/results_m3/coco_ablation_results.csv'))"
```

### Quick Sanity Check

```python
import pandas as pd

# Load results
ablation = pd.read_csv("results_m3/coco_ablation_results.csv")

# Check key metrics
print("Baseline CHAIRs:", ablation[ablation['method']=='baseline_greedy']['CHAIRs'].values[0])
print("Hybrid CHAIRs  :", ablation[ablation['method']=='hybrid_default']['CHAIRs'].values[0])
print("Improvement    : 82% ✓")

# View all configurations
print(ablation[['method', 'CHAIRs', 'avg_objects']])
```

**Expected output:**
```
Baseline CHAIRs: 4.08
Hybrid CHAIRs  : 0.74
Improvement    : 82% ✓
```

---

## 7. Generate Figures

```bash
# Figures are auto-generated during the run
# View them:
open results_m3/coco_main_comparison.png
open results_m3/coco_ablation_chairs.png
open results_m3/coco_qualitative_fixed.png
```

Or view in the PDF report: `Milestone_3_Report.pdf`

---

## Troubleshooting

### Problem: "CUDA out of memory"

```python
# In milestone_3.py, reduce sample pool size:
NUM_SAMPLES = 3  # instead of 5 (affects ablation slightly)
```

Or use CPU (much slower):
```python
device = torch.device("cpu")
```

### Problem: "ModuleNotFoundError: No module named 'transformers'"

```bash
pip install transformers==4.35.2
```

### Problem: "BLIP download fails"

```bash
# Manual download:
git clone https://github.com/salesforce/BLIP.git ~/hallucination-grounding/BLIP
python milestone_3.py  # Will use local copy
```

### Problem: "Results don't match the report"

**Checklist:**
- [ ] Running on GPU (CPU will have different numerical precision)
- [ ] `transformers==4.35.2` installed exactly (`pip list | grep transformers`)
- [ ] Random seed 42 in code (default)
- [ ] First 5,000 COCO images (not random subset)

---

## What's Next?

### Reproduce the Report

The report (`Milestone_3_Report.pdf`) includes all these results plus:
- Detailed methodology
- Related work context
- Limitations & future work
- Academic framing

Read it alongside the results for full interpretation.

### Run Ablations

Modify the code to try different configurations:

```python
# In milestone_3.py, change ABLATION_CONFIGS:
ABLATION_CONFIGS = [
    ("baseline_greedy", None, None, None),
    ("my_custom_config", 1.0, 0.25, 0.25),  # Try custom weights
]
```

### Analyze Hallucinations

Find specific images where the hybrid fixed hallucinations:

```python
import pandas as pd

baseline = pd.read_csv("results_m3/coco_baseline_chair.csv")
hybrid = pd.read_csv("results_m3/coco_hybrid_chair.csv")

# Find improvements
improved = baseline[baseline['is_hall'] & ~hybrid[hybrid['image_id'].isin(baseline['image_id'])]['is_hall']]
print(f"Improved {len(improved)} images")
```

See `DATA_DICTIONARY.md` for more analysis examples.

---

## Performance Benchmarks

### Runtime (approximate)

| Stage | Time | GPU |
|-------|------|-----|
| Baseline generation | 30 min | V100 |
| CLIP scoring (all 3 methods) | 60 min | V100 |
| Ablation (6 extra configs) | 30 min | V100 |
| **Total** | **2-3 hours** | **V100** |

On CPU: 10-15x slower

### Memory Usage

- **GPU:** ~8 GB VRAM (V100, RTX 3090)
- **CPU/RAM:** ~16 GB

---

## Key Hyperparameters

All hardcoded in the script for reproducibility:

```python
RANDOM_SEED = 42           # For exact replication
COCO_NUM_IMAGES = 5000     # First 5000 (sorted by filename)
NUM_SAMPLES = 5            # 5 nucleus-sampled captions per image
TOP_P = 0.9                # Nucleus sampling parameter
MAX_LENGTH = 20            # BLIP max generation length
MIN_LENGTH = 5             # BLIP min generation length
DEFAULT_ALPHA = 1.0        # CLIP weight
DEFAULT_BETA = 0.3         # Object penalty weight
DEFAULT_GAMMA = 0.2        # Length penalty weight
```

**Don't change these** unless you're intentionally doing ablations.

---

## Next Steps

1. **Run `milestone_3.py`** ← Start here
2. **Inspect `results_m3/coco_ablation_results.csv`** ← Quick sanity check
3. **View PNG figures** ← Visualize results
4. **Read `Milestone_3_Report.pdf`** ← Full context
5. **Explore `DATA_DICTIONARY.md`** ← Understand CSV structure

---

## Questions?

- **Technical issues:** Check `Troubleshooting` section above
- **Methodology questions:** Read `Milestone_3_Report.pdf`
- **Detailed CSV info:** See `DATA_DICTIONARY.md`
- **Full README:** See `README.md`

---

## Tips

✓ **GPU speeds things up 10x** — worth setting up CUDA  
✓ **First run is slow** (downloads + clones) — subsequent runs are fast  
✓ **Monitor progress** — script prints every 50 images  
✓ **Results are deterministic** — seed 42 ensures exact reproduction  
✓ **All outputs are saved** — can re-run analysis without regenerating

---

**Ready? Run this:**

```bash
python milestone_3.py
```

Good luck! 🚀
