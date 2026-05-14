"""
Milestone 3: Grounding Image Captions — Reducing Hallucination in Vision-Language Models
Authors : Naveen Manikandan (manikan2) | Varun Teja Goud Madhagouni (varuntej)
"""

# imports
import os, sys, json, re, random, zipfile, urllib.request
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# configuration and paths
PROJECT_ROOT   = os.path.expanduser("~/hallucination-grounding")
BLIP_DIR       = os.path.join(PROJECT_ROOT, "BLIP")
CHAIR_DIR      = os.path.join(PROJECT_ROOT, "chair_official")

COCO_DIR       = os.path.join(PROJECT_ROOT, "data", "coco")
COCO_IMG_DIR   = os.path.join(COCO_DIR, "val2017")
COCO_ANNOT_DIR = os.path.join(COCO_DIR, "annotations")
INSTANCES_FILE = os.path.join(COCO_ANNOT_DIR, "instances_val2017.json")
CAPTIONS_FILE  = os.path.join(COCO_ANNOT_DIR, "captions_val2017.json")

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results_m3")

RANDOM_SEED        = 42
COCO_NUM_IMAGES    = 5000      
NUM_SAMPLES        = 5          
TOP_P              = 0.9
MAX_LENGTH         = 20
MIN_LENGTH         = 5
USE_RANDOM_COCO_SUBSET = False  

DEFAULT_ALPHA = 1.0
DEFAULT_BETA  = 0.3
DEFAULT_GAMMA = 0.2

for d in [COCO_DIR, COCO_IMG_DIR, COCO_ANNOT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("project root:", PROJECT_ROOT)
print("results dir :", RESULTS_DIR)

# helper: download zip
def get_zip(url, save_to, extract_to):
    print("downloading", url)
    urllib.request.urlretrieve(url, save_to)
    with zipfile.ZipFile(save_to, "r") as z:
        z.extractall(extract_to)
    os.remove(save_to)
    print("done:", url)

# clone and patch BLIP
if not os.path.exists(BLIP_DIR):
    os.system(f"git clone https://github.com/salesforce/BLIP.git {BLIP_DIR}")
    print("BLIP cloned.")
else:
    print("BLIP already exists.")

sys.path.insert(0, BLIP_DIR)

med_path = os.path.join(BLIP_DIR, "models", "med.py")
with open(med_path, "r") as f:
    code = f.read()

if "from transformers.modeling_utils import apply_chunking_to_forward" in code:
    code = code.replace(
        "from transformers.modeling_utils import apply_chunking_to_forward",
        "from transformers.pytorch_utils import apply_chunking_to_forward"
    )
    with open(med_path, "w") as f:
        f.write(code)
    print("fixed med.py import.")
else:
    print("no med.py fix needed.")

os.chdir(BLIP_DIR)
print("working dir:", os.getcwd())

# download COCO val2017 and annotations
imgs = [f for f in os.listdir(COCO_IMG_DIR) if f.endswith(".jpg")]
if len(imgs) < 100:
    print("downloading COCO val2017 images (~800 MB)")
    get_zip(
        "http://images.cocodataset.org/zips/val2017.zip",
        os.path.join(COCO_DIR, "val2017.zip"),
        COCO_DIR
    )
else:
    print("COCO images already downloaded:", len(imgs))

if not os.path.exists(INSTANCES_FILE) or not os.path.exists(CAPTIONS_FILE):
    print("downloading COCO annotations (~240 MB)")
    get_zip(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        os.path.join(COCO_DIR, "annotations.zip"),
        COCO_DIR
    )
else:
    print("COCO annotations already exist.")

# clone CHAIR synonym list
if not os.path.exists(CHAIR_DIR):
    os.system(f"git clone https://github.com/LisaAnne/Hallucination.git {CHAIR_DIR}")

syn_file = None
for root, dirs, files in os.walk(CHAIR_DIR):
    for f in files:
        if f == "synonyms.txt":
            syn_file = os.path.join(root, f)
            break
    if syn_file:
        break

if syn_file is None:
    raise FileNotFoundError("Could not find synonyms.txt from CHAIR repository.")

print("synonym file:", syn_file)

# load BLIP model
from models.blip import blip_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

model = blip_decoder(
    pretrained="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth",
    image_size=384,
    vit="base"
)
model.eval()
model = model.to(device)
print("BLIP model loaded.")

# image transform
transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275,  0.40821073),
        std =(0.26862954, 0.26130258, 0.27577711)
    )
])

# load COCO annotations and synonyms
with open(INSTANCES_FILE, "r") as f:
    instances = json.load(f)

with open(CAPTIONS_FILE, "r") as f:
    captions_data = json.load(f)

cat_map = {c["id"]: c["name"] for c in instances["categories"]}
COCO_CATS = set(cat_map.values())

gt_objects = defaultdict(set)
for ann in instances["annotations"]:
    gt_objects[ann["image_id"]].add(cat_map[ann["category_id"]])

fname_to_id = {img["file_name"]: img["id"] for img in instances["images"]}
id_to_fname = {v: k for k, v in fname_to_id.items()}

import inflect
inf = inflect.engine()

def singular(w):
    r = inf.singular_noun(w)
    return r if r else w

syn_map = {}
with open(syn_file, "r") as f:
    for line in f:
        words = [w.strip().lower() for w in line.strip().split(",")]
        if not words:
            continue
        category = words[0]
        for w in words:
            if w:
                syn_map[w] = category

print("COCO categories:", len(COCO_CATS))
print("images with seg GT:", len(gt_objects))
print("reference captions:", len(captions_data["annotations"]))
print("synonym entries:", len(syn_map))
print("sanity: man->", syn_map.get("man"), " bike->", syn_map.get("bike"), " fridge->", syn_map.get("fridge"))

# object extraction and CHAIR-style GT
def extract_objects(text):
    """
    CHAIR-style object mention extractor:
    - lowercases and cleans caption
    - singularizes words
    - checks two-word compounds first
    - maps synonyms to COCO 80 categories
    """
    text  = text.lower()
    text  = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [singular(w) for w in text.split()]
    found = []
    i = 0
    while i < len(words):
        matched = False
        if i + 1 < len(words):
            bi = words[i] + " " + words[i+1]
            if bi in syn_map and syn_map[bi] in COCO_CATS:
                found.append(syn_map[bi])
                i += 2
                matched = True
        if not matched:
            w = words[i]
            if w in syn_map and syn_map[w] in COCO_CATS:
                found.append(syn_map[w])
            i += 1
    return found

ref_gt = defaultdict(set)
for ann in captions_data["annotations"]:
    ref_gt[ann["image_id"]].update(extract_objects(ann["caption"]))

combined_gt = {}
for img_id in set(gt_objects.keys()) | set(ref_gt.keys()):
    combined_gt[img_id] = gt_objects.get(img_id, set()) | ref_gt.get(img_id, set())

print("images in combined GT:", len(combined_gt))

# CHAIR-style evaluation
def chair(captions, gt):
    total_caps = len(captions)
    hall_caps  = 0
    total_objs = 0
    hall_objs  = 0
    details    = []

    for c in captions:
        img_gt   = gt.get(c["image_id"], set())
        mentions = extract_objects(c["caption"])
        wrong    = [o for o in mentions if o not in img_gt]

        total_objs += len(mentions)
        hall_objs  += len(wrong)
        is_hall = len(wrong) > 0
        if is_hall:
            hall_caps += 1

        details.append({
            "image_id": c["image_id"],
            "fname": c.get("fname", id_to_fname.get(c["image_id"], "")),
            "caption": c["caption"],
            "gt": sorted(img_gt),
            "mentioned": mentions,
            "wrong": wrong,
            "hallucinated": is_hall
        })

    return {
        "CHAIRs": round(hall_caps / total_caps * 100, 2) if total_caps else 0,
        "CHAIRi": round(hall_objs / total_objs * 100, 2) if total_objs else 0,
        "hall_caps": hall_caps,
        "total_caps": total_caps,
        "hall_objs": hall_objs,
        "total_objs": total_objs,
        "details": details
    }

def avg_stats(caption_list):
    lengths = [len(c["caption"].split()) for c in caption_list]
    objects = [len(extract_objects(c["caption"])) for c in caption_list]
    return {
        "avg_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "avg_objects": round(sum(objects) / len(objects), 2) if objects else 0
    }

# image caption generation helpers
def blip_caption_for_image(img_path, sample=False, top_p=TOP_P):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        if sample:
            out = model.generate(
                tensor,
                sample=True,
                top_p=top_p,
                num_beams=1,
                max_length=MAX_LENGTH,
                min_length=MIN_LENGTH
            )
        else:
            out = model.generate(
                tensor,
                sample=False,
                num_beams=1,
                max_length=MAX_LENGTH,
                min_length=MIN_LENGTH
            )
    return out[0]

def generate_baseline_and_samples(image_records, img_dir, tag):
    """
    image_records: list of dicts with keys fname and image_id.
    returns:
        baseline: one greedy caption per image
        sampled : NUM_SAMPLES nucleus-sampled captions per image
    """
    baseline = []
    sampled = []

    for i, rec in enumerate(image_records):
        fname = rec["fname"]
        img_path = os.path.join(img_dir, fname)

        base_cap = blip_caption_for_image(img_path, sample=False)
        baseline.append({"image_id": rec["image_id"], "fname": fname, "caption": base_cap})

        caps = []
        for _ in range(NUM_SAMPLES):
            caps.append(blip_caption_for_image(img_path, sample=True, top_p=TOP_P))

        sampled.append({"image_id": rec["image_id"], "fname": fname, "captions": caps})

        if (i + 1) % 50 == 0 or (i + 1) == len(image_records):
            print(f"{tag}: {i+1}/{len(image_records)}")
            print("  baseline:", base_cap)
            print("  sample1 :", caps[0])

    return baseline, sampled

# select COCO images
all_imgs = sorted([f for f in os.listdir(COCO_IMG_DIR) if f.endswith(".jpg")])
valid = [f for f in all_imgs if f in fname_to_id and fname_to_id[f] in combined_gt]

if USE_RANDOM_COCO_SUBSET:
    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(valid, min(COCO_NUM_IMAGES, len(valid)))
    selected = sorted(selected)
else:
    selected = valid[:min(COCO_NUM_IMAGES, len(valid))]

coco_records = [{"fname": f, "image_id": fname_to_id[f]} for f in selected]

print("COCO total images:", len(all_imgs))
print("COCO valid images:", len(valid))
print("COCO selected:", len(coco_records))

# generate COCO baseline + samples
coco_baseline, coco_sampled = generate_baseline_and_samples(
    coco_records, COCO_IMG_DIR, tag="COCO"
)

pd.DataFrame(coco_baseline).to_csv(os.path.join(RESULTS_DIR, "coco_baseline_captions.csv"), index=False)

sample_rows = []
for s in coco_sampled:
    row = {"image_id": s["image_id"], "fname": s["fname"]}
    for i, cap in enumerate(s["captions"]):
        row[f"sample_{i+1}"] = cap
    sample_rows.append(row)
pd.DataFrame(sample_rows).to_csv(os.path.join(RESULTS_DIR, "coco_sampled_captions.csv"), index=False)

# COCO baseline CHAIR + sample variation
coco_baseline_result = chair(coco_baseline, combined_gt)

print("\nCOCO baseline CHAIR-style:")
print(coco_baseline_result)

variation_rows = []
for pos in range(NUM_SAMPLES):
    caps = [{"image_id": s["image_id"], "fname": s["fname"], "caption": s["captions"][pos]} for s in coco_sampled]
    r = chair(caps, combined_gt)
    variation_rows.append({
        "sample_position": pos + 1,
        "CHAIRs": r["CHAIRs"],
        "CHAIRi": r["CHAIRi"],
        "hall_caps": r["hall_caps"],
        "hall_objs": r["hall_objs"]
    })
    print(f"sample {pos+1}: CHAIRs={r['CHAIRs']}% CHAIRi={r['CHAIRi']}%")

variation_df = pd.DataFrame(variation_rows)
variation_df.to_csv(os.path.join(RESULTS_DIR, "coco_sample_variation.csv"), index=False)

chairs_vals = variation_df["CHAIRs"].tolist()
print("sample CHAIRs variance:", round(float(np.var(chairs_vals)), 4))

# load CLIP and scoring functions
from transformers import CLIPModel, CLIPProcessor

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("CLIP loaded.")

def clip_score(img, caption):
    inputs = clip_processor(
        text=[caption],
        images=img,
        return_tensors="pt",
        padding=True
    ).to(device)
    with torch.no_grad():
        img_feat = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_feat = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        cos = (img_feat * txt_feat).sum().item()
    return (cos + 1) / 2

def obj_penalty(caption):
    return len(set(extract_objects(caption))) / 10.0

def len_penalty(caption):
    return max(0, len(caption.split()) - 10) / 20.0

def hybrid_score(img, caption, a=DEFAULT_ALPHA, b=DEFAULT_BETA, g=DEFAULT_GAMMA):
    return (a * clip_score(img, caption)
            - b * obj_penalty(caption)
            - g * len_penalty(caption))

def select_captions(records, img_dir, baseline, sampled, method="clip", a=1.0, b=0.3, g=0.2):
    selected = []

    for i, s in enumerate(sampled):
        img = Image.open(os.path.join(img_dir, s["fname"])).convert("RGB")
        pool = [baseline[i]["caption"]] + s["captions"]

        best_cap, best_score = None, -1e9
        for cap in pool:
            if method == "clip":
                sc = clip_score(img, cap)
            elif method == "hybrid":
                sc = hybrid_score(img, cap, a=a, b=b, g=g)
            else:
                raise ValueError(f"unknown method: {method}")

            if sc > best_score:
                best_score = sc
                best_cap = cap

        selected.append({
            "image_id": s["image_id"],
            "fname": s["fname"],
            "caption": best_cap,
            "selection_score": best_score
        })

    return selected

# COCO CLIP-only and default hybrid
coco_clip_selected = select_captions(
    coco_records, COCO_IMG_DIR, coco_baseline, coco_sampled, method="clip"
)

coco_hybrid_selected = select_captions(
    coco_records, COCO_IMG_DIR, coco_baseline, coco_sampled,
    method="hybrid", a=DEFAULT_ALPHA, b=DEFAULT_BETA, g=DEFAULT_GAMMA
)

coco_clip_result = chair(coco_clip_selected, combined_gt)
coco_hybrid_result = chair(coco_hybrid_selected, combined_gt)

print("\nCOCO main results:")
print(f"Baseline  CHAIRs={coco_baseline_result['CHAIRs']} CHAIRi={coco_baseline_result['CHAIRi']}")
print(f"CLIP only CHAIRs={coco_clip_result['CHAIRs']} CHAIRi={coco_clip_result['CHAIRi']}")
print(f"Hybrid    CHAIRs={coco_hybrid_result['CHAIRs']} CHAIRi={coco_hybrid_result['CHAIRi']}")

# COCO ablation study
ABLATION_CONFIGS = [
    # name,                  alpha,  beta,  gamma
    ("baseline_greedy",      None,   None,  None),
    ("clip_only",            1.0,    0.0,   0.0),
    ("hybrid_default",       1.0,    0.3,   0.2),
    ("object_penalty_only",  1.0,    0.3,   0.0),
    ("length_penalty_only",  1.0,    0.0,   0.2),
    ("weak_penalty",         1.0,    0.15,  0.1),
    ("strong_penalty",       1.0,    0.5,   0.3),
]

ablation_rows = []
ablation_outputs = {}

for name, a, b, g in ABLATION_CONFIGS:
    if name == "baseline_greedy":
        captions = coco_baseline
        res = coco_baseline_result
    elif name == "clip_only":
        captions = coco_clip_selected
        res = coco_clip_result
    else:
        captions = select_captions(
            coco_records, COCO_IMG_DIR, coco_baseline, coco_sampled,
            method="hybrid", a=a, b=b, g=g
        )
        res = chair(captions, combined_gt)

    stats = avg_stats(captions)
    ablation_outputs[name] = captions

    ablation_rows.append({
        "method": name,
        "alpha": a,
        "beta": b,
        "gamma": g,
        "CHAIRs": res["CHAIRs"],
        "CHAIRi": res["CHAIRi"],
        "hall_caps": res["hall_caps"],
        "hall_objs": res["hall_objs"],
        "avg_length": stats["avg_length"],
        "avg_objects": stats["avg_objects"]
    })

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(os.path.join(RESULTS_DIR, "coco_ablation_results.csv"), index=False)
print("\nCOCO ablation:")
print(ablation_df.to_string(index=False))

# save COCO outputs and plots
def save_chair_details(res, fname):
    rows = []
    for d in res["details"]:
        rows.append({
            "image_id": d["image_id"],
            "fname": d["fname"],
            "caption": d["caption"],
            "gt": "|".join(d["gt"]),
            "mentioned": "|".join(d["mentioned"]),
            "wrong": "|".join(d["wrong"]),
            "is_hall": d["hallucinated"]
        })
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, fname), index=False)

save_chair_details(coco_baseline_result, "coco_baseline_chair.csv")
save_chair_details(coco_clip_result, "coco_clip_chair.csv")
save_chair_details(coco_hybrid_result, "coco_hybrid_chair.csv")

# Plot main comparison
main_methods = ["Baseline", "CLIP only", "Hybrid"]
main_chairs = [coco_baseline_result["CHAIRs"], coco_clip_result["CHAIRs"], coco_hybrid_result["CHAIRs"]]
main_chairi = [coco_baseline_result["CHAIRi"], coco_clip_result["CHAIRi"], coco_hybrid_result["CHAIRi"]]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, vals, title in [
    (axes[0], main_chairs, "CHAIRs (sentence level)"),
    (axes[1], main_chairi, "CHAIRi (instance level)")
]:
    bars = ax.bar(main_methods, vals)
    ax.set_ylabel("%")
    ax.set_title(title)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v}%", ha="center", va="bottom")
fig.suptitle(f"COCO val2017 — {len(coco_records)} images")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "coco_main_comparison.png"), dpi=160)
plt.close()

# Plot ablation
plt.figure(figsize=(10, 4))
plt.bar(ablation_df["method"], ablation_df["CHAIRs"])
plt.xticks(rotation=35, ha="right")
plt.ylabel("CHAIRs (%)")
plt.title("COCO Ablation Study: Effect of Hybrid Penalties")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "coco_ablation_chairs.png"), dpi=160)
plt.close()

# Qualitative COCO examples where hybrid fixes baseline
baseline_hall = {d["image_id"]: d for d in coco_baseline_result["details"] if d["hallucinated"]}
hybrid_clean = {d["image_id"]: d for d in coco_hybrid_result["details"] if not d["hallucinated"]}
improved = [img_id for img_id in baseline_hall if img_id in hybrid_clean]
print("COCO images improved by hybrid:", len(improved))

examples = improved[:3]
if examples:
    fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 5))
    if len(examples) == 1:
        axes = [axes]
    fig.suptitle("COCO examples where hybrid fixed baseline hallucination", fontsize=11)
    for ax, img_id in zip(axes, examples):
        fname = id_to_fname[img_id]
        img = Image.open(os.path.join(COCO_IMG_DIR, fname)).convert("RGB")
        b_cap = baseline_hall[img_id]["caption"]
        h_cap = hybrid_clean[img_id]["caption"]
        wrong = baseline_hall[img_id]["wrong"]
        ax.imshow(img)
        ax.set_title(f"baseline: {b_cap}\nhallucinated: {wrong}\n\nhybrid: {h_cap}", fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "coco_qualitative_fixed.png"), dpi=160)
    plt.close()


# final summary
print("\n" + "="*60)
print("All done. Outputs saved to:", RESULTS_DIR)
print("="*60)
print("\nCOCO files:")
print("  coco_baseline_captions.csv")
print("  coco_sampled_captions.csv")
print("  coco_sample_variation.csv")
print("  coco_baseline_chair.csv")
print("  coco_clip_chair.csv")
print("  coco_hybrid_chair.csv")
print("  coco_ablation_results.csv")
print("  coco_main_comparison.png")
print("  coco_ablation_chairs.png")
print("  coco_qualitative_fixed.png")
