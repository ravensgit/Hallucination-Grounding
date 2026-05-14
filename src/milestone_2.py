"""
Milestone 2: Grounding Image Captions — Reducing Hallucination in Vision-Language Models
Authors : Naveen Manikandan (manikan2) | Varun Teja Goud Madhagouni (varuntej)
Course  : University at Buffalo
"""

# Cell 1: imports 
import os, sys, json, re, random, zipfile, urllib.request
from collections import defaultdict
 
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')         
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# Cell 2: project paths 
PROJECT_ROOT   = os.path.expanduser('~/hallucination-grounding')
BLIP_DIR       = os.path.join(PROJECT_ROOT, 'BLIP')
CHAIR_DIR      = os.path.join(PROJECT_ROOT, 'chair_official')
DATA_DIR       = os.path.join(PROJECT_ROOT, 'data', 'coco')
IMG_DIR        = os.path.join(DATA_DIR, 'val2017')
ANNOT_DIR      = os.path.join(DATA_DIR, 'annotations')
RESULTS_DIR    = os.path.join(PROJECT_ROOT, 'results')
INSTANCES_FILE = os.path.join(ANNOT_DIR, 'instances_val2017.json')
CAPTIONS_FILE  = os.path.join(ANNOT_DIR, 'captions_val2017.json')

for d in [DATA_DIR, IMG_DIR, ANNOT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print('project root:', PROJECT_ROOT)

# Cell 3: clone BLIP 
if not os.path.exists(BLIP_DIR):
    os.system(f'git clone https://github.com/salesforce/BLIP.git {BLIP_DIR}')
    print('BLIP cloned.')
else:
    print('BLIP already exists.')

sys.path.insert(0, BLIP_DIR)

# Cell 4: fix BLIP import issue (transformers >= 4.35) 
med_path = os.path.join(BLIP_DIR, 'models', 'med.py')
with open(med_path, 'r') as f:
    code = f.read()

if 'from transformers.modeling_utils import apply_chunking_to_forward' in code:
    code = code.replace(
        'from transformers.modeling_utils import apply_chunking_to_forward',
        'from transformers.pytorch_utils import apply_chunking_to_forward'
    )
    with open(med_path, 'w') as f:
        f.write(code)
    print('fixed med.py import.')
else:
    print('no fix needed.')

# BLIP's code imports models relative to its own root so we cd into it
os.chdir(BLIP_DIR)
print('working dir:', os.getcwd())

# Cell 5: download COCO val2017 images 
def get_zip(url, save_to, extract_to):
    print('downloading', url, '...')
    urllib.request.urlretrieve(url, save_to)
    with zipfile.ZipFile(save_to, 'r') as z:
        z.extractall(extract_to)
    os.remove(save_to)
    print('done.')

imgs = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
if len(imgs) < 100:
    print('downloading COCO val2017 images (~800 MB)')
    get_zip(
        'http://images.cocodataset.org/zips/val2017.zip',
        os.path.join(DATA_DIR, 'val2017.zip'),
        DATA_DIR
    )
    print('images:', len(os.listdir(IMG_DIR)))
else:
    print('images already downloaded:', len(imgs))

# Cell 6: download COCO annotations 
if not os.path.exists(INSTANCES_FILE) or not os.path.exists(CAPTIONS_FILE):
    print('downloading COCO annotations (~240 MB)')
    get_zip(
        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        os.path.join(DATA_DIR, 'annotations.zip'),
        DATA_DIR
    )
else:
    print('annotations already exist.')

# Cell 7: clone CHAIR synonym list
if not os.path.exists(CHAIR_DIR):
    os.system(f'git clone https://github.com/LisaAnne/Hallucination.git {CHAIR_DIR}')

syn_file = None
for root, dirs, files in os.walk(CHAIR_DIR):
    for f in files:
        if f == 'synonyms.txt':
            syn_file = os.path.join(root, f)
            break
    if syn_file:
        break

print('synonym file:', syn_file)

# Cell 8: load BLIP model
from models.blip import blip_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device:', device)

model = blip_decoder(
    pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',
    image_size=384,
    vit='base'
)
model.eval()
model = model.to(device)
print('model loaded.')

# Cell 9: BLIP image transform 
transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275,  0.40821073),
        std =(0.26862954, 0.26130258, 0.27577711)
    )
])

# Cell 10: load segmentation ground truth 
instances = json.load(f)

cat_map = {c['id']: c['name'] for c in instances['categories']}

gt_objects = defaultdict(set)
for ann in instances['annotations']:
    gt_objects[ann['image_id']].add(cat_map[ann['category_id']])

fname_to_id = {img['file_name']: img['id'] for img in instances['images']}
COCO_CATS   = set(cat_map.values())

print('categories:', len(COCO_CATS))
print('images with GT:', len(gt_objects))

# Cell 11: load captions file 
with open(CAPTIONS_FILE, 'r') as f:
    captions_data = json.load(f)

print('reference captions:', len(captions_data['annotations']))

# Cell 12: build synonym map 
import inflect
inf = inflect.engine()

def singular(w):
    r = inf.singular_noun(w)
    return r if r else w

syn_map = {}
with open(syn_file, 'r') as f:
    for line in f:
        words = [w.strip().lower() for w in line.strip().split(',')]
        if not words:
            continue
        category = words[0]
        for w in words:
            if w:
                syn_map[w] = category

print('synonym entries:', len(syn_map))
print('man  ->', syn_map.get('man'))
print('bike ->', syn_map.get('bike'))
print('auto ->', syn_map.get('automobile'))

# Cell 13: extract objects from text
def extract_objects(text):
    text  = text.lower()
    text  = re.sub(r'[^a-z0-9 ]', ' ', text)
    words = [singular(w) for w in text.split()]
    found = []
    i = 0
    while i < len(words):
        matched = False
        if i + 1 < len(words):
            bi = words[i] + ' ' + words[i+1]
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

# quick sanity check
sample = captions_data['annotations'][0]['caption']
print('caption:', sample)
print('objects found:', extract_objects(sample))

# Cell 14: build combined ground truth
ref_gt = defaultdict(set)
for ann in captions_data['annotations']:
    ref_gt[ann['image_id']].update(extract_objects(ann['caption']))

combined_gt = {}
for img_id in set(gt_objects.keys()) | set(ref_gt.keys()):
    combined_gt[img_id] = gt_objects.get(img_id, set()) | ref_gt.get(img_id, set())

print('images in combined GT:', len(combined_gt))

for img_id in random.sample(list(combined_gt.keys()), 3):
    print('image:', img_id)
    print('  seg GT    :', gt_objects.get(img_id, set()))
    print('  caption GT:', ref_gt.get(img_id, set()))
    print('  combined  :', combined_gt[img_id])
    print()

# Cell 15: CHAIR metric 
def chair(captions, gt):
    total_caps = len(captions)
    hall_caps  = 0
    total_objs = 0
    hall_objs  = 0
    details    = []

    for c in captions:
        img_gt   = gt.get(c['image_id'], set())
        mentions = extract_objects(c['caption'])
        wrong    = [o for o in mentions if o not in img_gt]

        total_objs += len(mentions)
        hall_objs  += len(wrong)
        is_hall     = len(wrong) > 0
        if is_hall:
            hall_caps += 1

        details.append({
            'image_id'    : c['image_id'],
            'caption'     : c['caption'],
            'gt'          : sorted(img_gt),
            'mentioned'   : mentions,
            'wrong'       : wrong,
            'hallucinated': is_hall
        })

    CHAIRs = round(hall_caps / total_caps * 100, 2) if total_caps > 0 else 0
    CHAIRi = round(hall_objs / total_objs * 100, 2) if total_objs > 0 else 0

    return {
        'CHAIRs'    : CHAIRs,
        'CHAIRi'    : CHAIRi,
        'hall_caps' : hall_caps,
        'total_caps': total_caps,
        'hall_objs' : hall_objs,
        'total_objs': total_objs,
        'details'   : details
    }

# Cell 16: select 500 images
all_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
valid    = [f for f in all_imgs
            if f in fname_to_id and fname_to_id[f] in combined_gt]
selected = valid[:500]

print('total images :', len(all_imgs))
print('with GT      :', len(valid))
print('selected     :', len(selected))

# Cell 17: generate baseline captions
baseline = []

for i, fname in enumerate(selected):
    img    = Image.open(os.path.join(IMG_DIR, fname)).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(
            tensor,
            sample=False,
            num_beams=1,
            max_length=20,
            min_length=5
        )

    baseline.append({'image_id': fname_to_id[fname], 'caption': out[0]})

    if (i + 1) % 50 == 0:
        print(f'{i+1}/500 -- {out[0]}')

print('done:', len(baseline), 'captions generated.')

# Cell 18: run CHAIR on baseline 
result = chair(baseline, combined_gt)

print('baseline CHAIR results:')
print('CHAIRs:', result['CHAIRs'], '%')
print('CHAIRi:', result['CHAIRi'], '%')
print('hallucinated captions:', result['hall_caps'], '/', result['total_caps'])
print('hallucinated objects :', result['hall_objs'], '/', result['total_objs'])

# Cell 19: save baseline results
rows = [{
    'image_id'    : d['image_id'],
    'caption'     : d['caption'],
    'gt_objects'  : '|'.join(sorted(d['gt'])),
    'hallucinated': '|'.join(d['wrong']),
    'is_hall'     : d['hallucinated']
} for d in result['details']]

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, 'baseline_chair.csv'), index=False)
print('saved:', len(df), 'rows.')
print(df[df['is_hall'] == True][['caption', 'hallucinated']].head(3))

# Cell 20: generate stochastic samples 
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

sampled = []

for i, fname in enumerate(selected):
    img    = Image.open(os.path.join(IMG_DIR, fname)).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    caps = []
    with torch.no_grad():
        for _ in range(5):
            out = model.generate(
                tensor,
                sample=True,
                top_p=0.9,
                max_length=20,
                min_length=5
            )
            caps.append(out[0])

    sampled.append({
        'image_id': fname_to_id[fname],
        'fname'   : fname,
        'captions': caps
    })

    if (i + 1) % 50 == 0:
        print(f'{i+1}/500')
        for c in caps:
            print(' ', c)

print('done.', len(sampled), 'images.')

# Cell 21: save sampled captions
rows = []
for s in sampled:
    row = {'image_id': s['image_id'], 'fname': s['fname']}
    for i, cap in enumerate(s['captions']):
        row[f'sample_{i+1}'] = cap
    rows.append(row)

df_s = pd.DataFrame(rows)
df_s.to_csv(os.path.join(RESULTS_DIR, 'sampled_captions.csv'), index=False)
print('saved:', len(df_s), 'rows')
print()
print('first image — all 5 samples:')
for i in range(1, 6):
    print(f'  sample {i}:', df_s[f'sample_{i}'].iloc[0])

# Cell 22: variation analysis
variation = []
for pos in range(5):
    caps = [{'image_id': s['image_id'], 'caption': s['captions'][pos]}
            for s in sampled]
    r = chair(caps, combined_gt)
    variation.append(r)
    print(f'sample {pos+1}: CHAIRs={r["CHAIRs"]}%  CHAIRi={r["CHAIRi"]}%')

chairs_vals = [r['CHAIRs'] for r in variation]
print()
print('variance   :', round(float(np.var(chairs_vals)), 4))
print('min CHAIRs :', min(chairs_vals), '%')
print('max CHAIRs :', max(chairs_vals), '%')
print('baseline   :', result['CHAIRs'], '%')

# Cell 23: load CLIP 
from transformers import CLIPModel, CLIPProcessor

clip_model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip_model     = clip_model.to(device)
clip_model.eval()
print('CLIP loaded.')

# Cell 24: hybrid scoring functions 
def clip_score(img, caption):
    inputs = clip_processor(
        text=[caption], images=img,
        return_tensors='pt', padding=True
    ).to(device)
    with torch.no_grad():
        img_feat = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
        txt_feat = clip_model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        cos = (img_feat * txt_feat).sum().item()
    return (cos + 1) / 2

def obj_penalty(caption):
    return len(set(extract_objects(caption))) / 10.0

def len_penalty(caption):
    return max(0, len(caption.split()) - 10) / 20.0

def hybrid_score(img, caption, a=1.0, b=0.3, g=0.2):
    return (a * clip_score(img, caption)
            - b * obj_penalty(caption)
            - g * len_penalty(caption))

# Cell 25: scoring breakdown for one image 
idx  = 0
s    = sampled[idx]
img  = Image.open(os.path.join(IMG_DIR, s['fname'])).convert('RGB')
pool = [baseline[idx]['caption']] + s['captions']

print(f'image: {s["fname"]}')
print(f'GT objects: {combined_gt[s["image_id"]]}')
print()

for j, cap in enumerate(pool):
    tag = 'baseline' if j == 0 else f'sample {j}'
    print(f'{tag}:')
    print(f'  caption : {cap}')
    print(f'  CLIP    : {round(clip_score(img, cap), 4)}')
    print(f'  obj pen : {round(obj_penalty(cap), 4)}')
    print(f'  len pen : {round(len_penalty(cap), 4)}')
    print(f'  hybrid  : {round(hybrid_score(img, cap), 4)}')
    print()

# Cell 26: apply selection 
clip_selected   = []
hybrid_selected = []

for i, s in enumerate(sampled):
    img  = Image.open(os.path.join(IMG_DIR, s['fname'])).convert('RGB')
    pool = [baseline[i]['caption']] + s['captions']

    best_clip, bc_score = None, -999
    for cap in pool:
        sc = clip_score(img, cap)
        if sc > bc_score:
            bc_score = sc;  best_clip = cap

    best_hybrid, bh_score = None, -999
    for cap in pool:
        sc = hybrid_score(img, cap)
        if sc > bh_score:
            bh_score = sc;  best_hybrid = cap

    clip_selected.append({'image_id': s['image_id'], 'caption': best_clip})
    hybrid_selected.append({'image_id': s['image_id'], 'caption': best_hybrid})

    if (i + 1) % 50 == 0:
        print(f'{i+1}/500')
        print(f'  baseline : {baseline[i]["caption"]}')
        print(f'  CLIP     : {best_clip}')
        print(f'  hybrid   : {best_hybrid}')

# Cell 27: run CHAIR on all three methods
clip_result   = chair(clip_selected,   combined_gt)
hybrid_result = chair(hybrid_selected, combined_gt)

print('results on 500 COCO images:')
print()
print(f'Baseline  CHAIRs={result["CHAIRs"]}%  CHAIRi={result["CHAIRi"]}%')
print(f'CLIP only CHAIRs={clip_result["CHAIRs"]}%  CHAIRi={clip_result["CHAIRi"]}%')
print(f'Hybrid    CHAIRs={hybrid_result["CHAIRs"]}%  CHAIRi={hybrid_result["CHAIRi"]}%')

# Cell 28: diagnostic stats 
def avg_stats(caption_list):
    lengths = [len(c['caption'].split()) for c in caption_list]
    objects = [len(extract_objects(c['caption'])) for c in caption_list]
    return {
        'avg_length' : round(sum(lengths) / len(lengths), 2),
        'avg_objects': round(sum(objects) / len(objects), 2)
    }

b_stats = avg_stats(baseline)
c_stats = avg_stats(clip_selected)
h_stats = avg_stats(hybrid_selected)

print('Method       CHAIRs  CHAIRi  AvgLen  AvgObj')
print('-' * 50)
print(f'Baseline     {result["CHAIRs"]}%   {result["CHAIRi"]}%   {b_stats["avg_length"]}   {b_stats["avg_objects"]}')
print(f'CLIP only    {clip_result["CHAIRs"]}%   {clip_result["CHAIRi"]}%   {c_stats["avg_length"]}   {c_stats["avg_objects"]}')
print(f'Hybrid       {hybrid_result["CHAIRs"]}%   {hybrid_result["CHAIRi"]}%   {h_stats["avg_length"]}   {h_stats["avg_objects"]}')

# Cell 29: save CLIP and hybrid results
def save_result(res, fname):
    rows = [{
        'image_id': d['image_id'],
        'caption' : d['caption'],
        'gt'      : '|'.join(sorted(d['gt'])),
        'wrong'   : '|'.join(d['wrong']),
        'is_hall' : d['hallucinated']
    } for d in res['details']]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
    print(fname, '—', df['is_hall'].sum(), 'hallucinated out of', len(df))

save_result(clip_result,   'clip_chair.csv')
save_result(hybrid_result, 'hybrid_chair.csv')

# Cell 30: summary CSV
summary = pd.DataFrame([
    {'method': 'baseline',
     'CHAIRs': result['CHAIRs'],       'CHAIRi': result['CHAIRi'],
     'hall_captions': result['hall_caps'],       'total_captions': result['total_caps']},
    {'method': 'clip_only',
     'CHAIRs': clip_result['CHAIRs'],   'CHAIRi': clip_result['CHAIRi'],
     'hall_captions': clip_result['hall_caps'],   'total_captions': clip_result['total_caps']},
    {'method': 'hybrid',
     'CHAIRs': hybrid_result['CHAIRs'], 'CHAIRi': hybrid_result['CHAIRi'],
     'hall_captions': hybrid_result['hall_caps'], 'total_captions': hybrid_result['total_caps']},
])
summary.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)
print('saved: summary.csv')
print()
print(summary.to_string(index=False))

# Cell 31: comparison bar plot 
methods = ['Baseline', 'CLIP only', 'Hybrid']
chairs  = [result['CHAIRs'], clip_result['CHAIRs'], hybrid_result['CHAIRs']]
chairi  = [result['CHAIRi'], clip_result['CHAIRi'], hybrid_result['CHAIRi']]
colors  = ['gray', 'steelblue', 'green']

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, vals, title in [
    (axes[0], chairs, 'CHAIRs (sentence level)'),
    (axes[1], chairi, 'CHAIRi (instance level)')
]:
    bars = ax.bar(methods, vals, color=colors)
    ax.set_title(title)
    ax.set_ylabel('%')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{v}%', ha='center', va='bottom')

fig.suptitle('Baseline vs CLIP-only vs Hybrid — COCO val2017 500 images')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'plot1_comparison.png'), dpi=150)
plt.close()
print('saved plot1_comparison.png')

# Cell 32: example images where hybrid fixed hallucination
baseline_hall = {d['image_id']: d for d in result['details']        if d['hallucinated']}
hybrid_clean  = {d['image_id']: d for d in hybrid_result['details'] if not d['hallucinated']}
improved      = [img_id for img_id in baseline_hall if img_id in hybrid_clean]
print('images improved by hybrid:', len(improved))

examples    = improved[:3]
id_to_fname = {v: k for k, v in fname_to_id.items()}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Examples where hybrid fixed baseline hallucination', fontsize=11)

for ax, img_id in zip(axes, examples):
    fname = id_to_fname[img_id]
    img   = Image.open(os.path.join(IMG_DIR, fname)).convert('RGB')
    b_cap = baseline_hall[img_id]['caption']
    h_cap = hybrid_clean[img_id]['caption']
    wrong = baseline_hall[img_id]['wrong']

    ax.imshow(img)
    ax.set_title(
        f'baseline: {b_cap}\nhallucinated: {wrong}\n\nhybrid: {h_cap}',
        fontsize=7
    )
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'plot5_examples.png'), dpi=150)
plt.close()
print('saved plot5_examples.png')

print()
print('all done. results in:', RESULTS_DIR)
