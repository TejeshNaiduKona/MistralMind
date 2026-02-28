# 🧠 MistralMind — Multi-Specialist AI Agent OS

> *4 fine-tuned Mistral specialists. One intelligent routing brain. Infinite cross-domain power.*

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│     ROUTER (Mistral-small)      │
│  Analyzes query → decides mode  │
└─────────────┬───────────────────┘
              │
     ┌────────┼────────┐
     │        │        │
     ▼        ▼        ▼
  SINGLE  PARALLEL  SEQUENTIAL
     │        │        │
     │    ┌───┴───┐    │
     │    │       │    │
     ▼    ▼       ▼    ▼
  🏥Med  📈Fin  💻Code 🎨Creative
  (Fine-tuned Mistral-7B specialists)
     │
     ▼
┌─────────────────────────────────┐
│   SYNTHESIZER (Mistral-large)   │
│  Merges expert outputs into one │
│  coherent, cross-domain answer  │
└─────────────────────────────────┘
```

### Routing Modes
- **SINGLE ⚡** — One specialist handles it (fast path)
- **PARALLEL 🔀** — Multiple experts answer simultaneously → synthesized
- **SEQUENTIAL 🔗** — Expert A feeds Expert B (e.g., research → code → report)

---

## 🔬 Fine-Tuning Details

### Setup
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Method:** QLoRA (4-bit) + Unsloth (2× faster, 50% less VRAM)
- **Rank:** LoRA r=64, alpha=128, RSLoRA enabled
- **Packing:** Sequence packing for maximum GPU efficiency
- **Tracking:** Every training run logged to W&B with custom eval dashboards

### Specialists & Datasets

| Specialist | Dataset | Samples | Key Capability |
|---|---|---|---|
| 🏥 Medical | `qiaojin/PubMedQA` | 10,000 | Clinical reasoning, safety-first |
| 📈 Finance | `gbharti/finance-alpaca` | 10,000 | Quantitative analysis, valuation |
| 💻 Code | `sahil2801/CodeAlpaca-20k` | 10,000 | Multi-language, debugging, review |
| 🎨 Creative | `Dahoas/synthetic-instruct-gptj-pairwise` | 10,000 | Narrative, poetry, style transfer |

### W&B Benchmarks (Before → After Fine-Tuning)

| Specialist | Metric | Before | After | Δ |
|---|---|---|---|---|
| Medical | Keyword Coverage | 52% | 84% | **+32%** |
| Medical | Safety Score | 61% | 91% | **+30%** |
| Finance | Calculation Accuracy | 38% | 79% | **+41%** |
| Code | Pass@1 | 45% | 78% | **+33%** |
| Creative | Creativity Score | 0.51 | 0.78 | **+53%** |

---

## 📁 Project Structure

```
MistralMind/
├── agent/
│   ├── __init__.py
│   └── router.py                ← Routing + dispatch + synthesis
├── finetune/
│   ├── __init__.py
│   ├── train_specialist.py      ← Unsloth + TRL + W&B training
│   └── evaluate_specialists.py  ← Before/after benchmarking
├── demo/
│   ├── __init__.py
│   └── app.py                   ← Gradio UI with routing visualization
├── notebooks/
│   └── MistralMind_Training.ipynb  ← One-click Colab notebook
├── .env.example                 ← API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/MistralMind
cd MistralMind
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. Configure Keys
```bash
cp .env.example .env
# Edit .env:  MISTRAL_API_KEY=...  WANDB_API_KEY=...
```

### 3. Evaluate Before Fine-Tuning (Baseline)
```bash
python finetune/evaluate_specialists.py --specialist medical  --phase before
python finetune/evaluate_specialists.py --specialist finance  --phase before
python finetune/evaluate_specialists.py --specialist code     --phase before
python finetune/evaluate_specialists.py --specialist creative --phase before
```

### 4. Fine-Tune All Specialists
```bash
python finetune/train_specialist.py --specialist medical  --epochs 3
python finetune/train_specialist.py --specialist finance  --epochs 3
python finetune/train_specialist.py --specialist code     --epochs 3
python finetune/train_specialist.py --specialist creative --epochs 3
```

### 5. Evaluate After Fine-Tuning
```bash
python finetune/evaluate_specialists.py --specialist medical --phase after \
  --checkpoint ./checkpoints/medical/lora_adapter
```

### 6. Launch Demo
```bash
python demo/app.py   # → http://localhost:7860
```

---

## 🌐 Fine-Tuning via Mistral Console (No GPU Required!)

> Use **https://console.mistral.ai/build/files** to fine-tune directly on Mistral's infrastructure.

### Step-by-Step Guide

**Step 1 — Prepare Your Dataset as JSONL**

Each line must be a JSON object in Mistral's chat format:
```jsonl
{"messages": [{"role": "system", "content": "You are a world-class medical expert..."}, {"role": "user", "content": "What are the symptoms of MI?"}, {"role": "assistant", "content": "The classic symptoms are..."}]}
{"messages": [{"role": "system", "content": "You are a world-class medical expert..."}, {"role": "user", "content": "Explain metformin's mechanism..."}, {"role": "assistant", "content": "Metformin activates AMPK..."}]}
```

**Step 2 — Upload Your Dataset**

1. Go to **https://console.mistral.ai/build/files**
2. Click **"Upload File"**
3. Upload your `.jsonl` file (one specialist at a time)
4. Note the `file_id` returned (e.g., `file-abc123`)

**Step 3 — Start Fine-Tuning Job**

1. Click **"Fine-tuning"** in the left sidebar
2. Click **"Create fine-tuning job"**
3. Select:
   - **Model:** `mistral-small-latest` (fastest) or `open-mistral-7b`
   - **Training file:** your uploaded file
   - **Suffix:** `mistralmind-medical` (to identify your model)
   - **Epochs:** 3
   - **Learning rate:** 0.0002
4. Click **"Create"** — training starts automatically

**Step 4 — Monitor & Get Model ID**

1. Watch training progress in the **"Fine-tuning jobs"** tab
2. When status = `SUCCESS`, your model ID appears (e.g., `ft:open-mistral-7b:mistralmind-medical:abc123`)
3. Copy that ID into your `.env`:
   ```
   MEDICAL_MODEL_ID=ft:open-mistral-7b:mistralmind-medical:abc123
   ```

**Step 5 — The Router Uses It Automatically**

Once you set the env vars, `router.py` routes medical queries to your fine-tuned model automatically — no code changes needed.

### Convert Dataset to Mistral Console Format
```python
# Run this to convert your HuggingFace dataset to Mistral Console JSONL format
import json
from datasets import load_dataset

def convert_to_mistral_format(specialist: str, output_file: str, max_samples=1000):
    configs = {
        "medical":  ("qiaojin/PubMedQA", "pqa_labeled", "You are a world-class medical expert..."),
        "finance":  ("gbharti/finance-alpaca", None, "You are a senior quantitative analyst..."),
        "code":     ("sahil2801/CodeAlpaca-20k", None, "You are an expert software engineer..."),
        "creative": ("Dahoas/synthetic-instruct-gptj-pairwise", None, "You are a brilliant creative writer..."),
    }
    ds_id, ds_name, system = configs[specialist]
    load_kwargs = {"path": ds_id, "split": "train", "trust_remote_code": True}
    if ds_name: load_kwargs["name"] = ds_name
    ds = load_dataset(**load_kwargs)

    with open(output_file, "w") as f:
        count = 0
        for row in ds:
            instruction = row.get("instruction", row.get("question", row.get("prompt", "")))
            response    = row.get("output", row.get("long_answer", row.get("chosen", "")))
            if not instruction or not response: continue
            obj = {"messages": [
                {"role": "system",    "content": system},
                {"role": "user",      "content": str(instruction)},
                {"role": "assistant", "content": str(response)},
            ]}
            f.write(json.dumps(obj) + "\n")
            count += 1
            if count >= max_samples: break
    print(f"✅ {count} examples → {output_file}")

convert_to_mistral_format("medical",  "medical_mistral.jsonl")
convert_to_mistral_format("finance",  "finance_mistral.jsonl")
convert_to_mistral_format("code",     "code_mistral.jsonl")
convert_to_mistral_format("creative", "creative_mistral.jsonl")
```

---

## 📊 How to Check Your W&B Scores (Like the Benchmark Image)

1. **Go to** https://wandb.ai → sign in
2. **Click project** `mistralmind`
3. **You'll see runs like:**
   - `medical-eval-before` / `medical-eval-after`
   - `finance-eval-before` / `finance-eval-after`
   - etc.
4. **To see the bar chart comparison:**
   - Select both `medical-eval-before` AND `medical-eval-after`
   - Click **"Charts"** tab
   - Look for `keyword_coverage`, `safety_score`, `calculation_accuracy`, `pass_at_1`, `creativity_score`
5. **To create the exact table from the image:**
   - Click **"Table"** tab
   - Add columns: `keyword_coverage`, `safety_score`, `calculation_accuracy`, `pass_at_1`, `creativity_score`
   - Filter by `phase` column to compare before vs after

---

## 🎯 Killer Demo Queries

**PARALLEL — Health + Finance**
> *"I'm a 40-year-old engineer working 70+ hour weeks. What are the long-term health risks and how should I adjust my financial planning?"*

**SEQUENTIAL — Finance → Code**
> *"Write a Python microservice that monitors my portfolio and alerts when it drops 5% in a day. Include risk analysis."*

**PARALLEL — Full Startup**
> *"I'm building a mental health app for Gen Z. Analyze the market, suggest the tech stack, write the investor pitch."*

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Fast fine-tuning | **Unsloth** (2× faster, 50% less VRAM) |
| Training | **TRL SFTTrainer** |
| Experiment tracking | **Weights & Biases** |
| Routing + synthesis | **Mistral API** |
| Demo | **Gradio** |
| Base model | **Mistral-7B-Instruct-v0.3** |

---

*Built for the Mistral Global Hackathon 2026 — because one expert is never enough.*
