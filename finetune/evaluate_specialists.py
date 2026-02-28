"""
MistralMind - Evaluation & Benchmarking Suite
===============================================
Evaluates each specialist BEFORE and AFTER fine-tuning.
Produces W&B before/after comparison charts matching the target scores:

  🏥 Medical   : Keyword Coverage 52% → 84% (+32%) | Safety Score 61% → 91% (+30%)
  📈 Finance   : Calculation Accuracy 38% → 79% (+41%)
  💻 Code      : Pass@1 45% → 78% (+33%)
  🎨 Creative  : Creativity Score 0.51 → 0.78 (+53%)

Run:
  python finetune/evaluate_specialists.py --specialist medical  --phase before
  python finetune/evaluate_specialists.py --specialist medical  --phase after \\
    --checkpoint ./checkpoints/medical/lora_adapter
"""

import os, re, time, argparse
from typing import Optional
from dataclasses import dataclass

import torch
import wandb
from dotenv import load_dotenv
from unsloth import FastLanguageModel

load_dotenv()

# ← FIXED: correct base model (was unsloth/Llama-3.2-3B-Instruct)
BASE_MODEL     = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 512
MAX_SEQ_LEN    = 2048

# ─────────────────────────────────────────────────────────────────────────────
# Result Type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    specialist:  str
    phase:       str
    metric_name: str
    score:       float
    details:     dict

# ─────────────────────────────────────────────────────────────────────────────
# Inference Helper
# ─────────────────────────────────────────────────────────────────────────────

def generate(model, tokenizer, system: str, user: str,
             temperature: float = 0.3, max_new: int = MAX_NEW_TOKENS) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    inputs   = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            inputs, max_new_tokens=max_new, temperature=temperature,
            do_sample=temperature > 0, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

# ─────────────────────────────────────────────────────────────────────────────
# 🏥 Medical — Keyword Coverage + Safety Score
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_SYSTEM = (
    "You are a world-class medical expert. Provide thorough, "
    "evidence-based answers with step-by-step reasoning. Always consider patient safety."
)

MEDICAL_QUESTIONS = [
    {
        "q": (
            "A 45-year-old male presents with sudden crushing chest pain radiating to the left "
            "arm, diaphoresis, and nausea for 30 minutes. BP 140/90, HR 105. "
            "What is the most likely diagnosis and immediate management steps?"
        ),
        "keywords":        ["myocardial infarction", "STEMI", "MI", "aspirin",
                            "nitroglycerin", "ECG", "troponin", "oxygen"],
        "safety_keywords": ["emergency", "call", "immediate", "urgent", "hospital"],
    },
    {
        "q": ("Explain the mechanism of metformin in type 2 diabetes and list its major contraindications."),
        "keywords":        ["AMPK", "gluconeogenesis", "hepatic", "glucose", "insulin",
                            "renal", "lactic acidosis"],
        "safety_keywords": ["contraindicated", "avoid", "caution", "renal failure"],
    },
    {
        "q": (
            "A child has fever >38.5°C, severe headache, neck stiffness, photophobia, "
            "and a petechial non-blanching rash. Most urgent diagnosis and immediate treatment?"
        ),
        "keywords":        ["meningococcal", "meningitis", "septicaemia",
                            "ceftriaxone", "penicillin", "lumbar puncture"],
        "safety_keywords": ["emergency", "immediate", "antibiotic", "urgent"],
    },
    {
        "q": "What is the pathophysiology of septic shock vs hypovolemic shock?",
        "keywords":        ["vasodilation", "cytokines", "inflammatory", "SVR",
                            "cardiac output", "distributive", "systemic vascular resistance"],
        "safety_keywords": [],
    },
    {
        "q": "A patient on warfarin has INR 8.5 and minor gum bleeding. Management options?",
        "keywords":        ["vitamin K", "hold", "warfarin", "INR",
                            "reversal", "FFP", "bleeding risk"],
        "safety_keywords": ["stop", "withhold", "monitor", "urgent"],
    },
]

def evaluate_medical(model, tokenizer, phase: str) -> list:
    results, scores = [], []
    for i, item in enumerate(MEDICAL_QUESTIONS):
        resp     = generate(model, tokenizer, MEDICAL_SYSTEM, item["q"]).lower()
        kw_hits  = sum(1 for kw in item["keywords"] if kw.lower() in resp)
        kw_score = kw_hits / len(item["keywords"])
        sf_hits  = sum(1 for kw in item["safety_keywords"] if kw.lower() in resp)
        sf_score = sf_hits / len(item["safety_keywords"]) if item["safety_keywords"] else 1.0
        combined = 0.65 * kw_score + 0.35 * sf_score
        scores.append(combined)
        results.append(EvalResult("medical", phase, f"q{i+1}_score", combined,
            {"keyword_coverage": kw_score, "safety_score": sf_score,
             "keyword_hits": kw_hits, "response_len": len(resp.split())}))

    # Two headline metrics matching the README table
    avg_kw = sum(r.details["keyword_coverage"] for r in results[:5]) / 5
    avg_sf = sum(r.details["safety_score"] for r in results[:5]) / 5
    results.append(EvalResult("medical", phase, "keyword_coverage", avg_kw, {}))
    results.append(EvalResult("medical", phase, "safety_score",     avg_sf, {}))
    results.append(EvalResult("medical", phase, "overall_score",
                               sum(scores) / len(scores), {"n": len(MEDICAL_QUESTIONS)}))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 📈 Finance — Calculation Accuracy
# ─────────────────────────────────────────────────────────────────────────────

FINANCE_SYSTEM = (
    "You are a senior quantitative analyst. "
    "Solve financial problems step by step. State your final numerical answer clearly."
)

FINANCE_QUESTIONS = [
    {"q": "A company: $500M revenue, 20% EBITDA margin, sector EV/EBITDA 12x, $50M net debt. Equity value?",
     "expected": 1150.0, "tolerance": 0.05,
     "note":     "(500×0.2×12)−50 = 1150M"},
    {"q": "Present value of $1,000 in 5 years at 8% discount rate?",
     "expected": 680.58, "tolerance": 0.03,
     "note":     "1000/1.08^5 = 680.58"},
    {"q": "Portfolio: 12% annual return, 4% risk-free rate, 15% std dev. Sharpe ratio?",
     "expected": 0.533, "tolerance": 0.05,
     "note":     "(12−4)/15 = 0.533"},
    {"q": ("Bond: $1,000 face, 6% annual coupon, 3 years to maturity, market rate 8%. Price?"),
     "expected": 948.46, "tolerance": 0.03,
     "note":     "60/1.08 + 60/1.08² + 1060/1.08³ ≈ 948.46"},
]

def extract_number(text: str) -> Optional[float]:
    for pat in [
        r'(?:equity value|answer|result|total|value|price|ratio)\s*(?:is|=|:)\s*\$?\s*([\d,]+\.?\d*)',
        r'\$\s*([\d,]+\.?\d*)',
        r'≈\s*([\d,]+\.?\d*)',
        r'=\s*([\d,]+\.?\d*)',
        r'([\d,]+\.\d{2,})',
    ]:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                v = float(m.group(1).replace(',', ''))
                if v > 0: return v
            except: continue
    return None

def evaluate_finance(model, tokenizer, phase: str) -> list:
    results, scores = [], []
    for i, item in enumerate(FINANCE_QUESTIONS):
        resp  = generate(model, tokenizer, FINANCE_SYSTEM, item["q"], temperature=0.1)
        ext   = extract_number(resp)
        if ext is not None:
            err   = abs(ext - item["expected"]) / abs(item["expected"])
            score = max(0.0, 1.0 - err / item["tolerance"])
        else:
            err, score = float("inf"), 0.0
        scores.append(score)
        results.append(EvalResult("finance", phase, f"q{i+1}_accuracy", score,
            {"expected": item["expected"], "extracted": ext,
             "rel_error": round(err, 4) if err != float("inf") else None,
             "formula": item["note"]}))

    results.append(EvalResult("finance", phase, "calculation_accuracy",
                               sum(scores) / len(scores), {"n": len(FINANCE_QUESTIONS)}))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 💻 Code — Pass@1
# ─────────────────────────────────────────────────────────────────────────────

CODE_SYSTEM = (
    "You are an expert Python developer. Write only the requested function. "
    "Wrap your code in a ```python``` block."
)

CODE_TASKS = [
    {"prompt": "Write `fibonacci(n)` returning the nth Fibonacci (0-indexed) using dynamic programming.",
     "tests":  [("fibonacci(0)", 0), ("fibonacci(1)", 1), ("fibonacci(10)", 55), ("fibonacci(20)", 6765)]},
    {"prompt": "Write `two_sum(nums, target)` returning indices of two numbers summing to target. Use O(n) hashmap.",
     "tests":  [("sorted(two_sum([2,7,11,15],9))", [0,1]), ("sorted(two_sum([3,2,4],6))", [1,2])]},
    {"prompt": "Write `is_palindrome(s)` checking if s is a palindrome ignoring spaces and case.",
     "tests":  [("is_palindrome('racecar')", True),
                ("is_palindrome('A man a plan a canal Panama')", True),
                ("is_palindrome('hello')", False)]},
    {"prompt": "Write `flatten(lst)` that recursively flattens a nested list of any depth.",
     "tests":  [("flatten([1,[2,[3,4]],5])", [1,2,3,4,5]),
                ("flatten([[1,2],[3,[4,[5]]]])", [1,2,3,4,5])]},
]

def extract_code(text: str) -> str:
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def evaluate_code(model, tokenizer, phase: str) -> list:
    results, total_pass, total_tests = [], 0, 0
    for i, task in enumerate(CODE_TASKS):
        resp   = generate(model, tokenizer, CODE_SYSTEM, task["prompt"], temperature=0.2, max_new=600)
        code   = extract_code(resp)
        ns, passed = {}, 0
        try:
            exec(code, ns)
            for expr, expected in task["tests"]:
                try:
                    if eval(expr, ns) == expected: passed += 1
                except: pass
        except: pass
        score = passed / len(task["tests"])
        total_pass  += passed
        total_tests += len(task["tests"])
        results.append(EvalResult("code", phase, f"task{i+1}_pass_rate", score,
            {"passed": passed, "total": len(task["tests"])}))

    results.append(EvalResult("code", phase, "pass_at_1",
                               total_pass / total_tests if total_tests else 0.0,
                               {"total_passed": total_pass, "total_tests": total_tests}))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 🎨 Creative — Creativity Score
# ─────────────────────────────────────────────────────────────────────────────

CREATIVE_SYSTEM = "You are a brilliant creative writer. Be vivid, specific, and surprising."

CREATIVE_PROMPTS = [
    "Write the opening paragraph of a noir detective story set in a neon-lit cyberpunk city.",
    "Write a haiku capturing the feeling of receiving a message from someone you've missed.",
    "A scientist discovers their life's work was fundamentally wrong. Describe the moment — 3 sentences.",
    "Write a product description for an imaginary perfume called 'Last Algorithm'.",
]

SENSORY_WORDS = [
    "glow", "shadow", "whisper", "silence", "gleam", "haze", "flicker", "hum",
    "neon", "smoke", "cold", "warm", "sharp", "bitter", "echo", "fade", "pulse",
    "hollow", "ache", "crystalline", "fractured", "distant", "trembling",
]

def score_creative(text: str) -> dict:
    words  = text.lower().split()
    ttr    = len(set(words)) / len(words) if words else 0
    wc     = len(words)
    sensory = min(1.0, sum(1 for w in SENSORY_WORDS if w in text.lower()) / 4)
    length  = wc / 20 if wc < 20 else 1.0 if wc <= 200 else max(0.5, 1.0 - (wc-200)/400)
    specifics = len(re.findall(r'\b([A-Z][a-z]+|\d+|[a-z]+-[a-z]+)\b', text))
    spec   = min(1.0, specifics / 8)
    combined = 0.30*ttr + 0.25*sensory + 0.25*length + 0.20*spec
    return {"ttr": round(ttr,3), "sensory": round(sensory,3),
            "length": round(length,3), "specificity": round(spec,3),
            "combined": round(combined,3), "word_count": wc}

def evaluate_creative(model, tokenizer, phase: str) -> list:
    results, scores = [], []
    for i, prompt in enumerate(CREATIVE_PROMPTS):
        resp   = generate(model, tokenizer, CREATIVE_SYSTEM, prompt, temperature=0.85, max_new=300)
        sc     = score_creative(resp)
        scores.append(sc["combined"])
        results.append(EvalResult("creative", phase, f"prompt{i+1}_score", sc["combined"],
            {**sc, "preview": resp[:150]}))
    results.append(EvalResult("creative", phase, "creativity_score",
                               sum(scores)/len(scores), {"n": len(CREATIVE_PROMPTS)}))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────────────

EVALUATORS = {
    "medical":  evaluate_medical,
    "finance":  evaluate_finance,
    "code":     evaluate_code,
    "creative": evaluate_creative,
}

# Target scores from the benchmark table (for reference display)
TARGET_SCORES = {
    "medical":  {"before": {"keyword_coverage": 0.52, "safety_score": 0.61},
                 "after":  {"keyword_coverage": 0.84, "safety_score": 0.91}},
    "finance":  {"before": {"calculation_accuracy": 0.38},
                 "after":  {"calculation_accuracy": 0.79}},
    "code":     {"before": {"pass_at_1": 0.45},
                 "after":  {"pass_at_1": 0.78}},
    "creative": {"before": {"creativity_score": 0.51},
                 "after":  {"creativity_score": 0.78}},
}

def run_evaluation(specialist: str, phase: str, checkpoint: Optional[str] = None):
    wandb.init(
        project = os.getenv("WANDB_PROJECT", "mistralmind"),
        name    = f"{specialist}-eval-{phase}",
        tags    = ["eval", specialist, phase],
        config  = {"specialist": specialist, "phase": phase,
                   "model": checkpoint or BASE_MODEL},
    )

    model_path = checkpoint if (phase == "after" and checkpoint) else BASE_MODEL
    print(f"\n🔍 Loading: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    t0      = time.time()
    results = EVALUATORS[specialist](model, tokenizer, phase)
    elapsed = time.time() - t0

    # Log all metrics to W&B
    log_dict = {"eval_time_s": elapsed}
    for r in results:
        log_dict[r.metric_name] = r.score
        for k, v in r.details.items():
            if isinstance(v, (int, float)):
                log_dict[f"{r.metric_name}_{k}"] = v
    wandb.log(log_dict)
    wandb.log({"results_table": wandb.Table(
        columns=["metric", "score"],
        data=[[r.metric_name, round(r.score, 4)] for r in results]
    )})

    # Terminal summary with progress bars
    print(f"\n{'='*58}")
    print(f"  {'EVAL':} {specialist.upper()} | {phase.upper()}")
    print(f"{'='*58}")
    for r in results:
        bar = "█" * int(r.score*20) + "░" * (20 - int(r.score*20))
        print(f"  {r.metric_name:<35} {bar} {r.score:.3f}")
    print(f"\n  ⏱️  {elapsed:.1f}s")

    # Show how you track the scores from the image
    print(f"\n📊 HOW TO CHECK YOUR SCORES IN W&B:")
    print(f"   1. Go to https://wandb.ai → project 'mistralmind'")
    print(f"   2. Open run: '{specialist}-eval-{phase}'")
    print(f"   3. Click 'Charts' tab → see metric bars")
    print(f"   4. To compare before/after: click 'Group by' → specialist")

    wandb.finish()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist", choices=list(EVALUATORS.keys()), required=True)
    parser.add_argument("--phase",      choices=["before", "after"],     required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    run_evaluation(args.specialist, args.phase, args.checkpoint)
