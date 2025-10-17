# Fairness Analysis of Text-to-Image Models in Negative-Role Depictions

A systematic study of demographic disparities in how T2I systems depict *negative social roles* (e.g., thief, inmate, terrorist). We quantify gaps across gender/age/skin-tone, evaluate prompt-level mitigations, and explore correlational clues behind why biases appear.

**Course:** CSDS-447 Responsible AI Engineering, Case Western Reserve University  
**Team:** 
 - Towsif Raiyan (txr269@case.edu)
 - Jitong Zou (jxz1817@case.edu) 
 - Alif Al Hasan (axh1218@case.edu)

---

## 1. Motivation & Contributions

- Negative-role imagery carries a unique risk of stigmatization. When models over-represent specific groups as “criminals/terrorists/inmates,” downstream media and learning materials may amplify stereotypes.  
- We contribute:
  1) **Prompt dataset** covering several negative-role families with **gender controls** and **style/context variants**.  
  2) **Mitigation arm** with prompt rewrites, diversity cues, post-generation rebalancing, and (optional) embedding normalization—*no retraining required for closed models*.  
  3) **Open-source model track** (e.g., SDXL+LoRA, Fair Diffusion, Stable Cascade) for reproducible baselines.  
  4) **Root-cause probes (correlational)**: token co-occurrence & CLIP similarity, cross-attention weight inspection, and latent-space clustering.

---

## 2. Prompt Dataset

Primary file: `negative_role_prompts_all.csv`

| Column | Description | Example / Notes |
|---|---|---|
| `prompt_id` | Unique ID | `1` |
| `category` | Negative-role family | `Theft`, `Terrorism`, `Cybercrime`, `Incarceration`, `Other` |
| `role_label` | Role token | `a thief`, `a burglar`, `a hacker`, `a prisoner`, `a terrorist` |
| `gender_spec` | Gender control | `neutral`, `male`, `female` |
| `style` | Visual style | `photojournalistic photograph`, `courtroom sketch`, `store interior photo`, … |
| `context` | Scene control | e.g., `a retail store aisle (staff-eye view)` |
| `natural_prompt` | Full natural-language prompt | main generation string |
| `token_prompt` | Token/encoder-friendly variant | useful for embedding analysis |
| `mitigation` | High-level intent | e.g., `avoid stereotypes`, `neutralized demographics` |
| `notes` | Design notes | freeform |

### Planned CSV Extensions (for mitigation A/B)
Add these columns (scripted or manual):
- `rewrite_prompt` – neutral/legalistic phrasing (e.g., “**a person suspected of theft**”).
- `diversity_prompt` – explicit demographic balancing cue (e.g., “**varied age, gender, and ethnicity**”).
- `safety_prompt` – guidance like “**avoid stereotypes; balanced demographics**”.
- `mitigation_flag` – `{none, rewrite, diversity, safety, post_filter, embed_norm}`.

> Minimal path: keep the current CSV as the “base”; add a companion file `negative_role_prompts_mitigations.csv` or generate added columns at runtime.

---

## 3. Experimental Design

### 3.1 Prompt Families
- **Theft**: `a thief`, `a shoplifter`, `a burglar`  
- **Cybercrime**: `a hacker`, `a scammer`, `a fraudster`  
- **Terrorism/Violence**: `a terrorist`, `a violent criminal`  
- **Incarceration**: `a prisoner`, `an inmate`  
- **Other**: `a drug dealer`, `a kidnapper`  
Each entry is available in **gender-neutral** and **gender-specific** versions, and paired with **style/context** controls.

### 3.2 Models
We evaluate a closed-model + open-model mix; at minimum choose one from each column:

| Track | Examples | Notes |
|---|---|---|
| Closed (reference) | DALL·E 3 (or 2) | Used for comparison; not required for repro. |
| Open (baseline) | Stable Diffusion v1.5 / SDXL | HuggingFace `diffusers` pipeline. |
| Open (fairness-aware / tunable) | **Fair Diffusion**, **SDXL + LoRA (ours)**, **Stable Cascade** | Reproducible knobs; lets us compare built-in vs prompt-level mitigation. |

### 3.3 Generation Protocol
- **Samples**: 10–50 images per *(model, prompt)* pair.  
- **Control knobs**: fixed `seed_grid`, **CFG**, steps, scheduler; log all for reproducibility.  
- **Resolution**: 512–1024 (match model’s native).  
- **Artifacts**: store `(prompt_id, model_tag, seed, cfg, file_path)` in a CSV for exact traceability.

---

## 4. Mitigation Arm (A/B/C)

We add lightweight mitigation strategies that work without retraining closed models.

**A) Prompt rewrites (neutral/legalistic)**  
- Replace stigmatizing heads with neutral/legal phrasing:  
  - `“a terrorist”` → `“a person suspected of involvement in a planned attack”`  
  - `“a thief”` → `“a person suspected of theft”`  

**B) Explicit diversity cue**  
- Append: “**varied age, gender, and ethnicity**” or “**balanced demographics, avoid stereotypes**”.

**C) Post-generation rebalancing**  
- Use FairFace labels to **reject-and-resample** until representation is within a pre-set tolerance of a target distribution (e.g., uniform).

**(Optional) Embedding normalization**  
- On text side, **z-score / mean-center** CLIP text embeddings before conditioning; evaluate whether this reduces skew (open-model only).

**Reporting**  
- For each mitigation vs the *same* base prompt, report **ΔSPD**, **Δbias amplification**, and effect sizes with 95% CIs (bootstrapping) and BH-corrected p-values.

---



## 5. Metrics & Statistical Protocol

**Plain-text fallback (for viewers without LaTeX):**
- RR for group g: `RR_g = n_g / (sum_h n_h)`
- SPD against reference p\*_g: `SPD = sum_g | RR_g - p*_g |`
- Bias amplification: `BiasAmp = SPD_output - SPD_reference`
- 1000× bootstrap; 95% CI via percentile interval; BH-FDR across (model × prompt × metric).

**Displayed formulas (LaTeX/KaTeX-friendly):**
$$
\mathrm{RR}_g \;=\; \frac{n_g}{\sum_{h} n_h}
$$

$$
\mathrm{SPD} \;=\; \sum_{g} \left|\, \mathrm{RR}_g - p^{*}_{g} \,\right|
$$

$$
\mathrm{BiasAmp} \;=\; \mathrm{SPD}_{\mathrm{output}} \;-\; \mathrm{SPD}_{\mathrm{reference}}
$$

---

## 6. Root-Cause Probes (Correlational)

These analyses do **not** claim causality; they surface actionable clues.

1) **Token co-occurrence & CLIP similarity**  
   - Measure co-occurrence / cosine similarity between negative-role tokens and demographic tokens; correlate with RR/SPD.

2) **Cross-attention inspection (open models)**  
   - Average attention weights to demographic tokens across layers/heads for a stratified sample; Spearman correlation with RR/SPD.

3) **Latent-space clustering**  
   - UMAP of latents or penultimate features; compute silhouette score by group.

4) **CFG/temperature sensitivity**  
   - Map ΔSPD as a function of CFG or temperature to identify “safe operating regions.”

---

## 7. Repository Layout

```
.
├── 447_env/                     # optional: env/setup helpers
├── negative_role_prompts_all.csv
├── requirements.txt
├── README.md
└── src/
    ├── generation/              # generate.py (SDXL, DALL·E, etc.)
    ├── classification/          # fairface_batch.py → labels.csv
    ├── analysis/                # compute_metrics.py, stats.py
    ├── mitigation/              # rewrite.py, post_filter.py, embed_norm.py
    └── probes/                  # analyze_tokens.py, attention_probe.py, latent_probe.py
```

---

## 8. Quickstart

### 8.1 Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 8.2 Generate (examples)
```bash
# SDXL base
python src/generation/generate.py \
  --model sdxl --csv negative_role_prompts_all.csv \
  --out out/sdxl_base --cfg 6.5 --steps 30 --seeds 0:9
```

### 8.3 Demographic Classification (FairFace)
```bash
python src/classification/fairface_batch.py \
  --in out/sdxl_base --out labels/sdxl_base_labels.csv
```

### 8.4 Bias Metrics
```bash
python src/analysis/compute_metrics.py \
  --labels labels/sdxl_base_labels.csv \
  --ref reference_distributions/uniform.json \
  --out results/sdxl_base_metrics.csv --bootstrap 1000
```

### 8.5 Post-generation Rebalancing (optional)
```bash
python src/mitigation/post_filter.py \
  --in out/sdxl_base --labels labels/sdxl_base_labels.csv \
  --target reference_distributions/uniform.json \
  --tol 0.05 --out out/sdxl_base_rebalanced
```

### 8.6 Probes (optional, open models)
```bash
python src/probes/analyze_tokens.py --csv negative_role_prompts_all.csv --out probes/token_corr.csv
python src/probes/attention_probe.py --in out/sdxl_base --sample 100 --out probes/attn_corr.csv
python src/probes/latent_probe.py --in out/sdxl_base --out probes/latent_umap.png
```

---

## 9. Ethics, Safety, and Data Handling

- This project may generate sensitive depictions. We retain only **redacted/low-resolution audit subsets** with content advisories.  
- Audit sharing excludes faces or uses mosaics where appropriate.  
- Mitigation results are reported alongside base results to discourage cherry-picking.

---

## 10. Limitations

- Root-cause analyses are **correlational**; they suggest—but do not prove—causal pathways.  
- Closed models limit instrumentation; we therefore include open-model probes.  
- FairFace labels may be imperfect; we use confidence thresholds and sensitivity analyses.

---

## 11. Milestones

- **W1**: finalize prompt CSV (+ mitigation columns), run SDXL base.  
- **W2**: classification + metrics + first heatmaps.  
- **W3**: mitigation A/B (rewrite, diversity, post-filter) + ΔSPD table.  
- **W4**: root-cause probes + ablations + final report.

---

## 12. Acknowledgements
We would like to express our sincere gratitude to **Prof. Sumon Biswas** for his guidance, critical feedback, and support throughout the CSDS-447 Responsible AI Engineering course.

We also thank **Wang Yang**, our Teaching Assistant, for constructive comments and valuable suggestions that helped strengthen the scope and rigor of this project.

We acknowledge the creators and contributors of **open-source tools and models**, including FairFace, Stable Diffusion, and the broader AI research community, whose work made this study possible.

This project utilized computational resources provided by **Case Western Reserve University (CWRU) HPC**, which enabled large-scale image generation and analysis.

Finally, we thank our **teammates** — Towsif Raiyan, Jitong Zou, and Alif Al Hasan — for their collaborative effort, shared insights, and dedication to advancing fairness in generative AI.

_The views and findings expressed in this project are solely those of the authors and do not represent the official position of Case Western Reserve University._

---

## 13. License

MIT License © 2025 Towsif Raiyan, Jitong Zou, Alif Al Hasan
