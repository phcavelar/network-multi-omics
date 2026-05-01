# TODO — Reproducibility refactor

Scripts have been reorganised into `experiments/kotliarov2020/`, `experiments/tcga/`,
and `experiments/multimodal/`. The following items still need completing before the
reviewer response is ready.

---

## 1. Write README.md

- Describe what SCONE is and cite the paper.
- Document the repo structure (`nemo/`, `experiments/`, `notebooks/`, `pretrained/`, `yml/`).
- Environment setup: `bash setup_env.sh` + `conda activate nemo`.
- Training from scratch: Kotliarov2020 and TCGA sections with step-by-step commands.
- Reproducing results from pretrained weights: point to `notebooks/load_and_embed.ipynb`.
- Note the hard-coded `DATA_PATH` variables that users must edit (see item 3).

---

## 2. Write inference notebook: `notebooks/load_and_embed.ipynb`

Reviewer asked specifically for this. Must cover:

- Load the pretrained Kotliarov2020 RNA+CITE model from `pretrained/kotliarov2020-scone-run0/`.
  - Note: `SubsetContrastive(path_to_load_from=..., version_to_load_as=1)` — version arg is required.
- Download Kotliarov2020 data if not already present (reuse gdown calls from `experiments/kotliarov2020/02_train_scone.py`).
- Run inference on full dataset (no subsetting — use the full `adatas` directly through `gae.model(Xs, As)`).
- Compute Leiden clustering and report ARI + AMI against reference labels.
- Reproduce t-SNE coloured by batch, broad cell type, and fine cell type (Figure 3 of paper).

---

## 3. Externalise hard-coded data paths

Every script in `experiments/` currently has `DATA_PATH = "~/data/netemo"` or
`TCGA_PATH = "~/data/subtypemgtp"` hard-coded near the top. These must be changed
to read from an environment variable or a config file so users do not need to edit
source code.

Suggested approach: read `NEMO_DATA_PATH` / `TCGA_DATA_PATH` environment variables,
falling back to the current hard-coded defaults.

Affected files:
- `experiments/kotliarov2020/02_train_scone.py` (and all other kotliarov scripts)
- `experiments/tcga/01_train_scone.py` (and all other tcga scripts)

---

## 4. Train and release pretrained model weights

- Train at least one full run of the Kotliarov2020 RNA+CITE experiment
  (i.e. `experiments/kotliarov2020/02_train_scone.py`, repetition 0).
- Save the model with `gae.save(...)` and copy the output directory into `pretrained/kotliarov2020-scone-run0/`.
- If the weights exceed ~50 MB, host via GitHub Releases or Zenodo and update the README with the download link.
- The saved directory contains: `metadata.json`, `loading_dict.json`, `model`,
  `mod_criteria_0`, `mod_criteria_1`, `contrastive_modules_0`, `contrastive_modules_1`,
  `opt`, `sched`, `history.json`.

---

## 5. Decide fate of `yml/archive/netemo_cpu.yml`

`netemo_cpu.yml` was archived alongside the dev yml files but may be distinct from
`nemo_cpu.yml`. Confirm whether it is a duplicate before permanently removing it.

---

## 6. Fix `SubsetContrastive.load()` API (optional but recommended)

`load(path, v)` requires an explicit version integer. Consider defaulting `v=None`
to version 1 so users can call `SubsetContrastive(path_to_load_from="...")` without
guessing. Low-risk one-line change in `nemo/models/subsetcontrastive.py`.
