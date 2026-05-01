# SCONE: Subset-Contrastive Multi-Omics Network Embedding

SCONE is a scalable framework for multi-omics data integration using graph contrastive learning. It trains on overlapping random subsets of a similarity graph rather than the full graph, keeping memory growth linear rather than quadratic. This makes it practical for large single-cell datasets where full-graph methods run out of memory.

The method is described in:

> Avelar P.H.C., Cardoso-Silva J., Wu M., Tsoka S. (2025). SCONE: Subset-Contrastive Multi-Omics Network Embedding. *[TBC]*.

## Repository structure

```
network-multi-omics/
├── nemo/                   The SCONE Python package
├── experiments/            Scripts to reproduce all experiments in the paper
│   ├── kotliarov2020/      Single-cell PBMC dataset (Kotliarov et al. 2020)
│   ├── tcga/               Bulk multi-omics dataset (TCGA, seven cancer types)
│   └── multimodal/         Multi-modal extension experiments
├── notebooks/              Inference walkthrough (load pretrained model, embed new data)
├── pretrained/             Released model weights for the Kotliarov2020 RNA+CITE experiment
└── yml/                    Conda environment files
```

## Environment setup

SCONE requires Python 3.10+, PyTorch, and PyTorch Geometric.

**Linux / macOS:** use the `setup_env.sh` script, which creates a conda environment called `nemo` and selects the correct dependencies for your hardware:

```bash
bash setup_env.sh
conda activate nemo
```

**Windows:** run the equivalent PowerShell script:

```powershell
.\setup_env.ps1
conda activate nemo
```

Both scripts detect whether a CUDA-capable GPU is available and install from `yml/nemo_gpu.yml` or `yml/nemo_cpu.yml` accordingly.

## Training from scratch

### Kotliarov2020 (single-cell, RNA + CITE-seq)

Download the data first:

```bash
cd experiments/kotliarov2020
jupyter nbconvert --to notebook --execute 01_download_data.ipynb
```

Then train SCONE with both modalities (RNA + CITE-seq). This runs eight independent replicates, as reported in the paper:

```bash
python 02_train_scone.py
```

Trained models are saved to `~/data/netemo/models/` and embeddings to `~/data/netemo/results/`. Set the `NEMO_DATA_PATH` environment variable to change these locations. Training takes approximately [X hours] on a single GPU (NVIDIA RTX A4000, 8 GB VRAM) or [Y hours] on CPU.

Key hyperparameters (matching the paper):

| Parameter | Value |
|---|---|
| Subset size | 20% of dataset |
| KNN neighbours | 15 |
| Epochs | 128 |
| Learning rate | 1e-4 |
| Reconstruction weight β | 10 |
| Contrastive weight α | 1 |

## Reproducing results from pretrained weights

A trained model from run 0 of the Kotliarov2020 RNA+CITE experiment is provided in `pretrained/kotliarov2020-scone-run0/`. The notebook `notebooks/load_and_embed.ipynb` walks through loading this model and reproducing the t-SNE visualisations in Figure 3 of the paper.

To run the notebook:

```bash
conda activate nemo
jupyter notebook notebooks/load_and_embed.ipynb
```

The notebook covers:

1. Loading the pretrained model
2. Running inference on the full Kotliarov2020 dataset (no subsetting at inference)
3. Computing Leiden clustering and ARI/AMI against reference labels
4. Reproducing the t-SNE plots coloured by batch, broad cell type, and fine cell type
