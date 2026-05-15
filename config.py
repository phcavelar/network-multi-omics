"""
Centralised configuration for all network-multi-omics experiment scripts.

Paths are read from environment variables so users do not need to edit source code.
Set the variables below in your shell or conda environment before running any script.

    export NEMO_DATA_PATH=/my/data/netemo        # model outputs, embeddings, results
    export TCGA_DATA_PATH=/my/data/subtypemgtp   # TCGA omics data

If neither variable is set, the defaults below are used (matching the original hard-coded values).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_PATH = Path(os.environ.get("NEMO_DATA_PATH", "~/data/netemo")).expanduser()
TCGA_PATH = Path(os.environ.get("TCGA_DATA_PATH", "~/data/subtypemgtp")).expanduser()

# ---------------------------------------------------------------------------
# Subdirectory names (relative to DATA_PATH)
# ---------------------------------------------------------------------------

MODEL_DIR = "models"
RESULTS_PATH = "results"

# ---------------------------------------------------------------------------
# Kotliarov2020 dataset — filenames and Google Drive download links
# ---------------------------------------------------------------------------

SCRNA_FNAME = "kotliarov2020-expressions.h5ad"
SCRNA_LINK = "https://drive.google.com/uc?id=1wA3VBUnYEW2qHPk9WijNTKjV9KriWe8y"

SCCITE_FNAME = "kotliarov2020-proteins.h5ad"
SCCITE_LINK = "https://drive.google.com/uc?id=112mdDX76LZRL33tBLYhfYRRXOUrLUhw-"
