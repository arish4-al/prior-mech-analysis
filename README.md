# Prior Mechanism Analysis

Analysis tools and model fitting code for studying the mechanisms of prior in decision-making using neural data from the International Brain Lab (IBL) brain-wide map dataset (https://www.nature.com/articles/s41586-025-09235-0).

**Preprint:** [https://www.biorxiv.org/content/10.64898/2025.12.15.694430v1](https://www.biorxiv.org/content/10.64898/2025.12.15.694430v1)

## Overview

This project analyzes how prior expectations influence neural activity across brain regions during perceptual decision-making tasks. The analysis includes:

- **Neural data analysis**: Computing prior sensitivity metrics across brain regions
- **Model fitting**: Two-stage optimization to fit a computational model
  - Stage 1: Retinal parameter fitting (`fit_retinal.py`)
  - Stage 2: Network weight fitting (`fit_weights.py`)

## Installation

Requires access to the IBL brain-wide map dataset (https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html).

## Main Scripts

- `block_analysis_allsplits.py` - Computes prior sensitivity metrics across brain regions
- `fit_retinal.py` - Stage 1: Fits retinal front-end parameters
- `fit_weights.py` - Stage 2: Fits network weights and gains
- `model_functions.py` - Model implementation and simulation
- `analysis_functions.py` - Analysis and plotting utilities
- `analysis_figs.ipynb` - Notebook for generating analysis figures

## Usage

See individual script files for usage examples. Results are saved to `one.cache_dir/manifold/res/` for analysis and timestamped directories under `save_dir` for model fitting.

