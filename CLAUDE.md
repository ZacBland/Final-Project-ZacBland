# CLAUDE.md

## Project Overview

This is a Deep Learning final project for predicting nutritional content (calories, mass, fat, carbs, protein) from food images using the Nutrition5k dataset. The project follows the architecture from the CVPR 2021 paper (arXiv 2103.03375).

## Key Architecture Decisions

- **InceptionV3** pretrained on ImageNet is used as the backbone (the paper used InceptionV2 + JFT-300M which is not publicly available)
- Pretrained InceptionV3 requires `aux_logits=True` during construction; we set `backbone.aux_logits = False` and `backbone.AuxLogits = None` after loading weights
- Multi-task regression: 5 outputs (calories, mass, fat, carbs, protein)
- Loss: L1 (MAE), Optimizer: RMSProp, matching the paper

## Repository Layout

- `Code/train.py` — training script (all config via global vars at top, no argparse)
- `Code/test.py` — evaluation script (imports model/dataset from train.py)
- `scripts/download_nutrition5k.py` — downloads images from GCS bucket, resizes to save space
- `scripts/lookup_dish.py` — CLI tool to look up a dish's nutrition by ID
- `scripts/compute_statistics.py` — computes MAE/MAE% from a predictions CSV (from original Nutrition5k repo)
- `data/` — dataset directory (not committed, downloaded via script)
- `checkpoints/` — saved model weights (not committed)
- `plots/` — auto-generated training curves
- `Proposal/` — project proposal PDF
- `Final-Project-Report/` — final report PDF
- `Final-Presentation/` — presentation slides PDF

## Data Details

- GCS bucket: `gs://nutrition5k_dataset/nutrition5k_dataset`
- Uses `gcloud storage cp` (NOT deprecated `gsutil`)
- Metadata filename is `ingredients_metadata.csv` (plural "ingredients")
- Not all dishes in metadata have overhead images (~3,500 of 5,006)
- Train split: 4,058 dishes, Test split: 708 dishes
- Metadata CSV has no header. Format: `dish_id,cal,mass,fat,carb,protein,[ingr_id,ingr_name,ingr_grams,ingr_cal,ingr_fat,ingr_carb,ingr_protein]*N`
- Side-angle videos are `.h264` format, need ffmpeg to extract frames
- Side-angle frames are typically upside down — use `RandomVerticalFlip(p=0.8)` during training

## Framework Versions

- PyTorch: `2.3.1+cu121`
- torchvision: `0.18.1+cu121`
- Python: 3.10+
- Training server: Ubuntu Linux with NVIDIA GPU

## Important Patterns

- train.py and test.py use **global variables** for configuration (no argparse)
- test.py imports `NutritionModel`, `Nutrition5kDataset`, `LABEL_NAMES`, `SIDE_CAMERAS`, `SIDE_ANGLE_VFLIP_PROB` from train.py
- Training plots are saved every 5 epochs to `./plots/` with matplotlib Agg backend
- Training history is saved as `plots/training_history.json`
- The download script lists overhead dish IDs from the GCS bucket directly (not from metadata) since not all metadata dishes have images

## Project Deadlines

- Proposal: March 26, 2026 (done)
- Presentation slides PDF: April 27, 2026
- Oral presentations: April 28-30, 2026
- Final report: May 5, 2026

## Code Style

- Scripts use descriptive global variables at the top for configuration
- No argparse — all settings are edited directly in the file
- Matplotlib uses Agg backend for headless server rendering
- Print progress inline with `\r` for batch-level updates during training
