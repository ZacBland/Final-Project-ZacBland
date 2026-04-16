# Final-Project-ZacBland

Predicting nutritional content (calories, mass, fat, carbs, protein) from food images using deep learning. Based on the [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) dataset and CVPR 2021 paper (arXiv 2103.03375).

## Repository Structure

```
Final-Project-ZacBland/
├── Proposal/                    # Project proposal (PDF)
├── Final-Project-Report/        # Final report (PDF)
├── Final-Presentation/          # Presentation slides (PDF)
├── Code/
│   ├── train.py                 # Model training script
│   └── test.py                  # Model evaluation / inference script
├── scripts/
│   ├── download_nutrition5k.py  # Download & resize dataset from GCS
│   ├── lookup_dish.py           # Look up nutrition stats for a dish by ID
│   └── compute_statistics.py    # Compute MAE/MAE% from predictions CSV
├── data/                        # Dataset (not committed — downloaded via script)
│   ├── metadata/
│   │   ├── dish_metadata_cafe1.csv
│   │   ├── dish_metadata_cafe2.csv
│   │   └── ingredients_metadata.csv
│   ├── dish_ids/splits/
│   │   ├── rgb_train_ids.txt    # 4,058 training dish IDs
│   │   └── rgb_test_ids.txt     # 708 test dish IDs
│   └── imagery/
│       ├── realsense_overhead/{dish_id}/rgb.png
│       └── side_angles/{dish_id}/{camera}/*.png
├── checkpoints/                 # Saved model weights (not committed)
├── plots/                       # Training plots (auto-generated)
└── FinalProject_DL_Sp26.pdf     # Project requirements
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.1 (recommended)
- Google Cloud SDK (`gcloud` CLI) for downloading the dataset

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ZacBland/Final-Project-ZacBland.git
cd Final-Project-ZacBland
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Or install manually:

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install Pillow matplotlib
```

If not using GPU (CPU only):

```bash
pip install torch==2.3.1 torchvision==0.18.1
pip install Pillow matplotlib
```

### 3. Install Google Cloud SDK (for data download only)

Follow: https://cloud.google.com/sdk/docs/install

The Nutrition5k bucket is public — no authentication or project setup needed. If prompted by `gcloud init`, you can skip project selection.

### 4. Install ffmpeg (only needed for side-angle video frame extraction)

```bash
sudo apt install ffmpeg
```

## Running the Code

Scripts should be run **in this order**:

### Step 1: Download the dataset

```bash
python scripts/download_nutrition5k.py --output_dir ./data --resolution 224
```

This downloads overhead RGB images from the GCS bucket and resizes them to 224x224 to save disk space. The full dataset is 181GB but this selective download uses much less.

**Options:**
- `--max_dishes 10` — download only 10 dishes (for testing)
- `--include_side_angles` — also download side-angle videos, extract frames, and resize
- `--frame_interval 5` — extract every 5th frame from videos (default, matches the paper)
- `--cameras camera_A camera_B` — which cameras to extract

**Example (full download with side angles):**
```bash
python scripts/download_nutrition5k.py --output_dir ./data --resolution 224 --include_side_angles
```

### Step 2: Train the model

```bash
cd Code
python train.py
```

All training settings are configured via global variables at the top of `train.py`:

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `"./data"` | Path to data directory |
| `EPOCHS` | `50` | Number of training epochs |
| `BATCH_SIZE` | `32` | Training batch size |
| `LEARNING_RATE` | `0.001` | Initial learning rate (RMSProp) |
| `VAL_SPLIT` | `0.1` | Fraction of training data for validation |
| `CHECKPOINT_DIR` | `"./checkpoints"` | Where to save model weights |
| `PLOTS_DIR` | `"./plots"` | Where to save training plots |
| `MAX_DISHES` | `None` | Limit dishes for smoke testing |
| `IMAGE_SOURCES` | `["overhead", "side_angles"]` | Which image types to train on |
| `SIDE_CAMERAS` | `["camera_A", "camera_B"]` | Which cameras to use |

**Running in background on a server:**
```bash
nohup python train.py > train.log 2>&1 &
tail -f train.log
```

**Outputs:**
- `checkpoints/best_model.pth` — best model (lowest validation MAE)
- `checkpoints/final_model.pth` — model after last epoch
- `plots/` — training curves (loss, per-nutrient MAE%, learning rate)
- `plots/training_history.json` — raw metrics for custom plotting

### Step 3: Evaluate on the test split

```bash
cd Code
python test.py
```

Settings are configured via global variables at the top of `test.py`:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT` | `"./checkpoints/best_model.pth"` | Model to evaluate |
| `OUTPUT_CSV` | `None` | Set to a path to save predictions CSV |
| `IMAGE_SOURCES` | `["overhead"]` | Should match training config |

**Save predictions for external evaluation:**
```bash
# Edit test.py: set OUTPUT_CSV = "predictions.csv"
python test.py
python ../scripts/compute_statistics.py ../data/metadata/dish_metadata_cafe1.csv predictions.csv results.json
```

### Step 4: Predict on a single image

```bash
cd Code
python predict.py path/to/image.png
python predict.py path/to/image.png --checkpoint ../checkpoints/best_model.pth
```

## Utility Scripts

### Look up a dish's nutrition info
```bash
python scripts/lookup_dish.py dish_1561662216
```

### Compute evaluation statistics from predictions CSV
```bash
python scripts/compute_statistics.py data/metadata/dish_metadata_cafe1.csv predictions.csv results.json
```

## Model Architecture

Based on the Nutrition5k CVPR 2021 paper:

- **Backbone:** InceptionV3 pretrained on ImageNet (substitute for the paper's InceptionV2 + JFT-300M)
- **Input:** 299x299 RGB images (resize 320, center crop 299)
- **Shared layers:** 2x FC(4096) with ReLU + Dropout(0.5)
- **Task heads:** 5 independent FC(4096) -> FC(1) outputs
- **Outputs:** calories (kcal), mass (g), fat (g), carbs (g), protein (g)
- **Loss:** L1 (Mean Absolute Error)
- **Optimizer:** RMSProp with StepLR decay

## Metrics

- **MAE** (Mean Absolute Error) — average absolute prediction error in original units
- **MAE%** — MAE as a percentage of the mean ground truth value

## Dataset

[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (CC BY 4.0 license) contains ~5,006 real-world food dishes from Google cafeterias with:
- Overhead RGB + depth images
- Side-angle video from 4 cameras
- Per-dish nutritional labels (calories, mass, fat, carbs, protein)
- Per-ingredient breakdown

## References

- Thames et al., "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food," CVPR 2021. [arXiv:2103.03375](https://arxiv.org/abs/2103.03375)
- [Nutrition5k Dataset](https://github.com/google-research-datasets/Nutrition5k)
- [PyTorch InceptionV3](https://pytorch.org/vision/stable/models/inception.html)
