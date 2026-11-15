# PHGNet Solar Panel Classification

A PyTorch research implementation for six-class solar-panel image classification using an EfficientNet-B0 backbone and optional persistence-diagram (topological) feature guidance.

> **Important:** The uploaded training script currently sets `MODES = ["msbase"]`. This runs the multi-scale vision baseline. To run the topology-based variants, change `MODES` to `["concat"]` or `["phg"]` in `phgnet_solar_panel.py`. The available `baseline`, `msbase`, `concat`, and `phg` modes are ablation configurations; do not describe an `msbase` run as a completed PHG/topology-guided experiment.

## Repository contents

```text
PHGNet_solar_panel_classification/
├── phgnet_solar_panel.py
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── README.md
```

## Method overview

The script is configured with:

- Backbone: `efficientnet_b0` from `timm`
- Input resolution: `384 × 384`
- Number of classes: `6`
- Optimizer: AdamW
- Loss: Focal Loss
- Validation: stratified 10-fold cross-validation
- Topological features: GUDHI cubical-complex persistence diagrams, computed from grayscale, inverted grayscale, and gradient filtrations
- Output: fold metrics, confusion matrices, training curves, persistence cache, and checkpoints while training

The `phg` mode applies persistence-guided feature fusion. The `concat` mode concatenates topological and visual features. `msbase` pools multi-scale visual features without using persistence diagrams. `baseline` uses the final backbone feature map.

## Dataset

The script expects a directory named `Faulty_solar_panel` containing one subdirectory per class:

```text
Faulty_solar_panel/
├── class_1/
│   ├── image_001.jpg
│   └── ...
├── class_2/
├── class_3/
├── class_4/
├── class_5/
└── class_6/
```

The current script has strict dataset checks. It requires **exactly 885 images** and the sorted-class counts to equal:

```python
[207, 193, 190, 103, 69, 123]
```

If your dataset has a different number of images or class distribution, the script will stop with an assertion error. Update the assertion in `SolarPanelPHDataset.__init__` only after verifying your actual dataset and documenting the change. Class names and label IDs are assigned alphabetically from folder names.

### Kaggle dataset setup

1. Create or open a Kaggle Notebook.
2. Add the solar-panel dataset to the notebook using **Add Input**.
3. Confirm that the input contains the `Faulty_solar_panel` folder and six class subfolders.
4. If Kaggle mounted it at a path not listed in `POSSIBLE_PATHS`, add the exact path to that list in `phgnet_solar_panel.py`.
5. Upload `phgnet_solar_panel.py` to the notebook or clone this repository.

## Installation

Use a Python environment with a PyTorch build compatible with your machine. Install PyTorch and torchvision using the command recommended by the official PyTorch installation selector for your operating system and CUDA version.

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

For Kaggle, PyTorch and torchvision are commonly preinstalled. Check the installed versions before replacing them.

## Running

Run from a terminal or a Kaggle Notebook:

```bash
python phgnet_solar_panel.py
```

The script automatically selects CUDA when available; otherwise it uses CPU. The configured image size, batch size, epoch count, folds, and output directory are defined near the top of the script.

### Selecting an experiment mode

Edit this line in `phgnet_solar_panel.py`:

```python
MODES = ["msbase"]
```

Choose one mode per run:

```python
MODES = ["baseline"]  # final backbone feature map
MODES = ["msbase"]    # multi-scale visual baseline
MODES = ["concat"]    # visual + persistence features
MODES = ["phg"]       # persistence-guided feature fusion
```

The script is configured for 120 epochs and 10 folds. Topological persistence computation and 384 × 384 images can be computationally expensive. A GPU-enabled Kaggle session is recommended. A full 10-fold experiment may take substantial time.

## Outputs

The output directory is:

- Kaggle: `/kaggle/working/phgnet_efficientnet_v8_885/`
- Other environments: `./phgnet_efficientnet_v8_885/`

Expected artifacts include:

- `final_results_v8_885.csv`: fold-level result records
- `pd_cache_v8_384_885.npz`: cached persistence diagrams and dataset metadata
- `curves_<mode>_f<fold>.png`: training and validation curves
- `cm_<mode>_f<fold>_s<seed>.npy`: confusion matrices

Download important outputs from Kaggle before the session ends.

## Reproducibility notes

- Random seeds are configured in `seed_everything`.
- The script uses `SPLIT_SEEDS = [42]` and `MODEL_SEEDS = [42]`.
- Class-to-index mapping is alphabetical, based on directory names.
- The persistence cache is tied to the dataset ordering and current preprocessing settings. Rebuild it if the dataset, image ordering, image size, or persistence settings change.
- The code saves fold results to CSV after each completed fold.

## Limitations and checks before publishing results

- The script currently enforces one exact dataset size and class-count distribution.
- The active mode in the committed script is `msbase`, not `phg`.
- The script uses pretrained EfficientNet-B0 weights from `timm`; first use may require internet access to download weights unless they are already cached.
- Report results only for the experiment mode that was actually run. Keep dataset source, split strategy, preprocessing, and model settings consistent when comparing ablations.
- Add the dataset's original citation and license here after confirming the dataset source and its terms of use.

## Citation

If you use this code in academic work, cite the repository and the associated research paper if/when one is published. Do not cite an unpublished paper or claim peer review unless that is accurate.
