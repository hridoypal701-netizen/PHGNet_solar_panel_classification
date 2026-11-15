# Dataset directory

Place the dataset here if running locally. The training script expects a folder named `Faulty_solar_panel` with six class subfolders:

```text
data/
└── Faulty_solar_panel/
    ├── class_1/
    ├── class_2/
    ├── class_3/
    ├── class_4/
    ├── class_5/
    └── class_6/
```

The script's `POSSIBLE_PATHS` list is currently configured primarily for Kaggle paths. Add the exact local path to that list before running locally.

The dataset is intentionally excluded from Git by `.gitignore`. Check the dataset provider's license and redistribution rules before sharing it.
