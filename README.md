# GPR Fault Slip Distribution Generator

Predicts earthquake fault geometry and slip statistics from a trained
Gaussian Process Regression (GPR) model, then generates a spatially
correlated slip distribution using a Fortran spectral slip generator
(Von Kármán / Exponential / Gaussian filters).

**Pipeline:** `Mw + fault style inputs` → **GPR model** → fault length,
width, asperity areas, mean slip/std → **slipgen.in** → **Fortran
program** → correlated slip grid.

## Repository structure

```
.
├── models/                     # Trained GPR model + preprocessing artifacts
│   ├── gpr_model.pkl            # MultiOutputRegressor wrapping per-target GaussianProcessRegressor estimators
│   ├── gpr_mw_scaler.pkl        # Scaler fit on the Mw (magnitude) input
│   ├── gpr_ohe.pkl              # OneHotEncoder fit on the categorical inputs (SVT, FM, TS, NM)
│   ├── gpr_y_scaler.pkl         # Scaler used to inverse-transform target predictions
│   └── gpr_metadata.pkl         # Column names, target names, valid categorical values, fixed defaults
├── src/
│   ├── run_pipeline.py          # Main Python pipeline (load model → predict → write slipgen.in)
│   └── slipgen_von_karman.f90   # Fortran spatial-correlation slip generator (consumes slipgen.in)
├── examples/
│   └── slipgen.in.example       # Sample generated input file, for reference
├── docs/
│   └── slipgen_format.md        # Description of the slipgen.in file format
├── requirements.txt
├── .gitattributes                # Git LFS tracking for the model binary
├── .gitignore
└── LICENSE
```

## Setup

### Python environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This repo uses **Git LFS** for `models/gpr_model.pkl` (~32 MB). Install
[Git LFS](https://git-lfs.com/) before cloning, or run `git lfs pull`
after cloning if the file appears as a small pointer file.

### Fortran compiler

The slip generator needs a Fortran compiler, e.g. `gfortran`:

```bash
# Debian/Ubuntu
sudo apt install gfortran

# macOS (Homebrew)
brew install gcc
```

Compile it:

```bash
gfortran -O2 -o slipgen src/slipgen_von_karman.f90
```

## Usage

### 1. Generate `slipgen.in` from a scenario

```bash
python src/run_pipeline.py
```

By default this runs a Mw 6.5 strike-slip scenario and writes
`slipgen.in` in the working directory. Edit the `run_pipeline(...)`
call at the bottom of `src/run_pipeline.py`, or import it into your
own script:

```python
from src.run_pipeline import run_pipeline

run_pipeline(
    Mw=7.2,
    SVT="asymetriccosine ",   # see metadata['svt_unique'] for valid values
    FM="rv",                   # fault mechanism: e.g. 'ss', 'rv', 'nm', 'th'
    TS="0",
    NM="1",
    model_choice="von_karman", # "von_karman", "exponential", or "gaussian"
    n_blocks=5,                 # 3, 4, or 5 recommended; higher gives unrealistic slips
    output_path="slipgen.in",
)
```

### 2. Run the Fortran slip generator

```bash
./slipgen
```

This reads `slipgen.in` and produces the correlated slip field plus
several diagnostic files (see `docs/slipgen_format.md`).

## Model notes

- The GPR model was trained with **scikit-learn 1.7.1** (pinned in
  `requirements.txt`). Loading with a different sklearn version may
  raise an `InconsistentVersionWarning` and can subtly change
  prediction output — re-pin or retrain if you need to upgrade.
- `models/gpr_metadata.pkl` contains the valid values for each
  categorical input (`svt_unique`, `fm_unique`, `ts_unique`,
  `nm_unique`) as well as the fixed defaults used when a category
  isn't specified (`svt_fixed`, `fm_fixed`, `ts_fixed`, `nm_fixed`).
- The default `NM="0"` in the `__main__` example is **not** one of the
  categories the encoder was trained on (`metadata['nm_unique']` is
  `['2','1','3','5','6','7','4']`) — the one-hot encoder will emit a
  `UserWarning` and encode it as all-zeros. Use a valid `NM` value from
  `metadata['nm_unique']` unless you specifically want the "unknown
  category" behavior.
- There is currently no training script in this repo — only the
  serialized model artifacts. If you have the notebook/script that
  produced them, consider adding it under a `training/` directory so
  the model can be reproduced or retrained.
## Training Dataset

Model training dataset can be accessed through this link: https://github.com/VarunD21029/LME-Source-Scaling-Relations/blob/main/supplementary_data_revised.xlsx

## License

See [LICENSE](LICENSE).
