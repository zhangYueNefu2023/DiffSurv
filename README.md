# DiffSurv

Implementation of DiffSurv, a Drop-and-Replace diffusion framework for robust multi-omics survival prediction with incomplete modality availability.

This repository contains the public code skeleton for model definition, nested preprocessing, training, DDIM-style Drop-and-Replace inference, and controlled missing-modality evaluation. Patient-level data, processed tensors, and trained checkpoints are not included.

## Repository structure

```text
DiffSurv/
├── configs/
│   └── default.yaml
├── scripts/
│   ├── preprocess_nested.py
│   ├── train.py
│   ├── evaluate.py
│   └── evaluate_missingness.py
├── src/
│   └── diffsurv/
│       ├── model.py
│       ├── diffusion.py
│       ├── losses.py
│       ├── data.py
│       ├── metrics.py
│       └── config.py
├── docs/
│   └── data_format.md
├── examples/
│   └── example_run.sh
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/zhangYueNefu2023/DiffSurv.git
cd DiffSurv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA-enabled training, install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Data preparation

The preprocessing script expects one folder per cancer type. Each cancer folder should contain survival and omics tables using the following naming pattern:

```text
raw_data/
├── BLCA/
│   ├── TCGA-BLCA.survival.tsv.gz
│   ├── TCGA-BLCA.star_tpm.tsv.gz
│   ├── TCGA-BLCA.mirna.tsv.gz
│   └── TCGA-BLCA.protein.tsv.gz
└── BRCA/
    ├── TCGA-BRCA.survival.tsv.gz
    ├── TCGA-BRCA.star_tpm.tsv.gz
    ├── TCGA-BRCA.mirna.tsv.gz
    └── TCGA-BRCA.protein.tsv.gz
```

Run nested preprocessing:

```bash
python scripts/preprocess_nested.py \
  --raw-data-dir raw_data \
  --output-dir processed_data \
  --num-folds 5 \
  --variance-ratio 0.80
```

This creates:

```text
processed_data/
├── tensors/
├── metadata_nested_cv.csv
├── union_features.json
└── fold_specific_features.json
```

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

To train selected folds only:

```bash
python scripts/train.py --config configs/default.yaml --folds 0 1
```

Checkpoints are saved to:

```text
checkpoints/diffsurv_fold{fold}.pt
```

## Evaluation

Evaluate trained checkpoints:

```bash
python scripts/evaluate.py --config configs/default.yaml
```

The script saves patient-level risk scores to:

```text
outputs/evaluation_risk_scores.csv
```

## Missing-modality robustness

Run controlled modality-missingness evaluation:

```bash
python scripts/evaluate_missingness.py \
  --config configs/default.yaml \
  --rates 0.0 0.1 0.2 0.3 0.5 0.7 0.8 0.9
```

The missingness experiment randomly masks observed modality tokens and uses Drop-and-Replace DDIM inference to reconstruct missing latent tokens.

## Notes

- This repository does not include TCGA, METABRIC, CPTAC, patient-level tensors, or trained model checkpoints.
- The condition vector encodes cancer type only unless users explicitly modify the model inputs.
- The default configuration follows the manuscript-level architecture, but users should tune training parameters for their own hardware and dataset.

## Citation

If you use this code, please cite the corresponding DiffSurv manuscript.

```bibtex
@article{diffsurv,
  title   = {DiffSurv: A Drop-and-Replace Diffusion Framework for Robust Multi-omics Survival Prediction},
  author  = {Zhang, Yue},
  journal = {Manuscript in preparation},
  year    = {2026}
}
```

## License

This project is released under the MIT License.
