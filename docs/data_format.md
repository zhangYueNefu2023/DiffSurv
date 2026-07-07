# Data Format

DiffSurv expects one directory per cancer type. Each directory should contain one survival table and up to three molecular modality tables.

## Survival table

Expected filename:

```text
TCGA-{CANCER}.survival.tsv.gz
```

Required columns:

```text
sample
OS.time
OS
```

`OS.time` is the survival or follow-up time. `OS` should be coded as `1` for observed death/event and `0` for censoring.

## Omics tables

Expected filenames:

```text
TCGA-{CANCER}.star_tpm.tsv.gz
TCGA-{CANCER}.mirna.tsv.gz
TCGA-{CANCER}.protein.tsv.gz
```

Expected format:

```text
feature_id    sample_1    sample_2    sample_3
feature_a     0.12        0.30        0.08
feature_b     1.40        1.10        0.92
```

Rows are molecular features and columns are patient/sample IDs. The preprocessing script transposes these matrices internally.

## Missing modalities

A modality can be absent for a patient or absent for an entire cancer type. During preprocessing, missing modality tensors are filled with zeros and recorded in `metadata_nested_cv.csv` through modality indicators:

```text
has_mrna
has_mirna
has_protein
```

These indicators are used as the modality mask during training and Drop-and-Replace inference.
