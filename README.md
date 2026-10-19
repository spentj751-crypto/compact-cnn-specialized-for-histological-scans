<div align="center">

# Compact CNN for Prostate Histology

**Efficient patch classification with a focus on CPU inference and patient-level validation**

ResNet18 · PyTorch · Digital Pathology

[Overview](#overview) · [Results](#results) · [Efficiency](#computational-efficiency) · [Methods](#methods) · [Limitations](#limitations)

</div>

---

## Overview

This research project explores compact neural networks for distinguishing **benign prostatic hyperplasia (BPH)** from **tumor tissue** in prostate histology patches. Its long-term goal is to make computational pathology more accessible to laboratories with limited computing resources.

The study compares two fully trainable **ResNet18** models with two classifiers built on a frozen **Phikon-v2** foundation model. It examines both computational cost and the effect of changing the patient represented in the training data.

| At a glance | Value |
| :--- | :--- |
| Compact architecture | ResNet18 with a two-class output layer |
| Model size | 11,177,538 parameters; approximately 45 MB of float32 weights |
| Measured CPU throughput | 143.92 patches/second on the reported benchmark system |
| Input patch size | 224 × 224 pixels |
| Study dataset | 24,607 patches from 11 patients |
| Evaluation design | Two patient-level splits, four models, three random seeds |

> **Research scope:** These are patch-level results from a small prostate histology cohort. They do not establish whole-slide diagnostic performance or clinical readiness.

### What the study shows

- **Compact inference is practical on a CPU.** ResNet18 used approximately 27× fewer parameters and achieved approximately 34× higher CPU throughput than Phikon-v2 with an MLP head in this benchmark.
- **Pretraining was associated with more consistent results across the two splits.** ImageNet-initialized ResNet18 achieved mean balanced accuracies of 97.80% and 98.03% across three seeds.
- **Training from scratch was sensitive to the patient split.** Its mean balanced accuracy changed from 99.77% to 89.45% when the training tumor patient was replaced.

The central question is how to combine a small computational footprint with reliable performance on patients not represented during training.

## Models

| ID | Architecture and initialization | Trainable parameters | Total parameters |
| :--- | :--- | ---: | ---: |
| **M1** | Frozen Phikon-v2 + nonlinear MLP head | 264,962 | 303,616,770 |
| **M2** | Frozen Phikon-v2 + linear head | 2,050 | 303,353,858 |
| **M3** | ResNet18 initialized with ImageNet weights | 11,177,538 | 11,177,538 |
| **M4** | ResNet18 trained from random initialization | 11,177,538 | 11,177,538 |

**M3 and M4 are the compact CNNs.** M4 uses the standard ResNet18 architecture and was trained entirely on this study's histology data. M1 and M2 serve as foundation-model comparators; their small classifier heads still require the full Phikon-v2 encoder for image inference.

## Results

All classification metrics below are percentages. **Balanced accuracy (BA)** is the mean of tumor sensitivity and BPH specificity. Each prediction corresponds to one image patch.

### Primary-seed results

**Split A — 6,765 validation patches**

| Model | Accuracy | BA | Macro-F1 | Tumor sensitivity | BPH specificity | Errors |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 98.83 | 98.87 | 98.78 | 98.67 | 99.07 | 79 |
| M2 | 98.83 | 98.89 | 98.78 | 98.62 | 99.15 | 79 |
| M3 | 97.72 | 97.69 | 97.63 | 97.84 | 97.55 | 154 |
| M4 | 99.63 | 99.69 | 99.62 | 99.39 | 100.00 | 25 |

**Split B — 7,884 validation patches**

| Model | Accuracy | BA | Macro-F1 | Tumor sensitivity | BPH specificity | Errors |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 98.91 | 98.96 | 98.79 | 98.81 | 99.11 | 86 |
| M2 | 99.16 | 99.24 | 99.07 | 99.00 | 99.48 | 66 |
| M3 | 97.54 | 97.96 | 97.30 | 96.63 | 99.29 | 194 |
| M4 | 89.32 | 86.18 | 87.60 | 96.11 | 76.24 | 842 |

The prespecified primary comparison was **M1 versus M3 in Split A at the primary seed**. M1 achieved a 1.18 percentage-point advantage in BA: 98.87% versus 97.69%. Other comparisons were exploratory.

M2 achieved similar performance to M1 while using approximately 129× fewer trainable head parameters. This reduction applies to the head, not to the complete inference model.

### Variation across three random seeds

Values are **mean BA (minimum–maximum)**. The ranges describe variation across seeds; they are not confidence intervals.

| Model | Split A | Split B | A − B, percentage points |
| :--- | ---: | ---: | ---: |
| M1 | 99.03 (98.87–99.26) | 98.98 (98.85–99.14) | +0.05 |
| M2 | 98.85 (98.84–98.89) | 99.26 (99.24–99.29) | −0.41 |
| M3 | 97.80 (97.59–98.12) | 98.03 (97.68–98.44) | −0.23 |
| M4 | 99.77 (99.69–99.85) | 89.45 (86.18–91.35) | +10.32 |

Differences are calculated from the rounded means. M4's decline occurred across all three seeds; its BA in Split B never exceeded 91.35%.

### Understanding the performance drop

At the primary seed, M4's lower performance in Split B was driven mainly by **false-positive tumor predictions on BPH tissue**:

- Tumor sensitivity remained at **96.11%**, while BPH specificity fell to **76.24%**.
- For patient BPH-2, **640 of 1,976 patches** were classified as tumor, giving a patient-specific patch accuracy of **67.61%**.
- **372 of 842 errors** assigned a probability of at least 0.95 to the incorrect class.
- All four models classified every patch from patient BPH-1 correctly in Split B.

The BPH validation patients were identical in both splits. Replacing the sole tumor patient used for training therefore changed M4's behavior on the same held-out BPH patient.

## Computational efficiency

| Metric | Phikon-v2 + MLP head (M1) | ResNet18 (M3, M4) |
| :--- | ---: | ---: |
| Total parameters | 303,616,770 | 11,177,538 |
| Approximate float32 weight size | 1.2 GB | 45 MB |
| CPU throughput, patches/second | 4.25 | 143.92 |
| Estimated CPU time for 24,607 patches | ≈97 minutes | ≈3 minutes |
| GPU throughput, patches/second | 445.03 | 8,949.30 |
| Peak GPU memory | 1.68 GiB | 0.33 GiB |
| Training time per run | 3–8 seconds for the head* | 7.7–21.8 minutes |

\* Phikon-v2 feature extraction took approximately **84 seconds once for the complete dataset** on the GPU. The 3–8 second head-training range was reported for M1 and M2 and excludes this preprocessing step. ResNet18 training updated the entire network.

**Benchmark conditions**

- **CPU:** batch size 16, float32, eight threads.
- **GPU:** batch size 64, bfloat16 mixed precision.
- **Input:** preloaded patches. Throughput does not represent the complete VSI-reading, patch-extraction, and inference pipeline.
- **Dataset processing times:** estimates derived from throughput, rather than timed end-to-end runs.

**Reported system:** Windows 11; AMD Ryzen 7 7800X3D, eight cores; 32 GB RAM; NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM; Python 3.11; PyTorch 2.12.

These measurements demonstrate CPU inference on the reported workstation. Performance on lower-specification laboratory computers remains to be measured.

## Methods

<details>
<summary><strong>Dataset construction and patient-level splits</strong></summary>

### Patch extraction

Images were read directly from **VSI files using Bio-Formats** at native resolution, without prior conversion. The physical scale was approximately **0.5119 μm/pixel** along both axes, so a 224 × 224 pixel patch covered approximately **114.7 × 114.7 μm**.

Patches were extracted on a non-overlapping grid with a stride of 224 pixels. A patch was eligible only if it lay entirely within an annotated region of one target class and did not intersect excluded regions.

Quality control required stained tissue to occupy at least 20% of each patch and excluded fully white or nearly uniform fields. Of **25,617 geometrically eligible patches**, **1,010** were rejected, all because of insufficient tissue coverage.

The final dataset contained:

| Class | Patients | Patches |
| :--- | ---: | ---: |
| Benign prostatic hyperplasia | 9 | 15,346 |
| Tumor | 2 | 9,261 |
| **Total** | **11** | **24,607** |

A heuristic indicator of possible blur was retained for error analysis only. It was not used to exclude patches.

### Patient-level split design

Training and validation were separated strictly by patient within each split. Because tumor tissue came from only two patients, their training and validation roles were exchanged:

- **Split A:** Cancer-2 in training; Cancer-1 in validation.
- **Split B:** Cancer-1 in training; Cancer-2 in validation.

The same two BPH patients were used for validation in both splits: BPH-1 contributed 718 patches and BPH-2 contributed 1,976. The other seven BPH patients remained in training.

| Split | Subset | Patients, BPH / tumor | BPH patches | Tumor patches | Total patches |
| :--- | :--- | :--- | ---: | ---: | ---: |
| A | Training | 7 / 1 (Cancer-2) | 12,652 | 5,190 | 17,842 |
| A | Validation | 2 / 1 (Cancer-1) | 2,694 | 4,071 | 6,765 |
| B | Training | 7 / 1 (Cancer-1) | 12,652 | 4,071 | 16,723 |
| B | Validation | 2 / 1 (Cancer-2) | 2,694 | 5,190 | 7,884 |

Patient separation, image checksums, and the absence of duplicate images between training and validation were checked programmatically. Both splits reuse the same overall cohort; they are not independent cohorts.

</details>

<details>
<summary><strong>Architecture and training protocol</strong></summary>

### Foundation-model comparators

M1 and M2 used a frozen Phikon-v2 ViT-L encoder with a fixed checkpoint revision. Each patch was converted into a **1,024-dimensional feature vector**, computed once and cached.

The M1 head consisted of:

`LayerNorm → Dropout(0.3) → Linear(1024, 256) → GELU → Dropout(0.3) → Linear(256, 2)`

M2 used a single `Linear(1024, 2)` layer. Only the classifier heads were trained; the Phikon-v2 encoder remained frozen.

### Compact networks

M3 and M4 used ResNet18 with its final layer replaced by a two-class classifier. All network parameters were trained. M3 started from ImageNet weights; M4 used random initialization.

### Optimization

Training settings were fixed before the experiments and recorded in a protocol file with a checksum. Hyperparameters were not tuned on the validation set.

| Setting | M1 / M2 | M3 | M4 |
| :--- | :--- | :--- | :--- |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 0.001 | 0.0001 | 0.001 |
| Weight decay | 0.0001 | 0.0001 | 0.0001 |
| Batch size | 256 | 128 | 128 |
| Maximum epochs | 50 | 50 | 50 |
| Early-stopping patience | 10 epochs | 10 epochs | 10 epochs |
| Augmentation | None | 90° rotations and flips | 90° rotations and flips |

The loss was unweighted cross-entropy. Color augmentation was not used.

To balance classes and patients, patches were sampled with replacement using weights inversely proportional to the number of training patients in their class and the number of patches contributed by their patient:

```text
patch_weight = 1 / (training_patients_in_class × patches_from_patient)
```

This sampling scheme gives equal expected contributions to both classes and to patients within each class.

The best checkpoint was selected using validation cross-entropy averaged with equal weights across classes and patients. Predictions used the class with the highest probability, corresponding to a 0.5 binary decision threshold. No probability calibration or threshold optimization was performed.

Each model–split configuration was evaluated with **three random seeds**, yielding **24 results**. The primary-seed M1 result for Split A was reused from an earlier run; the other **23 runs** followed the same protocol and comprised **566 epochs**, taking approximately **2.5 hours** in total.

</details>

<details>
<summary><strong>Evaluation and reproducibility</strong></summary>

The evaluation included accuracy, balanced accuracy, macro-F1, tumor sensitivity, BPH specificity, patient-specific patch accuracy, overlap between error sets, and confidently incorrect predictions.

- **Balanced accuracy:** mean of tumor sensitivity and BPH specificity.
- **Error overlap:** Jaccard index, defined as the intersection of two models' error sets divided by their union.
- **Confident error:** a prediction assigning probability ≥0.95 to the incorrect class.

Patches from the same patient are correlated observations. Patch-level p-values and confidence intervals were therefore not calculated. Results were summarized descriptively, with minimum–maximum ranges across seeds.

The study reports that metrics were independently recalculated from saved predictions and that the best checkpoints were rerun in a separate process. All **24 results** were reproduced exactly in those checks.

Study documentation was prepared with reference to the CLAIM reporting recommendations.

</details>

<details>
<summary><strong>Error overlap and image-quality analysis</strong></summary>

### Overlap between model errors

| Model pair | Jaccard index, Split A | Jaccard index, Split B |
| :--- | ---: | ---: |
| M1 / M2 | 0.65 | 0.58 |
| M1 / M3 | 0.28 | 0.14 |
| M1 / M4 | 0.05 | 0.02 |

M1 and M2 shared many errors, consistent with their common feature extractor. All four models were simultaneously wrong on only **2 patches in Split A** and **9 patches in Split B**. This observation alone does not establish the performance of an ensemble.

### Confidently incorrect predictions

Counts below use the primary seed and a probability threshold of 0.95 for the incorrect class.

| Model | Split A | Split B |
| :--- | ---: | ---: |
| M1 | 17 | 13 |
| M2 | 9 | 3 |
| M3 | 48 | 70 |
| M4 | 8 | 372 |

### Image quality and patch labels

For BPH-2, **158 of 1,976 patches** had a heuristic indicator of possible blur.

| Model and split | Error rate in blur-flagged patches | Error rate in other patches |
| :--- | ---: | ---: |
| M1, Split A | 7.59% | 0.72% |
| M3, Split A | 20.25% | 1.82% |
| M4, Split B | 27.85% | 32.78% |

The blur heuristic did not explain M4's decline in Split B: its error rate was not higher in the flagged subset.

Visual review of M1 errors also identified patches dominated by glandular lumen or fibrous stroma within larger annotated regions. Such patches may contain limited class-specific information when assigned the label of the entire region.

</details>

## Limitations

- **Small patient cohort.** The dataset contains 11 patients, including only two with tumor tissue. A large patch count does not replace patient diversity.
- **Validation-based checkpoint selection.** Validation data were used for early stopping and selection of the best checkpoint. The reported results are not measurements on an independent final test cohort.
- **Shared patients across experimental splits.** The BPH validation patients are reused, and the two tumor patients exchange roles. The two splits do not provide independent evidence of external generalization.
- **Patch-level task.** The experiment does not establish patient-level diagnoses, whole-slide aggregation, cancer grading, or performance on other organs and conditions.
- **Region-derived labels.** Some patches within an annotated region contain little tissue that distinguishes the target classes.
- **Uncalibrated probabilities.** High model confidence should not be interpreted as a calibrated probability of a correct prediction.
- **Hardware-specific measurements.** Throughput was measured on one workstation with preloaded inputs; performance on other computers and in a complete slide-processing workflow requires separate measurement.

## Development direction

The next research priorities are to expand the patient cohort, evaluate on an independent external dataset, assess probability calibration, improve handling of low-information patches, and benchmark the complete workflow on lower-specification CPUs.

The aim is a compact, reproducible histology model whose computational efficiency is supported by evidence of generalization across patients.
