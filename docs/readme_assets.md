# README figures and result notes

The homepage reuses two figures from **RAPTOR: Ridge-Adaptive Logistic Probes**, by
Ziqi Gao, Yaotian Zhu, Qingcheng Zeng, Xu Zhao, Ziqing Wang, Feng Ruan and Kaize Ding,
[arXiv:2602.00158v4](https://arxiv.org/abs/2602.00158v4). The paper figures retain the
paper's [CC BY 4.0 attribution](https://creativecommons.org/licenses/by/4.0/).
Only their file format changed: the complete PDF pages were rasterized to PNG
with white backgrounds. No labels, colors, layouts or numerical values were
changed. The pending [software license](../LICENSE) is separate.

## Original artwork

| Homepage asset | Paper source | Size |
|---|---|---|
| [RAPTOR overview](../assets/raptor_overview.png) | `main_ralop_v2.pdf`, included at `main.tex:166` | 1637 × 656 px |
| [Accuracy differences](../assets/accuracy_differences.png) | `raptor_accuracy_differences_5seeds.pdf`, included at `main.tex:603` | 1859 × 879 px |

The original PDFs match the existing
[manuscript identity record](validation/manuscript_identity.json). Their SHA256s:

```text
main_ralop_v2.pdf
6f48a23eb7135e06b6fb92662e217183e9c4d609d390dadf8cc7e68295922957
raptor_accuracy_differences_5seeds.pdf
db2408240ea9cc68b45425b406515a47a2887817d0eeeb70e1e85a7c758183a7
```

The distributed PNG SHA256s:

```text
assets/raptor_overview.png
0918bf26dde2bfcb55606617622556f42c5883f0731e30b7c998ca4e7b578b90
assets/accuracy_differences.png
12db1a1f2e25810072b36b13f5ac517c6aa9ced62663b14150bd8b85c3957e70
```

Conversion used PyMuPDF 1.26.5 / MuPDF 1.26.10, a 2× scale (144 dpi), RGB output,
and the entire first page of each single-page source. Obtain the v4 source from
the paper page, verify the source hashes above, and use this optional conversion
recipe; PyMuPDF is not needed to run RAPTOR:

```python
from pathlib import Path
import fitz

source = Path("paper-source")  # extracted v4 source, not distributed here
for original, target in [
    ("main_ralop_v2.pdf", "assets/raptor_overview.png"),
    ("raptor_accuracy_differences_5seeds.pdf", "assets/accuracy_differences.png"),
]:
    with fitz.open(source / original) as document:
        assert len(document) == 1
        document[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(target)
```

Both README images link to their full-resolution PNG. The white background keeps
paper labels legible in both light and dark GitHub themes.

## Reading the figures

The overview depicts frozen last-token hidden states, train-fitted
standardization, ridge-logistic fitting, and a normalized direction for optional
steering. It is a conceptual diagram, not the full accuracy protocol. In the
public RAPTOR implementation, validation chooses the native C value before a
train+validation refit. The raw affine readout is `h @ W + B`; the intercept is
not part of a robustness direction. Unit normalization in the drawing is not
applied to the Quick Start's returned `W` and `B`. See [accuracy](accuracy.md) for
the precise native regularization normalization and [steering](appendix_and_steering.md)
for the distinct `legacy_unit` and `raw_calibrated` control protocols.

The accuracy figure uses six models, six datasets, five methods and outer seeds
42–46. Top panels show RAPTOR-minus-baseline **average-layer test accuracy**;
bottom panels show **validation-selected best-layer test accuracy**. For each
method and seed, the best layer maximizes its own pre-refit score on the shared
outer validation split, with the lowest layer index breaking exact ties. Test
accuracy does not select that layer. Layer averages cover every fixed physical
layer. Each displayed color is the difference between the methods' five-seed
means in percentage points, using the original purple/white/green ±15 scale.
Purple favors RAPTOR and green favors the baseline. The figure does not plot
uncertainty; the accompanying tables report **sample SD (`ddof=1`)** across the
five outer seeds. It does not depict grouped sensitivity.

## Numeric statements on the homepage

These are descriptive checks of the existing numeric references, not newly
trained results or statistical-significance claims:

- In [accuracy_cells.csv](../reported_results/accuracy_cells.csv), filter to the
  six named models and six individual datasets (exclude `dataset=six_dataset_mean` rows),
  with `metric=average_layer_test_accuracy`. RAPTOR (`singlelr`) has a larger reported
  mean than `gcs`, `fixed_reg` and `dim_midpoint` in each of the 36 model–dataset
  settings. The RAPTOR-minus-xRFM difference changes sign by setting.
- In [robustness_settings.csv](../reported_results/robustness_settings.csv), the
  median of the 18 `mean` entries for `method=raptor` is
  `0.895030013922308`, displayed as **0.895**. Each setting first averages the
  per-layer mean absolute cosine of all 190 pairs from K=20 perturbations after
  20% training-pool drops. It is not an outer-seed SD or a proof of superiority:
  GCS and CAA have higher reported medians. Native direction coordinates and
  finite nonconverged directions remain as documented in [robustness.md](robustness.md).
- [timing_reported_mixed.csv](../reported_results/timing_reported_mixed.csv) retains
  mixed historical sources. The homepage makes no controlled speed ranking or
  cross-source speedup claim. The unified [timing workflow](timing.md) describes
  how to produce new five-method measurements with explicit resources.

The [reference manifest](../reported_results/manifest.json) records source and
released-data hashes. CAA midpoint accuracy, raw mean-difference robustness, and
optional CAA scalar-logistic + GCAV steering remain separate experiment identities.
Historical robustness pairing, steering trajectories and judge-score provenance
limitations
remain in [provenance.md](provenance.md); reusing artwork does not resolve them.
