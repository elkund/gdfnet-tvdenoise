# Data licenses and modifications

The distributed data artifacts in `data/generated/` and `data/calibration/` are
released under **CC BY-SA 4.0**. The Python code and the model weights in
`models/` are released under the MIT license in `LICENSE`.

## Sources

- **MedMNIST+** (PathMNIST, ChestMNIST, BloodMNIST) is distributed under
  **CC BY 4.0**. See Yang, Shi, and Ni, "MedMNIST Classification Decathlon: A
  Lightweight AutoML Benchmark for Medical Image Analysis", IEEE ISBI (2021);
  Yang et al., "MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D
  biomedical image classification", Scientific Data (2023); and the MedMNIST+
  documentation.
- **Medical Segmentation Decathlon Task03 Liver** (LiTS abdominal CT) is
  distributed under **CC BY-SA 4.0**. The source snapshot used here is the
  `MedOtter/msd-liver` mirror of the MSD Task03_Liver dataset. See Antonelli et
  al., "The Medical Segmentation Decathlon", Nature Communications (2022), and
  Bilic et al., "The Liver Tumor Segmentation Benchmark (LiTS)", Medical Image
  Analysis (2023).

## Modifications

The data artifacts were derived from the sources above with these modifications:

- Conversion of source images to single-channel grayscale.
- LiTS Hounsfield-unit windowing to [-160, 240] HU.
- Liver-centred square cropping with a 1.05 margin and scanner-air rejection.
- Downsampling to 64, 128, and 224 pixels by Lanczos resampling.
- Patient-disjoint assignment of LiTS volumes to train, validation, and test splits.
- Additive white Gaussian noise injection at sampled noise-to-signal ratios.
- Monte Carlo degrees-of-freedom labelling with the calibrated-budget Chambolle-Pock solver.
- Compression of the bundled data tensors with LZMA (`.pt.xz`).

## Redistribution

Source image archives are not redistributed. `data/archives/` and
`data/lits_raw/` are local, gitignored working directories.

## Attribution

If you use these artifacts, cite MedMNIST (v1 and v2), the Medical Segmentation
Decathlon, and the LiTS benchmark. Preserve the CC BY-SA 4.0 share-alike terms
for derived data.
