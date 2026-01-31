# Ensemble_THERMOCAD

This repository contains preprocessing scripts and the Rodriguez-Guerrero Breast Thermography dataset (included in `data/`).

Important notes:
- The full `data/` folder is included by request. This can make the repository large — Git LFS is configured for `.jpg`, `.jpeg`, and `.png` files.
- If pushing to GitHub fails due to large files or missing Git LFS on the remote/client, install and enable Git LFS locally: `git lfs install`.

Contents:
- `*.py` preprocessing and analysis scripts
- `data/` original dataset (benign/malignant folders)

If you prefer data not to be tracked by Git directly, we can instead move it to Git LFS-only or ignore it and provide a download script.
