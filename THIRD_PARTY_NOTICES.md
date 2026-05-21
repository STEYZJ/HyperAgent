# Third-Party Notices

This document records HyperAgent's known third-party open-source dependencies and reference projects. It is an engineering compliance aid, not legal advice.

## Project License Status

HyperAgent does not yet include a project-level `LICENSE` file. Before publishing a formal open-source release, choose and add a project license, then make sure this notice file and release notes stay consistent with that choice.

## Direct Runtime Dependencies

These packages are imported by HyperAgent source code or are required by the supported runtime path.

| Dependency | Current version | License | How HyperAgent uses it | Compliance note |
| --- | --- | --- | --- | --- |
| NumPy | 1.24.3 | BSD-3-Clause | Array operations, HSI cube preprocessing, metrics data handling. | Keep copyright/license notices when redistributing. |
| SciPy | 1.10.1 | BSD License | `.mat` I/O and scientific utilities. | Keep SciPy notices; binary distributions may include bundled BLAS/LAPACK/runtime notices. |
| scikit-learn | 1.3.2 | BSD License | SVM baseline, confusion matrix, kappa metrics, train/test helpers. | Keep copyright/license notices when redistributing. |
| Matplotlib | 3.7.5 | PSF-style License | Headless plots and report figures. | Keep copyright/license notices when redistributing. |
| PyYAML | 6.0.2 | MIT | YAML config, experiment plans, workspace config. | Include MIT notice in source/binary redistributions. |
| PyTorch | 2.3.1+cu118 | BSD License | MLP baseline and generated model factories. | Keep notices; review CUDA/NVIDIA terms if redistributing GPU-enabled binaries or images. |
| tifffile | 2023.7.10 | BSD License | Optional TIFF hyperspectral reader. | Keep copyright/license notices when redistributing. |

## Environment Snapshot and Transitive Dependencies

`environment.txt` is a conda package snapshot and contains many transitive packages, including CUDA/NVIDIA runtime libraries, MKL, FFmpeg, image codecs, and PyTorch CUDA packages. These are not all direct HyperAgent imports, but they matter if you distribute a binary environment, Docker image, VM image, or packaged installer.

Before distributing anything beyond source code:

- Regenerate the environment snapshot.
- Export the full transitive dependency list.
- Review CUDA/NVIDIA, MKL, FFmpeg/x264, image codec, and system runtime redistribution terms.
- Include upstream notices required by any packaged binary dependencies.

## Reference Projects

The following projects influenced HyperAgent design but are not vendored or imported by the tracked HyperAgent source tree:

| Reference | License status checked | HyperAgent status |
| --- | --- | --- |
| NousResearch Hermes Agent | MIT in local `参考/hermes-agent/LICENSE` and upstream README. | Design reference only; `hermes_plugin/` is a thin local adapter and does not import Hermes Agent code. |
| esengine DeepSeek-Reasonix | Public GitHub reference reviewed for runtime/profile ideas. | Design reference only; HyperAgent implements its own `deepseek_reasonix.py` profiles. |
| openclaw/openclaw | Public GitHub reference reviewed for CLI/agent workflow ideas. | Design reference only; no OpenClaw code is vendored or imported. |
| Claude Code tutorials/docs | Product behavior reference only. | HyperAgent uses independently implemented command/UI patterns; no Claude Code source is included. |

The local `参考/` directory is ignored by git and must stay out of releases unless a separate license review is completed.

## Maintenance Rules

- Update this file whenever adding a new direct dependency to `environment.yml`, `pyproject.toml`, or source imports.
- Do not copy third-party source code into HyperAgent without preserving license headers and adding a clear notice here.
- For source-only releases, permissive runtime dependencies currently appear low risk, but retain notices and avoid implying upstream endorsement.
- For packaged binary releases, Docker images, or hosted installers, perform a full transitive dependency license review.
