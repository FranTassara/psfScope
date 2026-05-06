# psfScope

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**psfScope** is a Python tool for experimental point spread function (PSF)
characterisation in oblique plane microscopy (OPM) and light-sheet
microscopy (LSM).

It estimates the system PSF directly from three-dimensional images of
sub-diffraction fluorescent beads, and provides a graphical interface for
interactive quality control and spatial analysis of PSF variation across
the field of view.

## Features

- Automated bead detection with an anisotropic ellipsoidal footprint —
  handles `dz ≠ dx` without resampling.
- Fast isolation filtering via `scipy.spatial.cKDTree`
  (O(N log N), >60× faster than exhaustive pairwise search).
- Quality filtering via **1-D sequential** or **simultaneous 3-D
  Gaussian** fitting with analytical Jacobian.
- **Best-fraction selection**: keep only the sharpest N % of accepted
  beads for PSF averaging (`best_fraction`).
- **Finite-bead-size correction**: deconvolve the known bead diameter
  from the measured FWHM (`bead_diameter_nm`).
- Per-bead metrics: FWHM (Z, Y, X, XY), lateral ellipticity
  `(σ_x − σ_y) / σ_xy`, and SNR.
- Sub-pixel alignment with NaN-masked averaging.
- **Five-tab GUI** built on Python's standard `tkinter`:
  - **Estimation** — parameters, fitting mode, theoretical PSF panel,
    progress log, *Reset defaults* and *Export PDF report* buttons.
  - **PSF** — XY/XZ/YZ cross-sections with measured and theoretical
    FWHM readout (raw and bead-size-corrected when applicable).
  - **Beads** — spatial scatter, FWHM histograms, FWHM-vs-depth
    diagnostic, click-to-inspect bead viewer (cached ROIs — works
    after the source TIFF is moved or renamed).
  - **FOV Map** — spatial map of FWHM / ellipticity / SNR with
    percentile colour scaling and 300-dpi PNG export.
  - **FWHM diagnostics** — averaged-PSF profiles and per-bead FWHM
    histograms with Gaussian fits to the distribution mode.
- **Multi-page PDF report** export (cover with parameters and bead
  counts, PSF cross-sections, FWHM histograms, FOV map).
- CSV export of all per-bead measurements with full batch-mode
  provenance (`volume_id`, `source_file`).
- Born-Wolf theoretical PSF overlay
  (FWHM_xy = 0.51·λ/NA, FWHM_z = 0.887·λ/(n−√(n²−NA²))).
- Optional voxel-wise comparison against a vectorial (Richards-Wolf)
  or scalar (Gibson-Lanni) theoretical 3-D PSF, reporting MSE,
  normalised cross-correlation and Pearson similarity metrics.
- Batch mode: process multiple TIFFs and merge results.
- Parallel processing with `--n-jobs` (CLI) / worker threads (GUI).

## Installation

```bash
pip install psfscope
```

For the optional voxel-wise theoretical PSF comparison feature:

```bash
pip install "psfscope[theory]"
```

Or from source:

```bash
git clone https://github.com/FranTassara/psfScope.git
cd psfScope
pip install -e .
```

### Requirements

`numpy`, `scipy`, `scikit-image`, `tifffile`, `matplotlib` — all standard
scientific Python packages. The GUI uses Python's built-in `tkinter`,
so no additional GUI dependencies are required. The optional
`[theory]` extra installs `psfmodels` for full 3-D theoretical PSF
generation; everything else (including the Born-Wolf analytical
overlay) works without it.

## Quick start

### GUI

```bash
psfscope
```

Or from Python:

```python
from psf_gui import launch_gui
launch_gui()
```

### Programmatic API

```python
from postprocess_psf import estimate_psf_from_beads

psf, psf_path, bead_data = estimate_psf_from_beads(
    tif_path         = "beads_deskew_488.tif",
    dx               = 0.127,   # lateral pixel size [µm]
    dz               = 0.110,   # axial voxel size, deskewed [µm]
    best_fraction    = 0.7,     # keep the sharpest 70 % of beads
    bead_diameter_nm = 100.0,   # correct for 100 nm bead diameter
    return_bead_data = True,
)
```

`bead_data` is a dictionary with per-bead positions, fitted sigmas, FWHM
values (raw and bead-size-corrected), ellipticity, SNR, cached ROIs,
and selection masks. See the docstring of `estimate_psf_from_beads` for
the full key reference.

Selected `bead_data` keys:

| Key | Description |
|-----|-------------|
| `accepted_px` | (N, 3) pixel positions of all quality-filtered beads |
| `accepted_sigma_{z,y,x}` | Fitted Gaussian σ in µm |
| `accepted_fwhm_corrected_{z,y,x}` | Bead-size corrected FWHM in nm (when `bead_diameter_nm > 0`) |
| `accepted_used` | Boolean mask — True for beads used in PSF averaging |
| `accepted_not_used_best_fraction` | Boolean mask — True for beads excluded by `best_fraction` |
| `accepted_rois` | Cached 3-D ROI arrays (avoids re-reading the TIFF on click-to-inspect) |
| `accepted_snr` | Per-bead signal-to-noise ratio |
| `accepted_ellipticity` | Per-bead `(σ_x − σ_y) / σ_xy` |
| `psf_theoretical`, `psf_mse`, `psf_ncc`, `psf_pearson_r` | Voxel-wise comparison results (when `compare_theoretical=True`) |
| `n_total`, `n_edge`, `n_isolation`, `n_fit_ok`, `n_accepted`, `n_used` | Filter-funnel counts |

### FWHM measurement from an averaged PSF

```python
from postprocess_psf import measure_fwhm_from_averaged_psf

result = measure_fwhm_from_averaged_psf(
    psf_3d           = psf,
    voxel_size_nm    = (110.0, 127.0, 127.0),  # (dz, dy, dx) in nm
    bead_diameter_nm = 100.0,
)
# result keys:
#   fwhm_z_nm,           fwhm_y_nm,           fwhm_x_nm
#   fwhm_z_nm_corrected, fwhm_y_nm_corrected, fwhm_x_nm_corrected
```

### CLI

```bash
psfscope beads_deskew.tif \
    --dx 0.127 --dz 0.110 \
    --best-fraction 0.7 \
    --bead-diameter 100
```

For the full option list:

```bash
psfscope --help
```

### Generate synthetic test data

```bash
python generate_test_beads.py                         # writes test_beads.tif
python generate_test_beads.py --n-beads 30 --noise 60
```

## Note on `dz`

The input volume must be **deskewed**. For a deskewed OPM volume:

```
dz = galvo_step_um × sin(tilt_deg)
```

For a typical OPM with 0.168 µm galvo step and 41° tilt: `dz ≈ 0.110 µm`.

## Comparison with related tools

| Tool | Platform | GUI | Python API | OPM-native | Voxel-wise theory comparison |
|---|---|:---:|:---:|:---:|:---:|
| **psfScope** | Python | ✓ | ✓ | ✓ | ✓ |
| PSFj | Java standalone | ✓ | ✗ | ◐ | ✗ |
| MetroloJ_QC | Fiji plugin | ✓ | ✗ | ◐ | ✗ |
| localize-psf | Python library | ✗ | ✓ | ✓ | ◐ |

The combination of a built-in interactive GUI, a scriptable Python
core, depth-dependent FWHM diagnostics, and voxel-wise quantitative
comparison against vectorial / scalar theoretical PSFs is, to our
knowledge, not provided together by any other openly available tool.
A detailed feature-by-feature comparison is available in the
[paper](paper.md).

## Testing

```bash
pytest tests/
```

The test suite contains 46 tests organised as synthetic-bead generation
fixtures and per-feature unit tests, covering: PSF normalisation
invariants; filter-funnel counts; Gaussian fitting accuracy in 1-D and
3-D modes; `best_fraction` selection; bead-size correction formula;
the `_quality_check_3d` early-return regression; backward-compatible
2-tuple return; rejection of 2-D inputs; equivalence of the
`cKDTree`-based isolation filter against the reference O(N²)
implementation.

## Citation

If you use psfScope in your research, please cite:

> Tassara, F. J. & Gargiulo, J. (2026). psfScope: a Python tool for
> experimental point spread function characterisation in oblique plane
> microscopy. *Journal of Open Source Software*. DOI: pending.

## License

[MIT](LICENSE).
