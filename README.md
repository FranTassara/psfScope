# psfScope

**psfScope** is a Python tool for experimental point spread function (PSF) characterisation in oblique plane microscopy (OPM) and light-sheet microscopy (LSM).

It estimates the system PSF directly from three-dimensional images of sub-diffraction fluorescent beads, and provides a graphical interface for interactive quality control and spatial analysis of PSF variation across the field of view.

## Features

- Automated bead detection with an anisotropic ellipsoidal footprint (handles dz ≠ dx without resampling)
- Quality filtering via **1-D sequential** or **simultaneous 3-D Gaussian fitting**
- Per-bead metrics: FWHM (Z, Y, X, XY), lateral ellipticity (σ_x − σ_y) / σ_xy, and SNR
- Sub-pixel alignment with NaN-masked averaging
- **Four-tab GUI** built on Python's standard `tkinter`:
  - **Estimation** — parameters, fitting mode, theoretical PSF panel, progress log
  - **PSF** — XY/XZ/YZ cross-sections with measured and theoretical FWHM readout
  - **Beads** — spatial scatter, FWHM histograms, FWHM vs depth plot, click-to-inspect viewer
  - **FOV Map** — spatial map of FWHM / ellipticity / SNR across the field of view
- CSV export of all per-bead measurements
- Born-Wolf theoretical PSF overlay (FWHM_xy = 0.51·λ/NA, FWHM_z = 0.887·λ/(n−√(n²−NA²)))

## Installation

```bash
pip install psfscope
```

Or from source:

```bash
git clone https://github.com/FranTassara/psfScope.git
cd psfScope
pip install -e .
```

### Requirements

`numpy`, `scipy`, `scikit-image`, `tifffile`, `matplotlib` (all standard scientific Python packages). The GUI uses Python's built-in `tkinter` — no additional GUI dependencies.

## Usage

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
    return_bead_data = True,
)
```

`bead_data` is a dictionary with per-bead positions, fitted sigmas, FWHM values, ellipticity, SNR, and selection masks. See the docstring of `estimate_psf_from_beads` for the full key reference.

### Generate synthetic test data

```bash
python generate_test_beads.py               # writes test_beads.tif
python generate_test_beads.py --n-beads 30 --noise 60
```

## Note on dz

The input volume must be **deskewed**. For a deskewed OPM volume:

```
dz = galvo_step_um × sin(tilt_deg)
```

Default system parameters (0.168 µm step, 41° tilt): dz ≈ 0.110 µm.

## Testing

```bash
pytest tests/
```

## Citation

If you use psfScope in your research, please cite:

> Tassara, F. J. & Gargiulo, J. (2026). *psfScope: a Python tool for experimental point spread function characterisation in oblique plane microscopy.* Journal of Open Source Software. [DOI pending]

## License

MIT
