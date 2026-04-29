---
title: 'psfScope: a Python tool for experimental point spread function
  characterisation in oblique plane microscopy'

tags:
  - Python
  - microscopy
  - point spread function
  - oblique plane microscopy
  - light-sheet microscopy
  - deconvolution
  - fluorescence imaging

authors:
  - name: Francisco Joaquin Tassara
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Julian Gargiulo
    orcid: 0000-0000-0000-0000
    affiliation: 1

affiliations:
  - name: INSTITUTION, COUNTRY
    index: 1

date: 11 March 2026
bibliography: paper.bib
---

# Summary

The point spread function (PSF) of a fluorescence microscope describes how
the system spreads the image of a point-like emitter, and is the fundamental
quantity that governs both image quality and the performance of computational
post-processing steps such as deconvolution. In oblique plane microscopy
(OPM), an inherently anisotropic geometry produces axial voxel sizes that
differ significantly from lateral pixel sizes, and field-dependent optical
aberrations mean that a single theoretical PSF is often an inadequate model
of the true imaging system.

`psfScope` is a Python package that estimates the experimental PSF of
an OPM system directly from three-dimensional images of sub-diffraction
fluorescent beads. The package implements an automated pipeline: anisotropic
band-pass filtering, ellipsoidal local-maximum detection, per-bead quality
assessment via 1-D sequential or simultaneous 3-D Gaussian fitting, signal-to-noise
ratio (SNR) estimation, sub-pixel alignment, and robust NaN-masked averaging.
A lightweight graphical user interface (GUI) built on the Python standard
library provides interactive parameter tuning, live progress feedback, and
four dedicated visualisation tabs. These tabs provide: false-colour PSF
cross-sections with non-Gaussian FWHM readout and an optional Born-Wolf
reference overlay; a per-bead scatter map colour-coded by σ_xy with an
interactive click-to-inspect bead viewer; a field-of-view map of resolution
and ellipticity across the sample plane; and a FWHM diagnostics panel showing
averaged-PSF profiles alongside per-bead FWHM histograms with Gaussian
distribution fits. The tool runs entirely
in Python, requires only widely available scientific packages, and exports
results as standard 32-bit TIFF files ready for deconvolution pipelines.

# Statement of Need

Accurate PSF characterisation is a prerequisite for quantitative fluorescence
microscopy. Theoretical PSF models based on scalar or vectorial diffraction
theory [@born_wolf_1999; @gibson_lanni_1992] assume ideal optical conditions
and a homogeneous refractive index medium. In practice, residual aberrations,
coverslip tilt, immersion medium mismatch, and the oblique detection geometry
of OPM systems [@bouchard_2015; @millett_2019] cause the PSF to deviate
non-trivially from these models and to vary across the field of view.

Empirical PSF estimation from fluorescent beads smaller than the diffraction
limit is the standard approach to characterising the actual imaging PSF
[@hanser_2004]. Several tools exist for this purpose.
PSFj [@sage_2017] is a widely used ImageJ plugin, but it assumes isotropic
voxels and lacks a programmatic API, limiting its integration into automated
pipelines. The `localize-psf` library [@qi2lab_2023] is a Python package
designed for single-molecule localisation microscopy that includes bead-based
PSF estimation, but it targets isotropic data and requires a heavier
installation with GPU-optional dependencies. Neither tool provides a visual
map of PSF spatial variability across the microscope field of view.

`psfScope` was designed to fill this gap for OPM practitioners:

1. **Anisotropic geometry**: all detection and fitting steps are formulated
   in physical units (µm), correctly accounting for dz ≠ dx without
   resampling the data.
2. **Dual fitting modes**: the user can choose between fast sequential 1-D
   Gaussian fits along each axis or a simultaneous 3-D Gaussian fit to the
   full ROI volume. The 3-D mode is more accurate for laterally asymmetric PSFs
   while the 1-D mode is well-suited for rapid screening of large bead datasets.
3. **Spatial quality statistics**: beads are retained as individual measurements
   with their spatial coordinates, enabling a scatter map of FWHM, PSF
   ellipticity, and SNR as a function of sample-plane position. A dedicated
   FWHM-versus-depth scatter plot reveals depth-dependent spherical aberration
   that is invisible to tools reporting only a single aggregate PSF.
4. **Theoretical PSF comparison**: `psfScope` provides two complementary levels
   of theoretical reference. In the GUI, Born-Wolf analytical FWHMs
   (FWHM_xy = 0.51 · λ / NA, FWHM_z = 0.887 · λ / (n − √(n² − NA²)))
   are overlaid on all measurement plots as a quick visual benchmark.
   At the programmatic level, the `compare_theoretical` option calls the
   `psfmodels` package [@psfmodels] to generate a full vectorial (Richards-Wolf)
   or scalar (Gibson-Lanni [@gibson_lanni_1992]) PSF on the same voxel grid
   as the empirical PSF. Three quantitative similarity metrics are then reported:
   mean squared error (MSE), normalised cross-correlation (NCC), and Pearson
   correlation coefficient, enabling objective assessment of how closely the
   experimental PSF matches diffraction theory.
5. **Interactive bead inspector**: clicking any bead in the spatial scatter
   plot opens a per-bead Toplevel window showing the three ROI cross-sections
   and 1-D intensity profiles with Gaussian fit overlays, together with all
   fitted parameters, enabling rapid visual quality control.
6. **Zero-overhead installation**: the GUI uses Python's built-in `tkinter`
   module; the core dependencies are `numpy`, `scipy`, `scikit-image`,
   `tifffile`, and `matplotlib` — all part of the standard scientific Python
   stack. The optional `psfmodels` package (installable as
   `pip install psfscope[theory]`) is only required for quantitative
   theoretical PSF comparison and is not needed for routine use.
7. **Pipeline integration**: the core function `estimate_psf_from_beads`
   returns a normalised 3-D array (sum = 1) compatible with the
   Richardson–Lucy deconvolution implementation in the accompanying
   `postprocess_deconvolution` module.

# Implementation

## Algorithm

The PSF estimation pipeline operates on a deskewed ZYX volume of
sub-diffraction fluorescent beads and proceeds in seven steps
(Figure 1).

**Step 1 — Anisotropic band-pass filtering.**
A Difference-of-Gaussians (DoG) filter suppresses both slow background
illumination non-uniformity and high-frequency shot noise, while selectively
enhancing objects of bead-like size. The two Gaussian kernels are constructed
in physical units and applied with per-axis sigmas scaled by dz and dx
respectively, ensuring that the filter is isotropic in µm space rather than
in pixel space.

**Step 2 — Candidate detection.**
Local maxima are detected on the DoG image using `skimage.feature.peak_local_max`
[@scikit_image] with an ellipsoidal exclusion footprint whose axes are set by
the minimum bead-separation distance in µm, converted to per-axis pixel counts.
This is critical for OPM data where the axial footprint in pixels is
systematically smaller than the lateral footprint. A threshold (automatic or
user-supplied) removes dim candidates that are likely to be noise peaks. As a
second isolation step applied after detection, candidates whose nearest
neighbour is closer than `min_sep_um` in physical units are discarded; this
stricter criterion removes beads whose PSFs may optically overlap even if they
were detected as separate peaks, since overlapping PSFs produce systematic
biases in the Gaussian fit.

**Step 3 — ROI extraction.**
A user-defined ROI is extracted around each candidate. Candidates whose ROI
would extend beyond the volume boundary are discarded. Local background is
estimated as the 5th percentile of the ROI and subtracted before fitting.

**Step 4 — Quality filtering.**
Two fitting modes are available, selected by the user:

*1-D sequential mode (default)*: independent 1-D Gaussian profiles are fitted
along the Z, Y, and X axes through the intensity maximum using
`scipy.optimize.curve_fit` [@scipy] with physiologically plausible sigma bounds
enforced as hard constraints. The accepted fits then pass a cascaded quality filter: (i) all three fits must
converge; (ii) the goodness-of-fit coefficient of determination R² must exceed
a user-defined threshold (default 0.9), rejecting beads with non-Gaussian PSF
shapes caused by doublets, debris, or insufficient SNR; (iii) a sanity check
requires that the Gaussian centre is within `max_offset_px` pixels of the
intensity peak, the background is non-negative and below the ROI maximum, and
no sigma is within 5 % of its upper bound (which would indicate the optimiser
hit the constraint wall rather than converging to a physical minimum); and
(iv) an amplitude outlier filter based on the Hampel identifier discards
photobleached or saturated beads whose fitted amplitude deviates more than
3 × 1.4826 × MAD from the median amplitude of the surviving set. Running the
amplitude filter last ensures that the distribution on which the MAD criterion
operates consists only of beads with reliable, physically plausible fits.

*3-D simultaneous mode*: a full anisotropic 3-D Gaussian is fitted to the
entire ROI volume by minimising the residual sum of squares over all voxels.
The 8-parameter model
I(z, y, x) = A · exp(−((z−c_z)²/2σ_z² + (y−c_y)²/2σ_y² + (x−c_x)²/2σ_x²)) + bg
is initialised with sigma values from the 1-D empirical half-maximum widths and
centroid coordinates (c_z₀, c_y₀, c_x₀) from the radial symmetry algorithm of
@parthasarathy_2012, which provides sub-pixel accuracy without iterative fitting.
The optimisation is performed with `scipy.optimize.curve_fit` [@scipy] and
accelerated by supplying an analytical Jacobian of the 3-D Gaussian model,
reducing the number of function evaluations required for convergence. The same cascaded quality filter (sanity check, amplitude outlier removal, and
R² threshold) is applied as in 1-D mode. This mode captures PSF asymmetries
that sequential 1-D fits may miss, at the cost of approximately 10–100×
longer processing time per bead.

**Step 4b — SNR estimation.**
For each accepted bead, the signal-to-noise ratio is computed as the ratio of
the peak intensity (background already subtracted in Step 3) to the standard
deviation of the ROI outer shell — the first and last voxel layer along each
axis. These border voxels are far from the bead centre and sample the local
background noise. The SNR is stored as a per-bead attribute and exported in
the CSV table and FOV map, allowing beads acquired under low-signal conditions
to be identified and optionally excluded from downstream analysis.

**Step 4c — Parallel processing.**
The per-bead loop (Steps 3–4b) can be parallelised across multiple CPU threads
via the `n_jobs` parameter of `estimate_psf_from_beads`. Each bead is processed
as an independent task using Python's standard `concurrent.futures.ThreadPoolExecutor`.
NumPy array operations and SciPy optimisation release the Global Interpreter
Lock (GIL), so threads can execute concurrently on multiple cores without
additional dependencies. The practical benefit depends strongly on the fitting
mode: in 3-D mode, where per-bead processing time is dominated by iterative
Gaussian optimisation over all ROI voxels, multi-core speedup is noticeable; in
1-D mode, per-bead wall time is of the order of milliseconds and threading
overhead is likely to offset most of the gain. The default `n_jobs=1` preserves
sequential behaviour; increasing `n_jobs` is therefore recommended primarily
when using `fit_mode='3d'` on large bead datasets.

**Step 5 — Sub-pixel alignment and averaging.**
Each retained ROI is shifted by the sub-pixel offset of its fitted Gaussian
centre from the geometric ROI centre using `scipy.ndimage.shift` with cubic
interpolation. Border pixels created by the shift are assigned NaN. The
aligned ROIs are combined with `numpy.nanmean` [@numpy], so that border NaN
values are excluded from the average without introducing a zero-padding bias.

**Step 6 — Normalisation and export.**
The averaged PSF is normalised to unit sum and saved as a 32-bit floating-point
TIFF with ImageJ-compatible metadata. It can be loaded directly by deconvolution
software that accepts a sampled PSF kernel.

**Step 7 — Theoretical PSF comparison (optional).**
When `compare_theoretical=True`, `psfmodels` [@psfmodels] is called to generate
a theoretical PSF on the same ZYX voxel grid, using the user-supplied numerical
aperture, emission wavelength, and immersion refractive index. Both vectorial
(Richards-Wolf) and scalar (Gibson-Lanni) models are supported. The theoretical
volume is normalised to unit sum and compared to the empirical PSF by computing
three metrics: (i) mean squared error (MSE), measuring pixel-wise intensity
deviation; (ii) normalised cross-correlation (NCC), equivalent to the cosine
similarity of the two flattened intensity vectors and invariant to global scaling;
and (iii) Pearson correlation coefficient, which captures linear association
independently of mean and variance. All three metrics, together with the
theoretical PSF array itself, are returned in the `bead_data` dictionary for
downstream analysis or visualisation.

## Graphical User Interface

The GUI is implemented with the Python standard library `tkinter` module and
`matplotlib` [@matplotlib] embedded via `FigureCanvasTkAgg`. It is organised
into five tabs:

**Estimation** — file selection (single file or folder batch), all algorithm
parameters (pixel sizes, threshold, minimum bead separation, ROI dimensions,
R² threshold, fitting mode), an optional *Theoretical PSF* panel for entering
the emission wavelength λ, numerical aperture NA, and refractive index n to
compute Born-Wolf reference FWHMs, a real-time progress bar driven by a
thread-safe queue callback, and a scrollable log.

**PSF** — false-colour cross-sections (XY, XZ, YZ) of the estimated PSF with
crosshair cursors at the central plane. Two text lines are displayed below the
figure: a blue line with the measured FWHM values derived from 1-D cubic-spline
profiles through the peak voxel of the averaged PSF, with the half-maximum
crossing found by linear interpolation between adjacent oversampled points
(10× resolution; no Gaussian shape is assumed), and an orange line with the
theoretical Born-Wolf FWHMs (visible only when the theoretical overlay is
enabled).

**Beads** — a spatial scatter plot on the left panel shows all detected bead
candidates colour-coded by outcome: border-rejected (light gray ×),
quality-rejected (salmon ×), accepted but not used in the final PSF (steel
blue ○), and used in PSF (coloured by σ_xy, viridis_r). Clicking any bead
with the left mouse button opens a *Bead Inspector* Toplevel window that shows
the three ROI cross-section images and three 1-D intensity profiles with
Gaussian fit overlays for that individual bead, together with all fitted
sigma and FWHM values, the ellipticity, and the SNR. The right panel contains
three stacked subplots: FWHM_xy and FWHM_z histograms (green for beads used
in the PSF, steel blue for accepted-but-not-used), each with an optional
dashed orange reference line at the theoretical FWHM; and a scatter of
FWHM_xy as a function of bead axial position, which reveals depth-dependent
spherical aberration when beads span a range of depths.

**FOV Map** — a scatter plot of bead positions in the sample plane (XY
projection), coloured by a user-selected metric: FWHM_xy, FWHM_z, FWHM_x,
FWHM_y, PSF ellipticity (σ_x − σ_y) / σ_xy, or SNR. FWHM metrics use a
red-yellow-green diverging colormap (RdYlGn_r, low is green = good); ellipticity
uses a symmetric diverging colormap (RdBu_r) centred at zero; SNR uses the
plasma colormap (high is bright = good). Global minimum and maximum beads are
annotated directly on the scatter, and the plot title encodes the full range
and spread, making spatial patterns of resolution degradation and optical
aberration immediately visible.

**FWHM diagnostics** — a 2×3 grid of subplots showing the 1-D intensity
profiles of the averaged PSF (Z, Y, X, top row) with cubic-spline fits and
half-maximum crossings annotated, and per-bead FWHM histograms (Z, Y, X,
bottom row) with Gaussian fits to the distribution peak. The histogram fits
use a Poisson-weighted least-squares Gaussian to the modal region of each
distribution, reporting the fitted modal FWHM and its width as a concise
summary of PSF resolution across the bead ensemble.

A per-bead CSV table is exported from the Beads tab. Each row records the
bead position (pixels and µm), classification status, fitted sigma and FWHM
values for all three axes (σ_z, σ_y, σ_x, σ_xy), PSF ellipticity, and SNR.
In batch mode (multiple input volumes), two additional columns are included:
a zero-based `volume_id` integer and the `source_file` basename, preserving
full provenance for downstream statistical analysis or filtering in external
tools.

## Testing

The package includes a test suite that generates synthetic 3-D bead volumes
(Gaussians of known sigma embedded in Gaussian noise) using `numpy` and
`tifffile` [@tifffile], and verifies that:

- the returned PSF is float32, non-negative, and has unit sum;
- at least one valid bead is detected from a volume containing well-separated
  synthetic beads;
- `bead_data` contains all expected keys — including `accepted_snr`,
  `accepted_ellipticity`, and the theoretical comparison keys
  (`psf_theoretical`, `psf_mse`, `psf_ncc`, `psf_pearson_r`) — with
  consistent array shapes and `None` values when `compare_theoretical=False`;
- the estimated sigma values are within 30% of the ground truth;
- the `progress_callback` is called with fractions in [0, 1] ending at 1.0;
- the function returns `(psf, save_path)` when `return_bead_data=False`
  (backward compatibility);
- the PSF peak lies in the central region of the output volume;
- a 2-D input raises `ValueError`; and
- `fit_mode='3d'` produces a valid PSF consistent with the 1-D mode on the
  same synthetic data.

Tests are run with `pytest` and can be invoked from the package root with
`pytest tests/`.

# Acknowledgements

[ACKNOWLEDGEMENTS — funding, beamtime, colleagues]

# References
