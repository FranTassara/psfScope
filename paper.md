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
    orcid: 0000-0000-0000-0000     # TODO: replace with real ORCID
    affiliation: 1
  - name: Julian Gargiulo
    orcid: 0000-0000-0000-0000     # TODO: replace with real ORCID
    affiliation: 1

affiliations:
  - name: INSTITUTION, COUNTRY     # TODO: replace with real institution
    index: 1

date: 6 May 2026
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

`psfScope` is a Python package that estimates the experimental PSF of an OPM
system directly from three-dimensional images of sub-diffraction fluorescent
beads. The package implements an automated pipeline: anisotropic band-pass
filtering, ellipsoidal local-maximum detection, per-bead quality assessment
via 1-D sequential or simultaneous 3-D Gaussian fitting, signal-to-noise
ratio (SNR) estimation, optional finite-bead-size correction, best-fraction
selection of the sharpest beads, sub-pixel alignment, and robust
NaN-masked averaging. A lightweight graphical user interface (GUI) built on
the Python standard library exposes the same pipeline interactively across
five tabs covering parameter entry, PSF cross-sections, per-bead
diagnostics, an interactive field-of-view (FOV) map, and an averaged-PSF
diagnostics panel. Results are exported as 32-bit floating-point TIFF
files compatible with deconvolution pipelines, as a per-bead CSV table,
and as a multi-page PDF report.

# Statement of Need

Accurate PSF characterisation is a prerequisite for quantitative fluorescence
microscopy. Theoretical PSF models based on scalar or vectorial diffraction
theory [@born_wolf_1999; @gibson_lanni_1992] assume ideal optical conditions
and a homogeneous refractive index medium. In practice, residual aberrations,
coverslip tilt, immersion-medium mismatch, and the oblique detection geometry
of OPM systems [@bouchard_2015; @millett_2019] cause the PSF to deviate
non-trivially from these models and to vary across the field of view (FOV)
and along the optical axis.

Empirical PSF estimation from sub-diffraction fluorescent beads is the
standard approach to characterising the actual imaging response
[@hanser_2004]. Several open-source tools address this task.
PSFj [@theer_2014] is a widely used standalone Java application built on
ImageJ libraries that reports per-bead FWHM measurements and interpolated
spatial maps, but its detection and lateral fitting pipeline assumes square
pixels and it provides no programmatic API.
MetroloJ\_QC [@matthews_2010; @faklaris_2022] is a Fiji plugin designed for
routine microscope quality control; it measures per-bead FWHMs across
several microscope geometries but does not export a deconvolution-ready
averaged PSF, does not preserve per-bead spatial coordinates in its FOV
analysis, and provides no programmatic API.
The `localize-psf` library [@qi2lab_2023] is a scriptable Python package
that handles anisotropic voxels natively, supports GPU acceleration, and
exports a sub-pixel-aligned averaged PSF, but is distributed without a
graphical interface and is primarily targeted at single-molecule
localisation workflows.

`psfScope` complements these tools by combining, in a single open-source
application, several capabilities that are otherwise scattered across
different tools or absent altogether: a scriptable pure-Python core paired
with a built-in interactive GUI; a depth-dependent FWHM diagnostic that
exposes spherical-aberration gradients along the optical axis; voxel-wise
comparison of the empirical PSF against vectorial and scalar theoretical
models with mean squared error, normalised cross-correlation and Pearson
similarity metrics; live click-to-inspect quality control on individual
beads during the analysis itself; and a multi-page PDF report for routine
documentation. Table 1 summarises the main feature differences against the
most widely used open-source tools.

**Table 1.** Feature comparison between `psfScope` and the most widely used
open-source bead-based PSF tools. ✓ = supported, ✗ = not supported,
◐ = partial or restricted scope (see notes below the table).

| Feature | `psfScope` | PSFj [@theer_2014] | localize-psf [@qi2lab_2023] | MetroloJ\_QC [@matthews_2010; @faklaris_2022] |
|---|:---:|:---:|:---:|:---:|
| Platform | Python | Java standalone | Python | Fiji (Java) |
| Pure-Python install (no Java / ImageJ runtime) | ✓ | ✗ | ✓ | ✗ |
| Scriptable programmatic API | ✓ | ✗ | ✓ | ✗ |
| Built-in graphical user interface | ✓ | ✓ | ✗ | ✓ |
| Anisotropic voxels (dz ≠ dx) handled in physical units throughout | ✓ | ◐¹ | ✓ | ◐¹ |
| Sub-pixel-aligned averaged PSF saved as deconvolution-ready TIFF | ✓ | ✓ | ✓ | ✗ |
| Simultaneous 3-D Gaussian fit option (in addition to 1-D per axis) | ✓ | ✗² | ✓ | ✗ |
| Bead-size deconvolution / finite-bead-size correction | ✓ | ✓ | ✗ | ✗ |
| GPU acceleration (CUDA / CuPy) | ✗ | ✗ | ✓ | ✗ |
| Best-fraction selection (rank by σ, keep sharpest subset) | ✓ | ✗ | ✓ | ✗ |
| Per-bead spatial map of resolution across the FOV | ✓ | ✓ | ◐³ | ✗⁴ |
| FWHM-vs-axial-depth analysis (depth-dependent spherical aberration) | ✓ | ✗⁵ | ✗ | ✗ |
| Per-bead lateral asymmetry / ellipticity | ✓ | ✓ | ◐⁶ | ◐⁷ |
| Per-bead signal-to-noise ratio | ✓ | ✗ | ✗ | ✗⁸ |
| Interactive click-to-inspect of individual beads | ✓ | ◐⁹ | ✗ | ✗ |
| Voxel-wise comparison vs full 3-D theoretical PSF (MSE / NCC / r) | ✓ | ✗¹⁰ | ◐¹¹ | ✗¹⁰ |
| Per-bead tabular export for downstream statistical analysis | ✓ | ✓ | ✓ | ✓ |
| Batch processing of multiple bead volumes | ✓ | ✓ | ✓ | ✓ |
| Multi-page PDF report | ✓ | ✓ | ✗ | ✓ |

**Notes.**
¹ PSFj and MetroloJ\_QC accept calibrated voxels in physical units so the
FWHM is fitted in micrometres, but bead detection and lateral fitting
assume dx = dy and operate in pixel space, which limits applicability to
OPM volumes whose deskewed dz differs significantly from dx.
² PSFj uses an elliptic-Gaussian 2-D fit in the focal plane plus an
independent 1-D Gaussian fit along z; no joint 3-D minimisation is performed.
³ `localize-psf`'s `plot_bead_locations` overlays per-bead σ values onto
the maximum-intensity projection as a weighted scatter; it does not compute
an interpolated resolution heatmap.
⁴ MetroloJ\_QC's Batch PSF Profiler assumes a translation-invariant PSF and
aggregates FWHMs across the FOV without preserving individual bead spatial
coordinates.
⁵ PSFj reports a z₀ (best-focus axial position) heatmap useful for
diagnosing stage tilt and field flatness, but does not plot FWHM as a
function of bead axial depth, so depth-dependent spherical aberration
cannot be diagnosed directly.
⁶ `localize-psf` provides asymmetric and rotated 3-D Gaussian models that
recover separate σ\_x, σ\_y, σ\_z, but the symmetric model is the default
in `autofit_psfs`.
⁷ MetroloJ\_QC reports a Lateral Asymmetry Ratio per channel computed from
the *averaged* FWHM\_x and FWHM\_y, not per individual bead.
⁸ MetroloJ\_QC reports a Signal-to-Background ratio (bead intensity divided
by mean intensity in a surrounding annulus), which its documentation
explicitly distinguishes from an SNR estimate.
⁹ PSFj's bead inspection panel allows the user to select beads after the
analysis and generate a per-bead PDF report. `psfScope` instead opens a
live window with ROI cross-sections and 1-D intensity profiles overlaid
with Gaussian fits, enabling rapid visual quality control during the
analysis.
¹⁰ Theoretical comparison is restricted to analytical FWHM formulas
(Born-Wolf style) rather than voxel-wise comparison against a full 3-D
theoretical PSF.
¹¹ `localize-psf` includes Born-Wolf, Gibson-Lanni, and vectorial PSF models
and fits them to the empirical PSF, but does not by default report MSE,
normalised cross-correlation, or Pearson similarity metrics.

The combination of a pure-Python implementation with a built-in interactive
GUI, depth-dependent aberration diagnostics, voxel-wise quantitative
comparison against vectorial and scalar theoretical PSFs, and live
click-to-inspect quality control during analysis is, to our knowledge, not
provided by any other openly available tool.

# Implementation

## Algorithm

The PSF estimation pipeline operates on a deskewed ZYX volume of
sub-diffraction fluorescent beads and proceeds in eight steps (Figure 1).

**Step 1 — Anisotropic band-pass filtering.**
A Difference-of-Gaussians (DoG) filter suppresses both slow background
illumination non-uniformity and high-frequency shot noise, while selectively
enhancing objects of bead-like size. The two Gaussian kernels are
constructed in physical units and applied with per-axis sigmas scaled by
dz and dx respectively, ensuring that the filter is isotropic in µm space
rather than in pixel space.

**Step 2 — Candidate detection and isolation.**
Local maxima are detected on the DoG image using
`skimage.feature.peak_local_max` [@scikit_image] with an ellipsoidal
exclusion footprint whose axes are set by the minimum bead-separation
distance in µm, converted to per-axis pixel counts. This is critical for
OPM data where the axial footprint in pixels is systematically smaller
than the lateral footprint. A threshold (automatic or user-supplied)
removes dim candidates that are likely to be noise peaks. As a stricter
isolation step applied after detection, candidates whose nearest neighbour
is closer than `min_sep_um` in physical units are discarded; this removes
beads whose PSFs may optically overlap even if they were detected as
separate peaks. The isolation filter is implemented with
`scipy.spatial.cKDTree` [@scipy], yielding O(N log N) complexity. Each
axis is scaled by its voxel size before tree construction so a Euclidean
query on the scaled coordinates is equivalent to the original anisotropic
distance metric.

**Step 3 — ROI extraction.**
A user-defined ROI is extracted around each candidate. Candidates whose
ROI would extend beyond the volume boundary are discarded. Local background
is estimated as the 5th percentile of the ROI and subtracted before fitting.

**Step 4 — Quality filtering.**
Two fitting modes are available, selected by the user.

*1-D sequential mode (default)*: independent 1-D Gaussian profiles are
fitted along the Z, Y, and X axes through the intensity maximum using
`scipy.optimize.curve_fit` [@scipy] with physiologically plausible sigma
bounds enforced as hard constraints. The accepted fits then pass a
cascaded quality filter: (i) all three fits must converge; (ii) the
goodness-of-fit coefficient of determination R² must exceed a user-defined
threshold (default 0.9), rejecting beads with non-Gaussian PSF shapes
caused by doublets, debris, or insufficient SNR; (iii) a sanity check
requires that the Gaussian centre is within `max_offset_px` pixels of the
intensity peak, the background is non-negative and below the ROI maximum,
and no sigma is within 5 % of its upper bound (which would indicate the
optimiser hit the constraint wall rather than converging to a physical
minimum); and (iv) an amplitude outlier filter based on the Hampel
identifier discards photobleached or saturated beads whose fitted amplitude
deviates more than 3 × 1.4826 × MAD from the median amplitude of the
surviving set. Running the amplitude filter last ensures that the
distribution on which the MAD criterion operates consists only of beads
with reliable, physically plausible fits.

*3-D simultaneous mode*: a full anisotropic 3-D Gaussian is fitted to the
entire ROI volume by minimising the residual sum of squares over all
voxels. The 8-parameter model
I(z, y, x) = A · exp(−((z−c_z)²/2σ_z² + (y−c_y)²/2σ_y² + (x−c_x)²/2σ_x²)) + bg
is initialised with sigma values from the 1-D empirical half-maximum widths
and centroid coordinates (c_z₀, c_y₀, c_x₀) from a 3-D extension of the
radial-symmetry algorithm of @parthasarathy_2012, which gives a sub-pixel
initial estimate without requiring a separate iterative fit, reducing the
risk of the optimiser converging to a local minimum compared to seeding
from the integer-resolution intensity maximum. The optimisation is
performed with `scipy.optimize.curve_fit` [@scipy] and accelerated by
supplying an analytical Jacobian of the 3-D Gaussian model, reducing the
number of function evaluations required for convergence. The same
cascaded quality filter as in 1-D mode is applied. This mode captures PSF
asymmetries that sequential 1-D fits may miss, at the cost of approximately
10–100× longer processing time per bead.

**Step 4b — SNR estimation.**
For each accepted bead, the signal-to-noise ratio is computed as the ratio
of the peak intensity (background already subtracted in Step 3) to the
standard deviation of the ROI outer shell — the first and last voxel layer
along each axis. These border voxels are far from the bead centre and
sample the local background noise. The SNR is stored as a per-bead
attribute and exported in the CSV table and FOV map, allowing beads
acquired under low-signal conditions to be identified and optionally
excluded from downstream analysis.

**Step 4c — Finite-bead-size correction (optional).**
When the user supplies a non-zero `bead_diameter_nm`, the per-bead and
averaged-PSF FWHMs are corrected for the finite size of the bead by
deconvolving the bead diameter from the measured FWHM in quadrature:
FWHM_corrected = √(max(FWHM_measured² − D_bead², 0)) [@cole_2011]. Both
raw and corrected values are stored in the per-bead data dictionary, so
that downstream statistical analysis can use either depending on context.
When `bead_diameter_nm = 0` (default) the correction is skipped and
backward compatibility is preserved.

**Step 4d — Parallel processing.**
The per-bead loop (Steps 3–4c) can be parallelised across multiple CPU
threads via the `n_jobs` parameter of `estimate_psf_from_beads`. Each bead
is processed as an independent task using Python's standard
`concurrent.futures.ThreadPoolExecutor`. NumPy array operations and SciPy
optimisation release the Global Interpreter Lock, so threads can execute
concurrently on multiple cores without additional dependencies. The
practical benefit depends strongly on the fitting mode: in 3-D mode, where
per-bead processing time is dominated by iterative Gaussian optimisation
over all ROI voxels, multi-core speedup is noticeable; in 1-D mode,
per-bead wall time is of the order of milliseconds and threading overhead
is likely to offset most of the gain. The default `n_jobs = 1` preserves
sequential behaviour.

**Step 5 — Best-fraction selection.**
The accepted beads are ranked in ascending order by their mean lateral
sigma σ_xy = (σ_y + σ_x) / 2, and only the sharpest fraction (controlled
by the `best_fraction` parameter, default 1.0 = keep all) is retained for
averaging. This step discards beads that were defocused or in an aberrated
region of the sample during acquisition, following the approach of
@qi2lab_2023. Beads excluded at this step are flagged separately from
quality-rejected beads in `bead_data`, allowing the GUI to colour them
distinctly in the bead-map view.

**Step 6 — Sub-pixel alignment and averaging.**
Each retained ROI is shifted by the sub-pixel offset of its fitted Gaussian
centre from the geometric ROI centre using `scipy.ndimage.shift` with cubic
interpolation. Border pixels created by the shift are assigned NaN. The
aligned ROIs are combined with `numpy.nanmean` [@numpy], so that border
NaN values are excluded from the average without introducing a
zero-padding bias.

**Step 7 — Normalisation and export.**
The averaged PSF is normalised to unit sum and saved as a 32-bit
floating-point TIFF with ImageJ-compatible metadata. It can be loaded
directly by deconvolution software that accepts a sampled PSF kernel. The
FWHM of the averaged PSF is measured along each axis from 1-D
cubic-spline-interpolated profiles through the peak voxel, with the
half-maximum crossing located by linear interpolation between adjacent
oversampled points; no Gaussian shape is assumed.

**Step 8 — Theoretical PSF comparison (optional).**
When `compare_theoretical = True`, `psfmodels` [@psfmodels] is called to
generate a theoretical PSF on the same ZYX voxel grid, using the
user-supplied numerical aperture, emission wavelength, and immersion
refractive index. Both vectorial (Richards-Wolf) and scalar (Gibson-Lanni)
models are supported. The theoretical volume is normalised to unit sum
and compared to the empirical PSF by computing three metrics: (i) mean
squared error, measuring pixel-wise intensity deviation; (ii) normalised
cross-correlation, equivalent to the cosine similarity of the two
flattened intensity vectors and invariant to global scaling; and (iii)
Pearson correlation coefficient, which captures linear association
independently of mean and variance. All three metrics, together with the
theoretical PSF array itself, are returned in the `bead_data` dictionary
for downstream analysis or visualisation.

## Graphical User Interface

The GUI is implemented with the Python standard-library `tkinter` module
and `matplotlib` [@matplotlib] embedded via `FigureCanvasTkAgg`. It is
organised into five tabs.

**Estimation** — file selection (single file or folder batch), all
algorithm parameters (pixel sizes, threshold, minimum bead separation,
ROI dimensions, R² threshold, fitting mode, `best_fraction`,
`bead_diameter_nm`), an optional *Theoretical PSF* panel for entering the
emission wavelength λ, numerical aperture NA, and refractive index n to
compute Born-Wolf reference FWHMs, a real-time progress bar driven by a
thread-safe queue callback, and a scrollable log. Tooltips on every
parameter Entry document the typical range. A *Reset defaults* button
restores all parameters to their initial values, and an *Export PDF
report* button produces a multi-page PDF containing the parameters used,
filter-funnel counts, PSF cross-sections, per-bead FWHM histograms and
the FOV map.

**PSF** — false-colour cross-sections (XY, XZ, YZ) of the estimated PSF
with crosshair cursors at the central plane. Two text lines are displayed
below the figure: a blue line with the measured FWHM values derived from
the cubic-spline profiles described in Step 7 (and bead-size-corrected
when applicable), and an orange line with the theoretical Born-Wolf FWHMs
(visible only when the theoretical overlay is enabled).

**Beads** — a spatial scatter plot on the left panel shows all detected
bead candidates colour-coded by outcome: border-rejected (light gray ×),
quality-rejected (salmon ×), accepted but excluded by `best_fraction`
(steel blue ○), and used in PSF (coloured by σ_xy, viridis_r). Clicking
any bead with the left mouse button opens a *Bead Inspector* Toplevel
window that shows the three ROI cross-section images and three 1-D
intensity profiles with Gaussian fit overlays for that individual bead,
together with all fitted sigma and FWHM values, the ellipticity, and the
SNR. The ROI for each accepted bead is cached in `bead_data`, so the
inspector remains functional even after the source TIFF has been moved
or renamed. The right panel contains three stacked subplots: FWHM_xy and
FWHM_z histograms (green for beads used in the PSF, steel blue for
accepted-but-not-used), each with an optional dashed orange reference
line at the theoretical FWHM, and a scatter of FWHM_xy as a function of
bead axial position, which reveals depth-dependent spherical aberration
when beads span a range of depths.

**FOV Map** — a scatter plot of bead positions in the sample plane (XY
projection), coloured by a user-selected metric: FWHM_xy, FWHM_z, FWHM_x,
FWHM_y, PSF ellipticity (σ_x − σ_y) / σ_xy, or SNR. Colour limits are set
to the 5th–95th percentile of the chosen metric to suppress outlier
distortion; ellipticity uses a symmetric diverging colormap centred at
the median. Global minimum and maximum beads are annotated directly on
the scatter, and the plot title encodes the full range and spread, making
spatial patterns of resolution degradation and optical aberration
immediately visible. The FOV map can be exported as a 300-dpi PNG file.

**FWHM diagnostics** — a 2 × 3 grid of subplots showing the 1-D intensity
profiles of the averaged PSF (Z, Y, X, top row) with cubic-spline fits
and half-maximum crossings annotated, and per-bead FWHM histograms (Z, Y,
X, bottom row) with Gaussian fits to the modal region of each
distribution. The histogram fits use a Poisson-weighted least-squares
Gaussian, reporting the fitted modal FWHM and its width as a concise
summary of PSF resolution across the bead ensemble.

A per-bead CSV table is exported from the Beads tab. Each row records the
bead position (pixels and µm), classification status, fitted sigma and
FWHM values for all three axes (σ_z, σ_y, σ_x, σ_xy), bead-size-corrected
FWHM values when applicable, PSF ellipticity, and SNR. In batch mode
(multiple input volumes), two additional columns are included: a
zero-based `volume_id` integer and the `source_file` basename, preserving
full provenance for downstream statistical analysis or filtering in
external tools.

## Testing

The package includes a `pytest` test suite (46 tests, organised into
synthetic-bead generation fixtures and per-feature unit tests) covering:

- normalisation invariants of the returned averaged PSF (float32,
  non-negative, unit sum, peak in the central region);
- correctness of the filter-funnel counts (`n_total`, `n_edge`,
  `n_isolation`, `n_fit_ok`, `n_accepted`, `n_used`);
- recovery of the ground-truth Gaussian sigma values within 30 % on
  synthetic data (both `fit_mode='1d'` and `fit_mode='3d'`);
- behaviour of `best_fraction` on a synthetic dataset with two
  populations of differing σ;
- correctness of the bead-size deconvolution formula on a synthetic
  Gaussian convolved with a sphere of known diameter;
- regression test for the early-return path in `_quality_check_3d`;
- progress-callback monotonicity and termination at 1.0;
- presence of all expected keys in `bead_data` (including `accepted_snr`,
  `accepted_ellipticity`, `accepted_rois`, `accepted_fwhm_corrected_*`,
  `psf_theoretical`, `psf_mse`, `psf_ncc`, `psf_pearson_r`) with
  consistent shapes and `None` placeholders when
  `compare_theoretical=False`;
- backward-compatible 2-tuple return when `return_bead_data=False`;
- rejection of 2-D inputs with a clear `ValueError`;
- equivalence of the new `cKDTree`-based isolation filter to the
  reference O(N²) implementation on a fixed-seed dataset.

Tests are run from the package root with `pytest tests/`.

# Acknowledgements

The authors thank [COLLEAGUES, FACILITY STAFF — TODO]. This work was
supported by [FUNDING SOURCES — TODO].

# References
