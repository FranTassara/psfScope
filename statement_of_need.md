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
[@hanser_2004]. Several open-source tools exist for this purpose.
PSFj [@theer_2014] is a widely used standalone Java application that reports
FWHM measurements and per-bead spatial maps, but its detection and fitting
pipeline assumes square pixels and lacks a programmatic API, limiting its
integration into automated workflows.
MetroloJ\_QC [@matthews_2010; @faklaris_2022] is a Fiji plugin designed for
routine microscope quality control that measures per-bead FWHMs, but it
assumes isotropic voxels, does not export a deconvolution-ready PSF kernel,
and provides no scriptable API.
The `localize-psf` library [@qi2lab_2023] is a Python package designed for
single-molecule localisation microscopy that includes bead-based PSF
estimation and a scriptable API, but it targets isotropic data and requires
a heavier installation with GPU-optional dependencies.
None of these tools natively handles the anisotropic voxel geometry of OPM
data throughout the entire pipeline — from detection and ROI extraction to
fitting — nor do they combine a spatial map of PSF variability across the
field of view with a scriptable pure-Python core and an integrated GUI.

`psfScope` complements these tools by combining, in a single open-source
application, several capabilities that are otherwise scattered across
different tools or absent altogether: a scriptable pure-Python core with a
built-in interactive GUI; a depth-dependent FWHM diagnostic that exposes
spherical-aberration gradients along the optical axis; voxel-wise comparison
of the empirical PSF against vectorial and scalar theoretical models with MSE,
normalised cross-correlation and Pearson similarity metrics; and
click-to-inspect quality control on individual beads during the analysis
itself. Table 1 summarises the main feature differences.

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
| Bead size deconvolution / finite-bead-size correction | ✗³ | ✓ | ✗ | ✗ |
| GPU acceleration (CUDA / CuPy) | ✗ | ✗ | ✓ | ✗ |
| Best-fraction selection (rank by σ, keep sharpest subset) | ✗⁴ | ✗ | ✓ | ✗ |
| Per-bead spatial map of resolution across the FOV | ✓ | ✓ | ◐⁵ | ✗⁶ |
| FWHM-vs-axial-depth analysis (depth-dependent spherical aberration) | ✓ | ✗⁷ | ✗ | ✗ |
| Per-bead lateral asymmetry / ellipticity | ✓ | ✓ | ◐⁸ | ◐⁹ |
| Per-bead signal-to-noise ratio | ✓ | ✗ | ✗ | ✗¹⁰ |
| Interactive click-to-inspect of individual beads | ✓ | ◐¹¹ | ✗ | ✗ |
| Voxel-wise comparison vs full 3-D theoretical PSF (MSE / NCC / r) | ✓ | ✗¹² | ◐¹³ | ✗¹² |
| Per-bead tabular export for downstream statistical analysis | ✓ | ✓ | ✓ | ✓ |
| Batch processing of multiple bead volumes | ✓ | ✓ | ✓ | ✓ |

**Notes.**
¹ PSFj and MetroloJ\_QC accept calibrated voxels in physical units so the
FWHM is fitted in µm, but bead detection and lateral fitting assume dx = dy
and operate in pixel space, which limits applicability to OPM volumes whose
deskewed dz differs significantly from dx.
² PSFj uses an elliptic-Gaussian 2-D fit in the focal plane plus an
independent 1-D Gaussian fit along z; no joint 3-D minimisation is performed.
³ `psfScope` currently assumes sub-diffraction beads. For ~200 nm beads at
high NA this introduces a small positive bias on the recovered FWHM that can
be removed in post-processing.
⁴ `localize-psf` exposes the `psf_percentiles` parameter, which keeps only
beads in the smallest σ percentile when forming the averaged PSF. `psfScope`
reaches the final bead set through a sequence of edge, isolation, fit-quality,
amplitude, R², and sanity filters but does not include a percentile-based
selection step on σ.
⁵ `localize-psf`'s `plot_bead_locations` overlays per-bead σ values onto the
maximum-intensity projection as a weighted scatter; it does not compute an
interpolated resolution heatmap.
⁶ MetroloJ\_QC's Batch PSF Profiler assumes a translation-invariant PSF and
aggregates FWHMs across the FOV without preserving individual bead spatial
coordinates.
⁷ PSFj reports a z₀ (best-focus axial position) heatmap useful for diagnosing
stage tilt and field flatness, but does not plot FWHM as a function of bead
axial depth; depth-dependent spherical aberration therefore cannot be
diagnosed directly.
⁸ `localize-psf` provides asymmetric and rotated 3-D Gaussian models that
recover separate σ\_x, σ\_y, σ\_z, but the symmetric model is the default
in `autofit_psfs`.
⁹ MetroloJ\_QC reports a Lateral Asymmetry Ratio per channel computed from
the *averaged* FWHM\_x and FWHM\_y, not per individual bead.
¹⁰ MetroloJ\_QC reports a Signal-to-Background ratio (bead intensity divided
by mean intensity in a surrounding annulus), which its documentation explicitly
distinguishes from an SNR estimate.
¹¹ PSFj's bead inspection panel allows the user to select beads after the
analysis and generate a per-bead PDF report. `psfScope` instead opens a live
window with ROI cross-sections and 1-D intensity profiles overlaid with
Gaussian fits, enabling rapid visual quality control during the analysis.
¹² Theoretical comparison is restricted to analytical FWHM formulas
(Born-Wolf style) rather than voxel-wise comparison against a full 3-D
theoretical PSF.
¹³ `localize-psf` includes Born-Wolf, Gibson-Lanni, and vectorial PSF models
and fits them to the empirical PSF, but does not by default report MSE,
normalised cross-correlation, or Pearson similarity metrics.

The combination of a pure-Python implementation with a built-in interactive
GUI, depth-dependent aberration diagnostics, voxel-wise quantitative
comparison against vectorial and scalar theoretical PSFs, and live
click-to-inspect quality control during analysis is, to our knowledge, not
provided by any other openly available tool.

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
