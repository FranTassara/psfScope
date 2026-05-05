<!--
========================================================================
 Tabla de comparación de features para psfScope (v4)
 Insertala en el paper (sección "Statement of Need") justo después
 del párrafo donde se mencionan PSFj y localize-psf.

 Cambios respecto a la v3 (después de leer el código fuente real
 de qi2lab/localize-psf):

   CRÍTICO — La columna de localize-psf estaba siendo INFRA-vendida
   en la v3. Correcciones celda por celda:

     * Anisotropic voxels end-to-end: ◐ → ✓
       get_filter_kernel(sigmas, drs) y get_max_filter_footprint
       aceptan drs y sigmas por-eje. La pipeline completa
       (DoG, NMS footprint, fit) opera en µm con escalado
       por-eje. Eso anula uno de los aportes que el paper de
       psfScope reclama como propio. → IMPORTANTE para el paper.

     * Per-bead lateral asymmetry / ellipticity: ✗ → ◐
       gaussian3d_asymmetric_pixelated y _rotated_pixelated
       fittean sigma_x ≠ sigma_y ≠ sigma_z, pero el modelo
       default en autofit_psfs() es el simétrico. ◐ refleja
       que la capacidad existe pero no es default.

     * Bead size correction: ✓ → ✗ (no encontré ninguna
       referencia a bead diameter / size correction en el
       código). Era un error mío en la v3.

     * Per-bead spatial map of resolution: ✗ → ◐
       autofit_psfs llama a plot_bead_locations con
       weights=fit_params[to_keep, 4], que es un scatter
       sigma_xy vs posición sobre el max-projection. No es
       un heatmap interpolado como el de PSFj o el de
       psfScope, pero da info espacial.

     * GPU support: nueva fila. localize-psf tiene CuPy +
       Gpufit. psfScope no.

     * Best-fraction selection (filtrar por percentil de sigma):
       nueva fila. localize-psf tiene psf_percentiles. PSFj y
       MetroloJ_QC no. psfScope DEBERÍA tenerlo según el paper
       pero NO está implementado en el código actual — esto es
       un desfase paper↔código que ya marqué antes. Si vas a
       JOSS, o lo implementás (5 líneas) o lo sacás del paper.

   PSFj — confirmado contra paper Theer 2014 + manual:
     * platform corregida: standalone Java (NO ImageJ plugin)
     * cita corregida en el bib: @theer_2014, no @sage_2017

 Lo que sigue siendo único de psfScope:
   1. Built-in GUI + scriptable Python core (combo)
   2. Pure-Python sin runtime de Java
   3. FWHM-vs-axial-depth diagnostic
   4. Comparación voxel-wise con métricas MSE/NCC/Pearson
      contra PSF teórico 3-D
   5. Click-to-inspect en vivo durante el análisis

 Eso son cinco aportes reales. Es menos que los siete u ocho que
 daba a entender el paper original, pero es suficiente para JOSS.
 NO es suficiente para Nature Methods.
========================================================================
-->

`psfScope` complements existing bead-based PSF tools by combining,
in a single open-source application, several capabilities that are
otherwise scattered across different tools or absent altogether: a
scriptable pure-Python core with a built-in interactive GUI; a
depth-dependent FWHM diagnostic that exposes spherical-aberration
gradients along the optical axis; voxel-wise comparison of the
empirical PSF against vectorial / scalar theoretical models with
MSE, normalised cross-correlation and Pearson similarity metrics;
and click-to-inspect quality control on individual beads during the
analysis itself. Table&nbsp;1 summarises the main feature differences
against the most widely used open-source tools.

**Table 1.** Feature comparison between `psfScope` and the most
widely used open-source bead-based PSF tools. ✓ = supported,
✗ = not supported, ◐ = partial or restricted scope (see notes
below the table).

| Feature                                                                | `psfScope` | PSFj [@theer_2014] | localize-psf [@qi2lab_2023] | MetroloJ_QC [@matthews_2010; @faklaris_2022] |
|------------------------------------------------------------------------|:----------:|:------------------:|:---------------------------:|:--------------------------------------------:|
| Platform                                                               |   Python   |  Java standalone   |           Python            |                 Fiji (Java)                  |
| Pure-Python install (no Java / ImageJ runtime)                         |     ✓      |         ✗          |              ✓              |                      ✗                       |
| Scriptable programmatic API                                            |     ✓      |         ✗          |              ✓              |                      ✗                       |
| Built-in graphical user interface                                      |     ✓      |         ✓          |              ✗              |                      ✓                       |
| Anisotropic voxels (dz ≠ dx) handled in physical units throughout      |     ✓      |         ◐¹         |              ✓              |                      ◐¹                      |
| Sub-pixel-aligned averaged PSF saved as deconvolution-ready TIFF       |     ✓      |         ✓          |              ✓              |                      ✗                       |
| Simultaneous 3-D Gaussian fit (in addition to 1-D per axis)            |     ✓      |         ✗²         |              ✓              |                      ✗                       |
| Bead size deconvolution / finite-bead-size correction                  |     ✗³     |         ✓          |              ✗              |                      ✗                       |
| GPU acceleration (CUDA / CuPy)                                         |     ✗      |         ✗          |              ✓              |                      ✗                       |
| Best-fraction selection (rank by σ, keep sharpest fraction)            |     ✗⁴     |         ✗          |              ✓              |                      ✗                       |
| Per-bead spatial map of resolution across the FOV                      |     ✓      |         ✓          |              ◐⁵             |                      ✗⁶                      |
| FWHM-vs-axial-depth analysis                                           |     ✓      |         ✗⁷         |              ✗              |                      ✗                       |
| Per-bead lateral asymmetry / ellipticity                               |     ✓      |         ✓          |              ◐⁸             |                      ◐⁹                      |
| Per-bead signal-to-noise ratio                                         |     ✓      |         ✗          |              ✗              |                      ✗¹⁰                     |
| Interactive click-to-inspect of individual beads                       |     ✓      |         ◐¹¹        |              ✗              |                      ✗                       |
| Voxel-wise comparison vs full 3-D theoretical PSF (MSE / NCC / r)      |     ✓      |         ✗¹²        |              ◐¹³            |                      ✗¹²                     |
| Per-bead tabular export for downstream statistical analysis            |     ✓      |         ✓          |              ✓              |                      ✓                       |
| Batch processing of multiple bead volumes                              |     ✓      |         ✓          |              ✓              |                      ✓                       |

**Notes.** ¹ PSFj and MetroloJ_QC accept calibrated voxels in physical
units, so the FWHM is fitted in micrometres. However, bead detection
and lateral fitting assume `dx = dy` and operate in pixel space, which
limits applicability to OPM / light-sheet volumes whose deskewed `dz`
differs significantly from `dx`. ² PSFj uses an elliptic-Gaussian
2-D fit in the focal plane plus an independent 1-D Gaussian fit along
z; no joint 3-D minimisation is performed. ³ `psfScope` currently
assumes sub-diffraction beads. For ~200 nm beads at high NA this
introduces a small positive bias on the recovered FWHM that can be
removed in post-processing. ⁴ `localize-psf` exposes the
`psf_percentiles` parameter which keeps only beads in the smallest
σ percentile when forming the averaged PSF. `psfScope` reaches the
final bead set through a sequence of edge / isolation / fit-quality /
amplitude / R² / sanity filters but does not currently include a
percentile-based step on σ. ⁵ `localize-psf`'s `plot_bead_locations`
overlays per-bead σ values onto the maximum-intensity projection as a
weighted scatter; it does not compute an interpolated FOV heatmap.
⁶ MetroloJ_QC's Batch PSF Profiler explicitly assumes a
translation-invariant PSF and aggregates FWHMs across the FOV without
preserving spatial coordinates. ⁷ PSFj reports a `z₀` (best-focus
axial position) heatmap that diagnoses stage tilt and field flatness,
but no plot of FWHM as a function of bead axial depth, so
depth-dependent spherical aberration cannot be diagnosed directly.
⁸ `localize-psf` provides asymmetric and rotated 3-D Gaussian models
(`gaussian3d_asymmetric_pixelated`, `gaussian3d_asymmetric_rotated_pixelated`)
that recover separate σ_x, σ_y, σ_z, but the symmetric model is the
default in `autofit_psfs`. ⁹ MetroloJ_QC reports a Lateral Asymmetry
Ratio (LAR) per channel computed from the averaged FWHM_x and
FWHM_y, not per individual bead. ¹⁰ MetroloJ_QC reports a
Signal-to-Background ratio, computed as bead intensity divided by
mean intensity in a surrounding annulus; its manual explicitly
states this is not an SNR estimate. ¹¹ PSFj's "Bead inspection"
panel allows the user to select beads after the analysis and to
generate a per-bead PDF report. `psfScope` instead opens a live
Toplevel window with ROI cross-sections and 1-D intensity profiles
overlaid with Gaussian fits, enabling rapid visual quality control
during the analysis. ¹² Theoretical comparison is restricted to
analytical FWHM formulas (Born-Wolf style, with several microscope
geometries supported in MetroloJ_QC) rather than a voxel-wise
comparison against a full 3-D theoretical PSF. ¹³ `localize-psf`
includes Born-Wolf, Gibson-Lanni and vectorial PSF models (via
`psfmodels`) and *fits* them to the empirical PSF, but does not by
default report MSE, normalised cross-correlation or Pearson
similarity metrics against the empirical PSF — only fit residuals.

The combination of a pure-Python implementation with a built-in
interactive GUI, depth-dependent aberration diagnostics, voxel-wise
quantitative comparison against vectorial / scalar theoretical PSFs,
and live click-to-inspect quality control during analysis is, to our
knowledge, not provided by any other openly available tool.

<!--
========================================================================
 Bib entries actualizadas para paper.bib

 NOTA CRÍTICA: la cita actual @sage_2017 es INCORRECTA — corresponde a
 DeconvolutionLab2 (Sage et al., Methods 2017), no a PSFj. Reemplazar
 @sage_2017 por @theer_2014 en TODAS las menciones de PSFj en el paper.
 Si querés conservar la cita a Sage et al. para algo (e.g., como
 ejemplo de tooling de deconvolución), está bien, pero NO es la
 referencia de PSFj.

 Otra corrección al paper: el texto dice "PSFj is a widely used ImageJ
 plugin". No es plugin: es una app Java standalone que usa librerías
 de ImageJ + µManager. Sugerencia: "PSFj [@theer_2014] is a widely
 used standalone Java application built on ImageJ libraries..."

   @article{theer_2014,
     author  = {Theer, Patrick and Mongis, Cyril and Knop, Michael},
     title   = {{PSFj}: know your fluorescence microscope},
     journal = {Nature Methods},
     year    = {2014},
     volume  = {11},
     number  = {10},
     pages   = {981--982},
     doi     = {10.1038/nmeth.3102}
   }

   @inproceedings{matthews_2010,
     author    = {Matthews, C{\'e}dric and Cordeli{\`e}res, Fabrice P.},
     title     = {{MetroloJ}: an {ImageJ} plugin to help monitor microscopes' health},
     booktitle = {ImageJ User and Developer Conference Proceedings},
     year      = {2010}
   }

   @article{faklaris_2022,
     author  = {Faklaris, Orestis and Bancel-Vall{\'e}e, Leslie and
                Dauphin, Aur{\'e}lien and Monterroso, Baptiste and
                Fr{\`e}re, Perrine and Geny, David and Manoliu, Tudor and
                Rossi, Sylvain and Cordelières, Fabrice and
                Schapman, Damien and Nitschke, Roland and
                Cau, Julien and Guilbert, Thomas},
     title   = {Quality assessment in light microscopy for routine use
                through simple tools and robust metrics},
     journal = {Journal of Cell Biology},
     volume  = {221},
     year    = {2022},
     doi     = {10.1083/jcb.202107093},
   }

========================================================================
-->
