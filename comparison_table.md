<!--
========================================================================
 Tabla de comparación de features para psfScope (v3)
 Insertala en el paper (sección "Statement of Need") justo después
 del párrafo donde se mencionan PSFj y localize-psf.

 Cambios respecto a la v2:
   - PSFj ahora basado en lectura directa del paper Theer, Mongis,
     Knop 2014 (Nat Methods) y del manual oficial. La columna fue
     revisada celda por celda. Varias celdas cambiaron:
       * Sub-pixel-aligned averaged PSF: ◐ → ✓ (PSFj exporta
         averaged PSF para deconvolución).
       * Click-to-inspect: ✗ → ◐ (PSFj tiene "bead reports"
         seleccionables; ver nota 7).
       * Per-bead SNR: ✓ → ✗ (no figura en el CSV de PSFj).
       * Anisotropic voxels: ✗ → ◐ (calibrated voxels sí,
         pipeline anisotrópica end-to-end no).
       * Batch processing: ◐ → ✓ (PSFj soporta multi-stack).
       * FWHM-vs-depth: ◐ → ✗ (PSFj tiene z0 planarity, no
         FWHM-vs-depth).
   - Plataforma de PSFj corregida: es una app Java *standalone*
     que usa librerías de ImageJ + µManager, NO un plugin de
     ImageJ. El texto del paper también necesita corrección.
   - Agregada fila "Bead size correction" (PSFj corrige; psfScope
     no lo hace explícitamente — es un gap honesto).

 ATENCIÓN — bug en paper.bib:
   El paper actual cita PSFj como [@sage_2017]. Esa referencia
   corresponde a DeconvolutionLab2 (Sage et al., Methods 2017),
   no a PSFj. La cita correcta es Theer, Mongis & Knop 2014
   (Nat Methods, doi:10.1038/nmeth.3102). El bib entry está
   abajo. Reemplazar @sage_2017 por @theer_2014 en todas las
   menciones de PSFj.

 ATENCIÓN — texto del paper:
   El paper dice "PSFj is a widely used ImageJ plugin". No es
   un plugin: es una app Java standalone que corre afuera de
   ImageJ/Fiji. Sugerencia: "PSFj [@theer_2014] is a widely
   used standalone Java application built on ImageJ libraries..."
========================================================================
-->

`psfScope` complements existing bead-based PSF tools by targeting the
specific needs of OPM users — anisotropic voxels handled natively in
physical units throughout the entire pipeline (not only in the FWHM
fit), a simultaneous 3-D Gaussian fitting option, depth-dependent
aberration analysis, and quantitative voxel-wise comparison against
vectorial / scalar theoretical PSFs — all wrapped in a scriptable
pure-Python core with an integrated GUI. Table&nbsp;1 summarises the
main feature differences against the most widely used open-source
tools.

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
| Anisotropic voxels (dz ≠ dx) handled in physical units throughout      |     ✓      |         ◐¹         |              ◐¹             |                      ◐¹                      |
| Sub-pixel-aligned averaged PSF saved as deconvolution-ready TIFF       |     ✓      |         ✓          |              ✓              |                      ✗                       |
| Simultaneous 3-D Gaussian fit option (in addition to 1-D per axis)     |     ✓      |         ✗²         |              ✓              |                      ✗                       |
| Bead size deconvolution / finite-bead-size correction                  |     ✗³     |         ✓          |              ✓              |                      ✗                       |
| Per-bead spatial map of resolution across the FOV                      |     ✓      |         ✓          |              ✗              |                      ✗⁴                      |
| FWHM-vs-depth analysis (depth-dependent spherical aberration)          |     ✓      |         ✗⁵         |              ✗              |                      ✗                       |
| Per-bead lateral asymmetry / ellipticity                               |     ✓      |         ✓          |              ✗              |                      ◐⁶                      |
| Per-bead signal-to-noise ratio                                         |     ✓      |         ✗          |              ✗              |                      ✗⁷                      |
| Interactive click-to-inspect of individual beads                       |     ✓      |         ◐⁸         |              ✗              |                      ✗                       |
| Quantitative comparison with full 3-D theoretical PSF (MSE / NCC / r)  |     ✓      |         ✗⁹         |              ◐¹⁰            |                      ✗⁹                      |
| Per-bead tabular export for downstream statistical analysis            |     ✓      |         ✓          |              ✓              |                      ✓                       |
| Batch processing of multiple bead volumes                              |     ✓      |         ✓          |              ✓              |                      ✓                       |

**Notes.** ¹ All three tools accept calibrated voxels in physical
units, so FWHM is fitted in µm. However, bead-detection, ROI
extraction and lateral fits assume square pixels (dx = dy) or operate
in pixel space, which is appropriate for traditional widefield /
confocal but does not extend to OPM volumes where the deskewed dz
differs significantly from dx. `psfScope` formulates DoG filtering,
the ellipsoidal NMS footprint, and the isolation filter directly in
µm with per-axis scaling. ² PSFj uses a 2-D elliptic Gaussian fit in
the focal plane and an independent 1-D Gaussian fit along z; no joint
3-D minimisation. ³ PSFj corrects the measured FWHM for the finite
diameter of the bead (assuming a known nominal diameter). `psfScope`
currently assumes sub-diffraction beads; for ~200 nm beads at high NA
this introduces a small systematic positive bias on FWHM that should
be subtracted in post-processing, or addressed in a future release.
⁴ MetroloJ_QC's Batch PSF Profiler explicitly assumes a
translation-invariant PSF and aggregates FWHMs across the FOV without
preserving spatial coordinates. ⁵ PSFj reports a z₀ (best-focus axial
position) heatmap that is informative for stage tilt and field
flatness, but no plot of FWHM as a function of bead axial depth, so
depth-dependent spherical aberration cannot be diagnosed directly.
⁶ MetroloJ_QC reports a Lateral Asymmetry Ratio (LAR) per channel
from the *averaged* FWHM_x and FWHM_y, not per individual bead.
⁷ MetroloJ_QC reports a Signal-to-Background ratio computed as bead
intensity divided by mean intensity in a surrounding annulus; its
manual explicitly states this is **not** an SNR estimate. ⁸ PSFj's
"Bead inspection" panel allows the user to select beads and generate
a per-bead PDF report (lateral 2-D fit, axial 1-D fit, fit residuals
and parameters). `psfScope` instead opens a live Toplevel window with
ROI cross-sections and 1-D intensity profiles overlaid with Gaussian
fits, enabling more rapid visual quality control during analysis.
⁹ Theoretical comparison is restricted to scalar Born-Wolf-style FWHM
formulas (with several microscope geometries supported in MetroloJ_QC),
not voxel-wise comparison against a full 3-D theoretical PSF. ¹⁰
`localize-psf` includes Richards-Wolf-style models internally but does
not, by default, report MSE / NCC / Pearson similarity metrics against
the empirical PSF.

The combination of a scriptable Python core with a built-in
interactive GUI, an anisotropic OPM-native pipeline (DoG, detection
footprint and isolation operating in µm rather than pixels),
depth-dependent aberration diagnostics, and quantitative comparison
against vectorial / scalar theoretical PSFs is, to our knowledge, not
provided by any other openly available tool.

<!--
========================================================================
 Bib entries actualizadas para paper.bib

 NOTA: la cita actual @sage_2017 es INCORRECTA — corresponde a
 DeconvolutionLab2 (Sage et al., Methods 2017), no a PSFj. Hay que
 reemplazar @sage_2017 por @theer_2014 en TODAS las menciones de PSFj
 en el paper. Si querés conservar la cita a Sage et al. para algo
 (e.g., como ejemplo de tooling de deconvolución), está bien, pero
 NO es la referencia de PSFj.

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
     author  = {Faklaris, Orestis and Bancel-Vall{\'e}e, Leslie and others},
     title   = {Quality assessment in light microscopy for reproducible research},
     journal = {Journal of Cell Biology},
     year    = {2022},
     note    = {bioRxiv preprint: https://doi.org/10.1101/2021.06.16.448633.
                Companion publication of MetroloJ\_QC v1.3.1.1 (Oct.\ 2024).}
   }

========================================================================
-->
