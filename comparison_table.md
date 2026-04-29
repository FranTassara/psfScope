<!--
========================================================================
 Tabla de comparación de features para psfScope
 Insertala en el paper (sección "Statement of Need") justo después
 del párrafo donde se mencionan PSFj y localize-psf.

 NOTA — VERIFICAR antes de mandar:
   Las marcas para PSFj, localize-psf y MetroloJ_QC son las que mejor
   pude reconstruir, pero corroboralas en las versiones actuales de
   cada herramienta. Algunas que conviene chequear:
     - localize-psf: ¿maneja dz ≠ dx en unidades físicas, o asume
       voxels isotrópicos? Su target es SMLM, donde típicamente el
       voxel es isotrópico tras desenfoque.
     - PSFj: produce FOV maps (eso es seguro), pero ¿incluye
       per-bead SNR y elipticidad en el reporte que exporta?
     - MetroloJ_QC: orientado a QC rutinario; no recuerdo que
       genere un FOV map per-bead; sí reporta FWHM por bead.
     - El uso de ◐ está pensado para "parcial / depende de la versión".
========================================================================
-->

`psfScope` complements existing bead-based PSF tools by targeting the
specific needs of OPM users — anisotropic voxels handled natively in
physical units, spatial maps of resolution across the FOV, and a
scriptable pure-Python core with an integrated GUI. Table&nbsp;1 summarises
the main feature differences.

**Table 1.** Feature comparison between `psfScope` and the most widely
used open-source bead-based PSF tools. ✓ = supported, ✗ = not supported,
◐ = partial or dependent on configuration.

| Feature                                                       | `psfScope` |   PSFj [@sage_2017]   | localize-psf [@qi2lab_2023] |     MetroloJ_QC      |
|---------------------------------------------------------------|:----------:|:---------------------:|:---------------------------:|:--------------------:|
| Platform                                                      |   Python   |     ImageJ (Java)     |           Python            |      Fiji (Java)     |
| Open-source license                                           |     ✓      |           ✓           |              ✓              |          ✓           |
| Pure-Python install (no Java/ImageJ runtime)                  |     ✓      |           ✗           |              ✓              |          ✗           |
| Scriptable programmatic API                                   |     ✓      |           ✗           |              ✓              |          ✗           |
| Built-in graphical user interface                             |     ✓      |           ✓           |              ✗              |          ✓           |
| Anisotropic voxels (dz ≠ dx) handled in physical units        |     ✓      |           ✗           |              ◐              |          ✗           |
| Sub-pixel-aligned averaged PSF, deconvolution-ready TIFF      |     ✓      |           ✓           |              ✓              |          ✗           |
| Simultaneous 3-D Gaussian fit option (in addition to 1-D)     |     ✓      |           ✗           |              ✓              |          ✗           |
| Per-bead spatial map across the field of view                 |     ✓      |           ✓           |              ✗              |          ✗           |
| FWHM-vs-depth analysis (depth-dependent aberrations)          |     ✓      |           ✗           |              ✗              |          ✗           |
| Per-bead ellipticity and SNR exported                         |     ✓      |           ◐           |              ✗              |          ◐           |
| Interactive click-to-inspect of individual beads              |     ✓      |           ✗           |              ✗              |          ✗           |
| Quantitative comparison with vectorial / scalar theoretical PSF |   ✓      |           ✗           |              ◐              |          ✗           |
| Per-bead CSV export for downstream statistical analysis       |     ✓      |           ◐           |              ✓              |          ✓           |
| Batch processing of multiple bead volumes                     |     ✓      |           ◐           |              ✓              |          ◐           |

The combination of a scriptable Python core, a built-in interactive
GUI, anisotropic OPM-native geometry handling, spatial FOV diagnostics,
and quantitative comparison against vectorial and scalar theoretical
PSFs is, to our knowledge, not provided by any other openly available
tool.

<!--
========================================================================
 Bib entries que probablemente ya tenés:
   - @sage_2017       (PSFj)
   - @qi2lab_2023     (localize-psf)

 Bib entry adicional que necesitás agregar a paper.bib para MetroloJ_QC
 (cambialo si querés citar la versión más reciente):

   @article{metrolojqc_2022,
     author  = {Matthews, Cyril and Cordeli{\`e}res, Fabrice P.},
     title   = {{MetroloJ}: an {ImageJ} plugin to help monitor microscopes' health},
     journal = {Methods},
     year    = {2022},
     note    = {See also https://imagejdocu.list.lu/plugin/analysis/metroloj/start}
   }

 (Existe también la versión "MetroloJ_QC" mantenida por MontpellierRIO,
  https://github.com/MontpellierRessourcesImagerie/MetroloJ_QC ; usá la
  cita que mejor refleje la versión actual.)
========================================================================
-->
