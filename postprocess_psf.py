"""
Experimental PSF estimation from sub-diffraction fluorescent bead images.

Pipeline
--------
1. Anisotropic band-pass filter (Difference of Gaussians, DoG)
2. Local maximum detection with an ellipsoidal footprint (respects dz ≠ dx)
3. 3-D ROI extraction around each candidate
4. Quality filter: 1-D sequential Gaussian fits (default) or simultaneous 3-D fit
   - rejects beads outside sigma range, off-centre, or near the volume edge
   - returns the sub-pixel centre offset from the Gaussian fit
5. Percentile selection: keep the best X% by lateral sigma (sharpest beads)
6. Sub-pixel centring using the Gaussian offset + NaN-mean averaging
   (border pixels set to NaN do not contaminate the average)
7. Save as a normalised float32 TIFF (sum = 1)

Design notes
------------
- Anisotropic DoG and ellipsoidal footprint correctly handle oblique plane
  microscopy (OPM) data where the axial voxel size dz differs from dx.
- Sub-pixel alignment via 1-D Gaussian fitting (inspired by QI2lab/localize-psf)
  is more robust than phase cross-correlation for sparse bead images.
- NaN masking in ndi.shift + np.nanmean avoids border artefacts without
  zero-padding bias.
- best_fraction retains the sharpest subset, removing beads that were
  out-of-focus or aberrated during acquisition.

The resulting PSF can be used directly in postprocess_deconvolution.py as an
alternative to a theoretical PSF, without requiring external software (PSFj,
etc.).

Usage (Python)
--------------
    from postprocess_psf import estimate_psf_from_beads

    psf, psf_path, bead_data = estimate_psf_from_beads(
        tif_path         = "beads_deskew_488_0.tif",
        dx               = 0.127,   # lateral pixel size [µm]
        dz               = 0.110,   # axial voxel size, deskewed [µm]
        return_bead_data = True,
    )

Usage (CLI)
-----------
    python postprocess_psf.py beads_deskew.tif --dx 0.127 --dz 0.110

Note on dz
----------
The input volume must already be deskewed. For a deskewed OPM volume:
    dz = galvo_step_um × sin(tilt_deg)
For the default system parameters (0.168 µm step, 41° tilt):
    dz ≈ 0.168 × sin(41°) ≈ 0.110 µm
"""

import os
import argparse

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from tifffile import imread, imwrite


# =============================================================================
# Internal helpers
# =============================================================================

def _dog_filter(volume, sigma_small_um, sigma_large_um, dx, dz):
    """
    Anisotropic Difference-of-Gaussians (DoG) band-pass filter.

    Enhances point-like objects (beads) while suppressing slow background
    variations and high-frequency noise.  ndi.gaussian_filter uses 'reflect'
    boundary mode by default, which is appropriate here.

    Parameters
    ----------
    volume : ndarray (float32)
    sigma_small_um, sigma_large_um : float  — DoG sigmas in µm
    dx : float  — lateral pixel size in µm
    dz : float  — axial voxel size in µm
    """
    sz_s  = sigma_small_um / dz
    sxy_s = sigma_small_um / dx
    sz_l  = sigma_large_um / dz
    sxy_l = sigma_large_um / dx

    vol = volume.astype(np.float32)
    lo  = ndi.gaussian_filter(vol, sigma=(sz_s,  sxy_s, sxy_s))
    hi  = ndi.gaussian_filter(vol, sigma=(sz_l,  sxy_l, sxy_l))
    return lo - hi


def _fit_gaussian1d(coords, profile):
    """
    Fit A·exp(-(x-c)²/(2σ²)) + bg to a 1-D intensity profile.

    The initial sigma estimate uses the empirical FWHM of the profile.

    Returns
    -------
    popt : tuple (A, c, sigma, bg), or None if the fit fails.
    """
    bg0 = float(np.percentile(profile, 10))
    A0  = float(np.max(profile)) - bg0
    if A0 <= 0:
        return None

    c0 = float(coords[np.argmax(profile)])

    half  = bg0 + 0.5 * A0
    above = coords[profile > half]
    s0    = float((above[-1] - above[0]) / 2.355) if len(above) > 1 else float(coords[1] - coords[0])
    s0    = max(s0, float(coords[1] - coords[0]) * 0.5)

    try:
        popt, _ = curve_fit(
            lambda x, A, c, s, bg: A * np.exp(-(x - c) ** 2 / (2 * s ** 2)) + bg,
            coords,
            profile.astype(float),
            p0=[A0, c0, s0, bg0],
            bounds=([0, coords[0], 1e-6, 0], [np.inf, coords[-1], np.inf, np.inf]),
            maxfev=2000,
        )
        return popt  # (A, c, sigma, bg)
    except Exception:
        return None


def _quality_check_1d(roi, dx, dz, sigma_xy_bounds, sigma_z_bounds, max_center_offset_px):
    """
    Assess bead quality via 1-D Gaussian fits along Z, Y, and X.

    In addition to pass/fail, returns the sub-pixel offset of the fitted
    Gaussian centre from the geometric ROI centre.  This offset is used
    to sub-pixel-align the bead before averaging.

    Returns
    -------
    passes : bool
    sigma_z : float or None  [µm]
    sigma_y : float or None  [µm]
    sigma_x : float or None  [µm]
    offset_px : tuple (oz, oy, ox) in pixels, or None
        Shift needed to move the Gaussian centre to the ROI centre.
        Apply as ndi.shift(roi, -offset_px).
    """
    nz, ny, nx = roi.shape
    iz, iy, ix = np.unravel_index(np.argmax(roi), roi.shape)

    z_coords = np.arange(nz, dtype=float) * dz
    y_coords = np.arange(ny, dtype=float) * dx
    x_coords = np.arange(nx, dtype=float) * dx

    fz = _fit_gaussian1d(z_coords, roi[:, iy, ix])
    fy = _fit_gaussian1d(y_coords, roi[iz, :, ix])
    fx = _fit_gaussian1d(x_coords, roi[iz, iy, :])

    if fz is None or fy is None or fx is None:
        return False, None, None, None, None

    sz = abs(fz[2])
    sy = abs(fy[2])
    sx = abs(fx[2])

    if not (sigma_z_bounds[0]  <= sz <= sigma_z_bounds[1]):
        return False, sz, sy, sx, None
    if not (sigma_xy_bounds[0] <= sy <= sigma_xy_bounds[1]):
        return False, sz, sy, sx, None
    if not (sigma_xy_bounds[0] <= sx <= sigma_xy_bounds[1]):
        return False, sz, sy, sx, None

    # Offset of the Gaussian centre from the ROI geometric centre
    cz_fit, cy_fit, cx_fit = fz[1], fy[1], fx[1]
    roi_cz = (nz // 2) * dz
    roi_cy = (ny // 2) * dx
    roi_cx = (nx // 2) * dx

    if abs(cz_fit - roi_cz) > max_center_offset_px * dz:
        return False, sz, sy, sx, None
    if abs(cy_fit - roi_cy) > max_center_offset_px * dx:
        return False, sz, sy, sx, None
    if abs(cx_fit - roi_cx) > max_center_offset_px * dx:
        return False, sz, sy, sx, None

    offset_px = (
        (cz_fit - roi_cz) / dz,
        (cy_fit - roi_cy) / dx,
        (cx_fit - roi_cx) / dx,
    )
    return True, sz, sy, sx, offset_px


def _quality_check_3d(roi, dx, dz, sigma_xy_bounds, sigma_z_bounds, max_center_offset_px):
    """
    Assess bead quality via a simultaneous 3-D Gaussian fit.

    Fits  I(z,y,x) = A·exp(-((z-cz)²/2σz² + (y-cy)²/2σy² + (x-cx)²/2σx²)) + bg
    to the entire ROI volume in physical-unit (µm) coordinates.

    Initial parameter guesses are derived from the intensity-peak position and
    the empirical half-maximum width of each 1-D profile, providing a good
    starting point while keeping the optimisation tractable.

    Returns
    -------
    Same signature as _quality_check_1d:
    (passes, sigma_z, sigma_y, sigma_x, offset_px)
    """
    nz, ny, nx = roi.shape
    iz, iy, ix = np.unravel_index(np.argmax(roi), roi.shape)

    z_coords = np.arange(nz, dtype=float) * dz
    y_coords = np.arange(ny, dtype=float) * dx
    x_coords = np.arange(nx, dtype=float) * dx

    # Initial guesses
    bg0 = float(np.percentile(roi, 10))
    A0  = float(np.max(roi)) - bg0
    if A0 <= 0:
        return False, None, None, None, None

    cz0, cy0, cx0 = z_coords[iz], y_coords[iy], x_coords[ix]

    def _s0(coords, profile):
        half  = bg0 + 0.5 * A0
        above = coords[profile > half]
        return (float((above[-1] - above[0]) / 2.355) if len(above) > 1
                else float(coords[1] - coords[0]))

    sz0 = max(_s0(z_coords, roi[:, iy, ix]), float(z_coords[1]) * 0.5)
    sy0 = max(_s0(y_coords, roi[iz, :, ix]), float(x_coords[1]) * 0.5)
    sx0 = max(_s0(x_coords, roi[iz, iy, :]), float(x_coords[1]) * 0.5)

    # Build flattened coordinate arrays and data vector
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    coords_flat = (zz.ravel(), yy.ravel(), xx.ravel())
    data_flat   = roi.ravel().astype(float)

    def _gauss3d(coords, A, cz, cy, cx, sz, sy, sx, bg):
        z, y, x = coords
        return (A * np.exp(
            -((z - cz) ** 2 / (2 * sz ** 2) +
              (y - cy) ** 2 / (2 * sy ** 2) +
              (x - cx) ** 2 / (2 * sx ** 2))
        ) + bg)

    try:
        popt, _ = curve_fit(
            _gauss3d,
            coords_flat,
            data_flat,
            p0=[A0, cz0, cy0, cx0, sz0, sy0, sx0, bg0],
            bounds=(
                [0,      z_coords[0],  y_coords[0],  x_coords[0],  1e-6, 1e-6, 1e-6, 0      ],
                [np.inf, z_coords[-1], y_coords[-1], x_coords[-1], np.inf, np.inf, np.inf, np.inf],
            ),
            maxfev=5000,
        )
    except Exception:
        return False, None, None, None, None

    _, cz_fit, cy_fit, cx_fit, sz, sy, sx, _ = popt
    sz, sy, sx = abs(sz), abs(sy), abs(sx)

    # Sigma bounds check
    if not (sigma_z_bounds[0]  <= sz <= sigma_z_bounds[1]):
        return False, sz, sy, sx, None
    if not (sigma_xy_bounds[0] <= sy <= sigma_xy_bounds[1]):
        return False, sz, sy, sx, None
    if not (sigma_xy_bounds[0] <= sx <= sigma_xy_bounds[1]):
        return False, sz, sy, sx, None

    # Centre offset check
    roi_cz = (nz // 2) * dz
    roi_cy = (ny // 2) * dx
    roi_cx = (nx // 2) * dx

    if abs(cz_fit - roi_cz) > max_center_offset_px * dz:
        return False, sz, sy, sx, None
    if abs(cy_fit - roi_cy) > max_center_offset_px * dx:
        return False, sz, sy, sx, None
    if abs(cx_fit - roi_cx) > max_center_offset_px * dx:
        return False, sz, sy, sx, None

    offset_px = (
        (cz_fit - roi_cz) / dz,
        (cy_fit - roi_cy) / dx,
        (cx_fit - roi_cx) / dx,
    )
    return True, sz, sy, sx, offset_px


def _center_and_average(rois, offsets):
    """
    Sub-pixel-align each ROI using its Gaussian offset and compute the mean.

    Border pixels produced by ndi.shift are set to NaN (cval=np.nan) so
    that np.nanmean ignores them.  This avoids the zero-padding bias that
    would arise from filling missing pixels with 0.

    Parameters
    ----------
    rois    : list of ndarray (float32)
    offsets : list of tuple (oz, oy, ox) in pixels

    Returns
    -------
    psf : ndarray (float64)
    """
    aligned = []
    for roi, (oz, oy, ox) in zip(rois, offsets):
        shifted = ndi.shift(
            roi.astype(np.float64),
            shift=(-oz, -oy, -ox),
            order=3,
            mode='constant',
            cval=np.nan,
        )
        aligned.append(shifted)

    return np.nanmean(aligned, axis=0)


# =============================================================================
# Public API
# =============================================================================

def estimate_psf_from_beads(
    tif_path,
    dx,
    dz,
    threshold=None,
    min_sep_um=2.0,
    roi_um=(2.5, 2.5, 2.5),
    sigma_xy_bounds=(0.05, 0.60),
    sigma_z_bounds=(0.05, 1.00),
    dog_sigma_small_um=0.08,
    dog_sigma_large_um=0.50,
    max_center_offset_px=3,
    best_fraction=0.5,
    save_path=None,
    verbose=True,
    progress_callback=None,
    return_bead_data=False,
    fit_mode='1d',
):
    """
    Estimate the experimental PSF by averaging isolated sub-diffraction beads.

    Parameters
    ----------
    tif_path : str
        Deskewed bead volume (ZYX, any integer or float dtype).
    dx : float
        Lateral (XY) pixel size in µm.
    dz : float
        Axial (Z) voxel size in µm for the deskewed volume.
        For the OPM system: dz = galvo_step_um × sin(tilt_deg) ≈ 0.110 µm.
    threshold : float or None
        Absolute threshold on the DoG image for candidate detection.
        None (default) → automatic: mean + 5·std of positive DoG values.
    min_sep_um : float
        Minimum centre-to-centre separation between beads [µm].
        Converted to an anisotropic ellipsoidal footprint for peak_local_max.
    roi_um : tuple of float (rz, ry, rx)
        ROI half-size in µm per axis.  Each axis is rounded to an odd pixel count.
    sigma_xy_bounds : tuple (min_um, max_um)
        Acceptable range for the fitted lateral sigma [µm].
        Typical values for NA ≈ 1.1 at 488 / 561 nm: (0.05, 0.60).
    sigma_z_bounds : tuple (min_um, max_um)
        Acceptable range for the fitted axial sigma [µm].
    dog_sigma_small_um : float
        Small DoG sigma [µm]. Must be smaller than the expected bead sigma.
    dog_sigma_large_um : float
        Large DoG sigma [µm]. Must be larger than the expected bead sigma.
    max_center_offset_px : int
        Maximum allowed distance (in pixels) between the intensity peak and the
        geometric ROI centre.  Rejects beads drifted toward the ROI edge.
    best_fraction : float  (0 < x ≤ 1)
        Fraction of beads to retain for the final average, selected by
        ascending lateral sigma (sharpest first).
        0.5 → keep the best 50%;  1.0 → use all accepted beads.
    save_path : str or None
        Output TIFF path.  Default: <tif_path without extension>_psf.tif
    verbose : bool
        Print progress and summary statistics to stdout.
    progress_callback : callable(fraction: float, message: str) or None
        Called periodically with a progress fraction in [0, 1] and a short
        status string.  Intended for updating a GUI progress bar.
    return_bead_data : bool
        If True, return (psf, save_path, bead_data) instead of (psf, save_path).
        bead_data contains per-bead positions and fitted sigma values.
    fit_mode : {'1d', '3d'}
        '1d' (default) — sequential 1-D Gaussian fits along Z, Y, X axes.
             Fast and robust for beads with a roughly symmetric PSF.
        '3d' — simultaneous 3-D Gaussian fit to the full ROI volume.
             More accurate for asymmetric PSFs but ~10–100× slower per bead.

    Returns
    -------
    psf : np.ndarray, float32, shape (Nz, Ny, Nx)
        Normalised PSF: psf.sum() == 1.  Ready for use in
        postprocess_deconvolution.deconvolve_volume().
    save_path : str
        Path where the PSF TIFF was saved.
    bead_data : dict  (only when return_bead_data=True)
        Keys
        ----
        volume_shape        (nz, ny, nx) in pixels
        dx, dz              voxel sizes in µm
        candidates_px       all peak_local_max detections, shape (N, 3) [z, y, x]
        border_px           border-rejected candidates, shape (M, 3)
        rejected_px         quality-rejected candidates, shape (K, 3)
        accepted_px         quality-passed candidates (before best_fraction), shape (J, 3)
        accepted_sigma_z    fitted σ_z  [µm], shape (J,)
        accepted_sigma_y    fitted σ_y  [µm], shape (J,)
        accepted_sigma_x    fitted σ_x  [µm], shape (J,)
        accepted_sigma_xy      mean of σ_y and σ_x [µm], shape (J,)
        accepted_ellipticity   (σ_x - σ_y) / σ_xy [-], shape (J,)
                               0 = circular, >0 = x-elongated, <0 = y-elongated
        accepted_snr           peak-above-bg / std(ROI outer shell) [-], shape (J,)
        accepted_used          bool mask (J,): True = included in the final PSF
        n_total             total candidates detected
        n_border            candidates rejected for being too close to the border
        n_quality_rejected  candidates rejected by the quality filter
        n_accepted          beads that passed quality (len of accepted_* arrays)
        n_used              beads included in the final average
    """
    def _cb(frac, msg):
        if progress_callback is not None:
            progress_callback(frac, msg)

    # --- Load volume ---
    volume = imread(tif_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume (ZYX); got shape {volume.shape}")
    if verbose:
        print(f"[PSF] Loaded: {tif_path}  shape={volume.shape}  dtype={volume.dtype}")
    _cb(0.05, f"Loaded: {os.path.basename(tif_path)}  {volume.shape}")

    # --- DoG band-pass filter ---
    dog = _dog_filter(volume, dog_sigma_small_um, dog_sigma_large_um, dx, dz)
    dog = np.clip(dog, 0, None)
    _cb(0.12, "DoG filter applied")

    # --- Automatic threshold ---
    if threshold is None:
        vals = dog[dog > 0]
        if len(vals) == 0:
            raise RuntimeError(
                "The DoG filter produced no positive values. "
                "Check that the volume is not blank or empty."
            )
        threshold = float(vals.mean() + 5.0 * vals.std())
        if verbose:
            print(f"[PSF] Auto threshold: {threshold:.1f}")
    _cb(0.15, f"Threshold: {threshold:.1f}")

    # --- ROI shape (odd pixel counts) ---
    def _to_odd_px(size_um, vox_um):
        return max(2 * int(round(size_um / vox_um)) + 1, 3)

    roi_shape = (
        _to_odd_px(roi_um[0], dz),
        _to_odd_px(roi_um[1], dx),
        _to_odd_px(roi_um[2], dx),
    )
    rz, ry, rx = roi_shape[0] // 2, roi_shape[1] // 2, roi_shape[2] // 2
    if verbose:
        print(f"[PSF] ROI: {roi_shape} px  ({roi_um[0]:.2f} × {roi_um[1]:.2f} × {roi_um[2]:.2f} µm)")

    # --- Anisotropic ellipsoidal footprint for peak detection ---
    # Converts min_sep_um to per-axis pixel counts and builds an ellipsoid mask.
    # This correctly handles dz ≠ dx (inspired by QI2lab/localize-psf).
    z_sep_px  = max(1, int(np.ceil(min_sep_um / dz)))
    xy_sep_px = max(1, int(np.ceil(min_sep_um / dx)))
    zz, yy, xx = np.ogrid[
        -z_sep_px  : z_sep_px  + 1,
        -xy_sep_px : xy_sep_px + 1,
        -xy_sep_px : xy_sep_px + 1,
    ]
    footprint = ((zz / z_sep_px) ** 2 + (yy / xy_sep_px) ** 2 + (xx / xy_sep_px) ** 2) <= 1.0

    # --- Candidate detection ---
    candidates = peak_local_max(
        dog,
        footprint=footprint,
        threshold_abs=threshold,
        exclude_border=True,
    )
    if verbose:
        print(f"[PSF] Candidates detected: {len(candidates)}")
    _cb(0.20, f"{len(candidates)} candidates detected")

    # --- ROI extraction and quality filtering ---
    nz, ny, nx = volume.shape
    valid_rois       = []
    offsets_list     = []
    # Full lists (all accepted beads, before best_fraction filtering)
    all_sigma_z        = []
    all_sigma_y        = []
    all_sigma_x        = []
    all_sigma_xy       = []
    all_ellipticity    = []   # (σ_x - σ_y) / σ_xy  — lateral asymmetry
    all_snr            = []   # peak-signal / std(outer-shell background)
    all_accepted_pos   = []
    # Lists for bead_data
    border_list      = []
    rejected_list    = []
    n_border, n_quality = 0, 0

    n_cand = len(candidates)
    for i, (cz, cy, cx) in enumerate(candidates):
        # Update progress every ~5% of candidates
        if i % max(1, n_cand // 20) == 0:
            _cb(0.20 + 0.60 * i / max(n_cand, 1),
                f"Checking bead {i+1}/{n_cand} ...")

        # Reject candidates whose ROI extends beyond the volume boundary
        if (cz < rz or cz + rz + 1 > nz or
                cy < ry or cy + ry + 1 > ny or
                cx < rx or cx + rx + 1 > nx):
            n_border += 1
            border_list.append((int(cz), int(cy), int(cx)))
            continue

        roi = volume[
            cz - rz : cz + rz + 1,
            cy - ry : cy + ry + 1,
            cx - rx : cx + rx + 1,
        ].astype(np.float32)

        # Local background subtraction (5th percentile of the ROI)
        roi -= np.percentile(roi, 5)
        roi  = np.clip(roi, 0, None)

        if fit_mode == '3d':
            ok, sz, sy, sx, offset_px = _quality_check_3d(
                roi, dx, dz, sigma_xy_bounds, sigma_z_bounds, max_center_offset_px
            )
        else:
            ok, sz, sy, sx, offset_px = _quality_check_1d(
                roi, dx, dz, sigma_xy_bounds, sigma_z_bounds, max_center_offset_px
            )
        if not ok:
            n_quality += 1
            rejected_list.append((int(cz), int(cy), int(cx)))
            continue

        # SNR: peak signal (bg already subtracted) / std of the ROI outer shell
        # The outer shell (first/last slice along each axis) is far from the bead
        # centre and serves as a local background noise estimate.
        outer_shell = np.concatenate([
            roi[0].ravel(), roi[-1].ravel(),
            roi[:, 0, :].ravel(), roi[:, -1, :].ravel(),
            roi[:, :, 0].ravel(), roi[:, :, -1].ravel(),
        ])
        noise = float(np.std(outer_shell))
        snr   = float(np.max(roi)) / max(noise, 1e-6)

        valid_rois.append(roi)
        offsets_list.append(offset_px)
        all_sigma_z.append(sz)
        all_sigma_y.append(sy)
        all_sigma_x.append(sx)
        sxy_mean = (sy + sx) / 2.0
        all_sigma_xy.append(sxy_mean)
        all_ellipticity.append((sx - sy) / sxy_mean if sxy_mean > 0 else 0.0)
        all_snr.append(snr)
        all_accepted_pos.append((int(cz), int(cy), int(cx)))

    if verbose:
        print(f"[PSF] Rejected (border):  {n_border}")
        print(f"[PSF] Rejected (quality): {n_quality}")
        print(f"[PSF] Accepted beads:     {len(valid_rois)}")
    _cb(0.82, f"{len(valid_rois)} beads accepted  (border: {n_border}, quality: {n_quality})")

    if len(valid_rois) == 0:
        raise RuntimeError(
            "No valid beads found. Try:\n"
            "  - Lowering 'threshold' (or set to None for automatic)\n"
            "  - Widening 'sigma_xy_bounds' / 'sigma_z_bounds'\n"
            "  - Verifying that the input is a deskewed bead volume\n"
            "  - Increasing 'roi_um' if the beads appear large"
        )

    # --- Best-fraction selection by lateral sigma ---
    # Compute used_mask BEFORE filtering so bead_data includes all accepted beads.
    used_mask = np.ones(len(valid_rois), dtype=bool)

    sigma_z_list  = list(all_sigma_z)
    sigma_y_list  = list(all_sigma_y)
    sigma_x_list  = list(all_sigma_x)
    sigma_xy_list = list(all_sigma_xy)

    if 0.0 < best_fraction < 1.0 and len(valid_rois) > 1:
        cutoff = np.percentile(sigma_xy_list, best_fraction * 100)
        keep   = [i for i, s in enumerate(sigma_xy_list) if s <= cutoff]
        if len(keep) > 0:
            used_mask           = np.zeros(len(valid_rois), dtype=bool)
            used_mask[keep]     = True
            valid_rois_filt     = [valid_rois[i]   for i in keep]
            offsets_list_filt   = [offsets_list[i] for i in keep]
            sigma_z_list        = [all_sigma_z[i]  for i in keep]
            sigma_y_list        = [all_sigma_y[i]  for i in keep]
            sigma_x_list        = [all_sigma_x[i]  for i in keep]
            sigma_xy_list       = [all_sigma_xy[i] for i in keep]
            if verbose:
                print(f"[PSF] After selection (best {best_fraction*100:.0f}%): "
                      f"{len(valid_rois_filt)} beads")
        else:
            valid_rois_filt   = valid_rois
            offsets_list_filt = offsets_list
    else:
        valid_rois_filt   = valid_rois
        offsets_list_filt = offsets_list

    _cb(0.86, f"Selection: {used_mask.sum()} beads used in PSF")

    if verbose:
        sz_mean  = np.mean(sigma_z_list)
        sy_mean  = np.mean(sigma_y_list)
        sx_mean  = np.mean(sigma_x_list)
        sxy_mean = np.mean(sigma_xy_list)
        print(f"[PSF] σ_z  : {sz_mean:.3f} ± {np.std(sigma_z_list):.3f} µm"
              f"  →  FWHM_z  ≈ {sz_mean*2.355*1000:.0f} ± {np.std(sigma_z_list)*2.355*1000:.0f} nm")
        print(f"[PSF] σ_y  : {sy_mean:.3f} ± {np.std(sigma_y_list):.3f} µm"
              f"  →  FWHM_y  ≈ {sy_mean*2.355*1000:.0f} ± {np.std(sigma_y_list)*2.355*1000:.0f} nm")
        print(f"[PSF] σ_x  : {sx_mean:.3f} ± {np.std(sigma_x_list):.3f} µm"
              f"  →  FWHM_x  ≈ {sx_mean*2.355*1000:.0f} ± {np.std(sigma_x_list)*2.355*1000:.0f} nm")
        print(f"[PSF] σ_xy : {sxy_mean:.3f} ± {np.std(sigma_xy_list):.3f} µm"
              f"  →  FWHM_xy ≈ {sxy_mean*2.355*1000:.0f} ± {np.std(sigma_xy_list)*2.355*1000:.0f} nm")

    # --- Sub-pixel alignment and averaging ---
    _cb(0.88, "Averaging beads ...")
    psf = _center_and_average(valid_rois_filt, offsets_list_filt)

    # --- Normalise (min = 0, sum = 1) ---
    psf -= np.nanmin(psf)
    total = np.nansum(psf)
    if total > 0:
        psf /= total
    psf = np.nan_to_num(psf, nan=0.0).astype(np.float32)
    _cb(0.95, "PSF normalised")

    # --- Save ---
    if save_path is None:
        base      = os.path.splitext(tif_path)[0]
        save_path = base + '_psf.tif'

    imwrite(save_path, psf, imagej=True, metadata={'axes': 'ZYX'})
    if verbose:
        print(f"[PSF] Saved: {save_path}  shape={psf.shape}")
    _cb(1.00, f"Saved: {os.path.basename(save_path)}")

    if not return_bead_data:
        return psf, save_path

    # --- Build bead_data ---
    acc_px = (np.array(all_accepted_pos, dtype=int).reshape(-1, 3)
              if all_accepted_pos else np.empty((0, 3), dtype=int))
    bead_data = {
        'volume_shape':       volume.shape,
        'dx':                 dx,
        'dz':                 dz,
        'roi_shape':          roi_shape,
        'candidates_px':      candidates,
        'border_px':          np.array(border_list,   dtype=int).reshape(-1, 3),
        'rejected_px':        np.array(rejected_list, dtype=int).reshape(-1, 3),
        'accepted_px':        acc_px,
        'accepted_sigma_z':      np.array(all_sigma_z),
        'accepted_sigma_y':      np.array(all_sigma_y),
        'accepted_sigma_x':      np.array(all_sigma_x),
        'accepted_sigma_xy':     np.array(all_sigma_xy),
        'accepted_ellipticity':  np.array(all_ellipticity),
        'accepted_snr':          np.array(all_snr),
        'accepted_used':      used_mask,
        'n_total':            len(candidates),
        'n_border':           n_border,
        'n_quality_rejected': n_quality,
        'n_accepted':         len(all_accepted_pos),
        'n_used':             int(used_mask.sum()),
    }
    return psf, save_path, bead_data


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Estimate the experimental PSF by averaging sub-diffraction "
            "fluorescent beads in a deskewed OPM volume."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",
                   help="Deskewed bead TIFF (ZYX)")
    p.add_argument("--dx",            type=float, default=0.127,
                   help="Lateral pixel size in µm")
    p.add_argument("--dz",            type=float, default=0.110,
                   help="Axial voxel size in µm (deskewed). "
                        "Default: 0.168 × sin(41°) ≈ 0.110")
    p.add_argument("--threshold",     type=float, default=None,
                   help="DoG threshold (None = auto: mean + 5·std)")
    p.add_argument("--min-sep",       type=float, default=2.0,
                   help="Minimum bead separation [µm]")
    p.add_argument("--roi-um",        type=float, nargs=3, default=[2.5, 2.5, 2.5],
                   metavar=("Z", "Y", "X"),
                   help="ROI half-size in µm per axis")
    p.add_argument("--sigma-xy",      type=float, nargs=2, default=[0.05, 0.60],
                   metavar=("MIN", "MAX"),
                   help="Acceptable σ_xy range [µm]")
    p.add_argument("--sigma-z",       type=float, nargs=2, default=[0.05, 1.00],
                   metavar=("MIN", "MAX"),
                   help="Acceptable σ_z range [µm]")
    p.add_argument("--best-fraction", type=float, default=0.5,
                   help="Fraction of beads to use (lowest σ_xy). "
                        "1.0 = all, 0.5 = best 50%%")
    p.add_argument("--output",        type=str, default=None,
                   help="Output PSF TIFF path (default: <input>_psf.tif)")
    p.add_argument("--fit-mode",      type=str, default="1d",
                   choices=["1d", "3d"],
                   help="Fitting mode: '1d' (fast, sequential) or '3d' (accurate, slow)")
    return p.parse_args()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        from psf_gui import launch_gui
        launch_gui()
    else:
        args = _parse_args()
        estimate_psf_from_beads(
            tif_path         = args.input,
            dx               = args.dx,
            dz               = args.dz,
            threshold        = args.threshold,
            min_sep_um       = args.min_sep,
            roi_um           = tuple(args.roi_um),
            sigma_xy_bounds  = tuple(args.sigma_xy),
            sigma_z_bounds   = tuple(args.sigma_z),
            best_fraction    = args.best_fraction,
            save_path        = args.output,
            fit_mode         = args.fit_mode,
            verbose          = True,
        )
