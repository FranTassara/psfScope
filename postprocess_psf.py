"""
Experimental PSF estimation from sub-diffraction fluorescent bead images.

Pipeline
--------
1. Anisotropic band-pass filter (Difference of Gaussians, DoG)
2. Local maximum detection with an ellipsoidal footprint (respects dz ≠ dx),
   followed by a post-detection isolation filter (minimum nearest-neighbour
   distance in physical units)
3. 3-D ROI extraction around each candidate; edge candidates discarded
4. Per-bead Gaussian fitting: 1-D sequential (default) or simultaneous 3-D
   Cascaded quality filter applied after fitting:
   - fit convergence check
   - goodness-of-fit threshold (R² ≥ r2_threshold, default 0.9)
   - sanity check (centre displacement, unphysical background, sigma at bound)
   - amplitude outlier removal via Hampel identifier (3 × 1.4826 × MAD)
5. Sub-pixel alignment of each accepted ROI using the Gaussian centre offset,
   followed by NaN-masked averaging (border pixels set to NaN do not bias the mean)
6. Normalisation to unit sum and export as a 32-bit floating-point TIFF
7. Optional quantitative comparison against a theoretical PSF (psfmodels)

Design notes
------------
- Anisotropic DoG and ellipsoidal footprint correctly handle oblique plane
  microscopy (OPM) data where the axial voxel size dz differs from dx.
- Sub-pixel alignment via Gaussian centre offset is more robust than phase
  cross-correlation for sparse bead images.
- NaN masking in ndi.shift + np.nanmean avoids border artefacts without
  zero-padding bias.
- The 3-D fit uses a 3-D extension of the radial symmetry algorithm of
  Parthasarathy (2012) as a sub-pixel initial centroid estimate (p0), reducing
  the risk of the optimiser drifting to a local minimum relative to seeding
  from the integer-resolution argmax.
- An analytical Jacobian is supplied to curve_fit for the 3-D model, reducing
  the number of function evaluations required for convergence.
- n_jobs > 1 is beneficial mainly in 3-D mode; in 1-D mode, per-bead wall
  time is of the order of milliseconds and threading overhead dominates.

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
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from tifffile import imread, imwrite

try:
    import psfmodels as _psfmodels
    _HAS_PSFMODELS = True
except ImportError:
    _HAS_PSFMODELS = False


def _r2_score(y_true, y_pred):
    """Coefficient of determination R² = 1 - SS_res / SS_tot."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# =============================================================================
# Theoretical PSF helpers (require optional dependency: psfmodels)
# =============================================================================

def _theoretical_psf(shape, dx, dz, wavelength_um, na, ni, model='vectorial'):
    """
    Generate a normalised theoretical PSF using psfmodels.

    Parameters
    ----------
    shape : tuple (nz, ny, nx)
        Must match the empirical PSF shape.
    dx : float  — lateral pixel size [µm]
    dz : float  — axial voxel size [µm]
    wavelength_um : float  — emission wavelength [µm]
    na : float   — objective numerical aperture
    ni : float   — refractive index of the immersion medium
    model : {'vectorial', 'scalar'}
        'vectorial' — vectorial (Richards-Wolf) model.
        'scalar'    — scalar (Gibson-Lanni) model.

    Returns
    -------
    psf_theory : ndarray, float64, shape (nz, ny, nx)
        Normalised to sum = 1.

    Raises
    ------
    ImportError
        If psfmodels is not installed.  Install with:
            pip install "psfscope[theory]"
        or standalone:
            pip install "psfmodels>=0.3"
    RuntimeError
        If the installed psfmodels version has an incompatible API.
    """
    if not _HAS_PSFMODELS:
        raise ImportError(
            "psfmodels is required for theoretical PSF generation.\n"
            "Install it with:  pip install \"psfscope[theory]\"\n"
            "or standalone:    pip install \"psfmodels>=0.3\""
        )

    nz, ny, nx = shape
    # psfmodels generates square YX volumes (nx × nx); use the larger dimension
    # and crop to (ny, nx) afterwards if the ROI is non-square.
    n_lateral = max(ny, nx)

    fn_name = 'vectorial_psf_centered' if model == 'vectorial' else 'scalar_psf_centered'
    psf_fn  = getattr(_psfmodels, fn_name, None)
    if psf_fn is None:
        installed = getattr(_psfmodels, '__version__', 'unknown')
        raise RuntimeError(
            f"psfmodels {installed} does not expose '{fn_name}'. "
            f"Upgrade to psfmodels>=0.3:  pip install -U psfmodels"
        )

    psf_t = psf_fn(nz=nz, nx=n_lateral, dxy=dx, dz=dz,
                   wvl=wavelength_um, na=na, ni=ni)

    # Crop to (nz, ny, nx) when n_lateral exceeds one lateral dimension
    if n_lateral > ny or n_lateral > nx:
        cy = (n_lateral - ny) // 2
        cx = (n_lateral - nx) // 2
        psf_t = psf_t[:, cy : cy + ny, cx : cx + nx]

    psf_t = np.asarray(psf_t, dtype=np.float64)
    total = psf_t.sum()
    if total > 0:
        psf_t /= total
    return psf_t


def _psf_comparison_metrics(psf_empirical, psf_theoretical):
    """
    Compute similarity metrics between two normalised PSFs (sum = 1).

    Parameters
    ----------
    psf_empirical, psf_theoretical : ndarray
        Same shape, both normalised to sum = 1.

    Returns
    -------
    dict with:
        mse       : mean squared error  (lower is better; 0 = perfect)
        ncc       : normalised cross-correlation at zero lag  (1 = perfect)
        pearson_r : Pearson correlation coefficient  (1 = perfect)
    """
    e = psf_empirical.ravel().astype(float)
    t = psf_theoretical.ravel().astype(float)

    mse = float(np.mean((e - t) ** 2))

    # NCC: cosine similarity of flattened, normalised intensity vectors
    norm_e = np.linalg.norm(e)
    norm_t = np.linalg.norm(t)
    ncc = float(np.dot(e, t) / (norm_e * norm_t)) if (norm_e > 0 and norm_t > 0) else 0.0

    # Pearson r: linear correlation of intensity distributions
    e_c   = e - e.mean()
    t_c   = t - t.mean()
    denom = np.sqrt(np.dot(e_c, e_c) * np.dot(t_c, t_c))
    pearson_r = float(np.dot(e_c, t_c) / denom) if denom > 0 else 0.0

    return {'mse': mse, 'ncc': ncc, 'pearson_r': pearson_r}


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


def _quality_check_1d(roi, dx, dz, sigma_xy_bounds, sigma_z_bounds):
    """
    Assess bead quality via 1-D Gaussian fits along Z, Y, and X.

    Sigma bounds are used as curve_fit constraints but NOT as rejection
    criteria — the caller's filter stack (_filter_sanity, _filter_r2) handles
    rejection.  Returns ok=False only when a fit fails to converge.

    Returns
    -------
    ok         : bool
    sigma_z    : float or None  [µm]
    sigma_y    : float or None  [µm]
    sigma_x    : float or None  [µm]
    offset_px  : tuple (oz, oy, ox) in pixels, or None
    amplitude  : float or None  — mean of the three 1-D amplitudes
    background : float or None  — mean of the three 1-D backgrounds
    r2         : float or None  — minimum R² across the three 1-D fits
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
        return False, None, None, None, None, None, None, None

    sz = abs(fz[2])
    sy = abs(fy[2])
    sx = abs(fx[2])

    # Sub-pixel centre offset from the ROI geometric centre
    cz_fit, cy_fit, cx_fit = fz[1], fy[1], fx[1]
    roi_cz = (nz // 2) * dz
    roi_cy = (ny // 2) * dx
    roi_cx = (nx // 2) * dx

    offset_px = (
        (cz_fit - roi_cz) / dz,
        (cy_fit - roi_cy) / dx,
        (cx_fit - roi_cx) / dx,
    )

    amplitude  = float((fz[0] + fy[0] + fx[0]) / 3.0)
    background = float((fz[3] + fy[3] + fx[3]) / 3.0)

    def _gauss(x, A, c, s, bg):
        return A * np.exp(-(x - c) ** 2 / (2 * s ** 2)) + bg

    r2 = min(
        _r2_score(roi[:, iy, ix], _gauss(z_coords, *fz)),
        _r2_score(roi[iz, :, ix], _gauss(y_coords, *fy)),
        _r2_score(roi[iz, iy, :], _gauss(x_coords, *fx)),
    )

    return True, sz, sy, sx, offset_px, amplitude, background, r2


def _radial_symmetry_3d(roi):
    """
    Sub-pixel centroid estimation via 3-D radial symmetry.

    Extends the 2-D algorithm of Parthasarathy (2012, doi:10.1038/nmeth.2071)
    to 3-D by minimising the weighted sum of squared distances from
    intensity-gradient lines to the estimated centre.  Fully vectorised;
    no iterative fitting.

    Gradients are estimated at the centres of all unit cubes using the four
    body diagonals, then combined via a pseudo-inverse to recover (gz, gy, gx).
    Weights favour large-gradient regions close to the centroid, reducing the
    influence of distant, low-signal voxels.

    Falls back to the intensity-maximum position on numerical failure (singular
    matrix or degenerate gradient field).

    Parameters
    ----------
    roi : ndarray, shape (nz, ny, nx)

    Returns
    -------
    zc, yc, xc : float  — centroid in pixel units
    """
    nz, ny, nx = roi.shape

    # Midpoint coordinates (pixel units) of each unit cube centre
    zm = np.arange(nz - 1, dtype=float) + 0.5
    ym = np.arange(ny - 1, dtype=float) + 0.5
    xm = np.arange(nx - 1, dtype=float) + 0.5
    ZM, YM, XM = np.meshgrid(zm, ym, xm, indexing='ij')   # (nz-1, ny-1, nx-1)

    # Intensity differences along the 4 body diagonals of each unit cube
    g1 = roi[1:, 1:,  1:]  - roi[:-1, :-1, :-1]   # n ∝ [ 1,  1,  1]
    g2 = roi[1:, :-1, 1:]  - roi[:-1, 1:,  :-1]   # n ∝ [ 1, -1,  1]
    g3 = roi[1:, :-1, :-1] - roi[:-1, 1:,  1:]    # n ∝ [ 1, -1, -1]
    g4 = roi[1:, 1:,  :-1] - roi[:-1, :-1, 1:]    # n ∝ [ 1,  1, -1]

    # Pseudo-inverse maps 4 diagonal measurements → (gz, gy, gx) at each cube
    N     = np.array([[1,  1,  1],
                      [1, -1,  1],
                      [1, -1, -1],
                      [1,  1, -1]], dtype=float) / np.sqrt(3)  # (4, 3)
    Npinv = np.linalg.pinv(N)                                   # (3, 4)
    G     = np.stack([g1.ravel(), g2.ravel(), g3.ravel(), g4.ravel()])  # (4, Nk)
    gradk = Npinv @ G                                            # (3, Nk)

    # Weights: large gradient magnitude, small distance to centroid
    grad_norm_sq = np.einsum('ik,ik->k', gradk, gradk)          # (Nk,)
    grad_norm    = np.sqrt(grad_norm_sq)

    coords_flat = np.stack([ZM.ravel(), YM.ravel(), XM.ravel()])       # (3, Nk)
    centroid_gn = ((coords_flat * grad_norm).sum(axis=1)
                   / (grad_norm.sum() + 1e-12))                         # (3,)
    diff        = coords_flat - centroid_gn[:, None]
    dk          = np.sqrt(np.einsum('ik,ik->k', diff, diff))           # (Nk,)
    wk          = grad_norm_sq / (dk + 1e-6)

    # Normalised gradient directions, guarded against near-zero norms
    nk = gradk / (grad_norm + 1e-12)                                    # (3, Nk)

    # 3×3 linear system  M @ [zc, yc, xc]ᵀ = b
    # M_ij = Σk wk (δij − nki nkj)
    # b_i  = Σk wk (Pk_i − nki (nk · Pk))
    M         = wk.sum() * np.eye(3) - np.einsum('k,ik,jk->ij', wk, nk, nk)
    dot_nk_Pk = np.einsum('ik,ik->k', nk, coords_flat)
    b         = (np.einsum('k,ik->i',    wk, coords_flat)
                 - np.einsum('k,ik,k->i', wk, nk, dot_nk_Pk))

    try:
        zc, yc, xc = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        iz, iy, ix = np.unravel_index(np.argmax(roi), roi.shape)
        return float(iz), float(iy), float(ix)

    return (float(np.clip(zc, 0, nz - 1)),
            float(np.clip(yc, 0, ny - 1)),
            float(np.clip(xc, 0, nx - 1)))


def _quality_check_3d(roi, dx, dz, sigma_xy_bounds, sigma_z_bounds):
    """
    Assess bead quality via a simultaneous 3-D Gaussian fit.

    Fits  I(z,y,x) = A·exp(-((z-cz)²/2σz² + (y-cy)²/2σy² + (x-cx)²/2σx²)) + bg
    to the entire ROI volume in physical-unit (µm) coordinates.

    Sigma bounds are used as curve_fit constraints but NOT as rejection
    criteria — the caller's filter stack handles rejection.

    Returns
    -------
    Same extended signature as _quality_check_1d:
    (ok, sigma_z, sigma_y, sigma_x, offset_px, amplitude, background, r2)
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
        return False, None, None, None, None, None, None, None

    # Sub-pixel centroid via radial symmetry: better p0 for (cz, cy, cx) than
    # integer-resolution argmax, with no additional iterative fitting.
    iz_rs, iy_rs, ix_rs = _radial_symmetry_3d(roi)
    cz0 = iz_rs * dz
    cy0 = iy_rs * dx
    cx0 = ix_rs * dx

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

    def _gauss3d_jac(coords, A, cz, cy, cx, sz, sy, sx, bg):  # noqa: ARG001 — bg unused: ∂I/∂bg = 1
        # Analytical Jacobian of _gauss3d w.r.t. [A, cz, cy, cx, sz, sy, sx, bg].
        # Computing G once and reusing it avoids 8 redundant exp() calls per
        # iteration compared to numerical finite-difference approximation.
        z, y, x = coords
        ez = (z - cz) ** 2 / (2 * sz ** 2)
        ey = (y - cy) ** 2 / (2 * sy ** 2)
        ex = (x - cx) ** 2 / (2 * sx ** 2)
        G  = np.exp(-(ez + ey + ex))
        AG = A * G
        J       = np.empty((len(z), 8))
        J[:, 0] = G                               # ∂/∂A
        J[:, 1] = AG * (z - cz) / sz ** 2        # ∂/∂cz
        J[:, 2] = AG * (y - cy) / sy ** 2        # ∂/∂cy
        J[:, 3] = AG * (x - cx) / sx ** 2        # ∂/∂cx
        J[:, 4] = AG * (z - cz) ** 2 / sz ** 3  # ∂/∂sz
        J[:, 5] = AG * (y - cy) ** 2 / sy ** 3  # ∂/∂sy
        J[:, 6] = AG * (x - cx) ** 2 / sx ** 3  # ∂/∂sx
        J[:, 7] = 1.0                             # ∂/∂bg
        return J

    try:
        popt, _ = curve_fit(
            _gauss3d,
            coords_flat,
            data_flat,
            p0=[A0, cz0, cy0, cx0, sz0, sy0, sx0, bg0],
            jac=_gauss3d_jac,
            bounds=(
                [0,      z_coords[0],  y_coords[0],  x_coords[0],  1e-6, 1e-6, 1e-6, 0      ],
                [np.inf, z_coords[-1], y_coords[-1], x_coords[-1], np.inf, np.inf, np.inf, np.inf],
            ),
            maxfev=5000,
        )
    except Exception:
        return False, None, None, None, None, None, None, None

    A_fit, cz_fit, cy_fit, cx_fit, sz, sy, sx, bg_fit = popt
    sz, sy, sx = abs(sz), abs(sy), abs(sx)

    roi_cz = (nz // 2) * dz
    roi_cy = (ny // 2) * dx
    roi_cx = (nx // 2) * dx

    offset_px = (
        (cz_fit - roi_cz) / dz,
        (cy_fit - roi_cy) / dx,
        (cx_fit - roi_cx) / dx,
    )

    y_pred = _gauss3d(coords_flat, *popt)
    r2     = _r2_score(data_flat, y_pred)

    return True, sz, sy, sx, offset_px, float(A_fit), float(bg_fit), float(r2)


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


def _process_one_bead(cz, cy, cx, volume, nz, ny, nx, rz, ry, rx,
                       dx, dz, sigma_xy_bounds, sigma_z_bounds, fit_mode):
    """
    Extract ROI and run Gaussian fit for one bead candidate.

    Border/edge checking is performed externally by _filter_edge before
    calling this function.  Designed for both sequential loops and
    ThreadPoolExecutor (volume is read-only; SciPy/NumPy release the GIL).

    Returns
    -------
    dict with ``status`` in {'fit_ok', 'fit_failed'} plus, for fit_ok:
        roi, sz, sy, sx, offset_px, snr, amplitude, background, r2,
        peak_offset_px — peak displacement from ROI centre in pixels,
                         used by _filter_sanity.
    """
    roi = volume[
        cz - rz : cz + rz + 1,
        cy - ry : cy + ry + 1,
        cx - rx : cx + rx + 1,
    ].astype(np.float32)

    # Coarse baseline shift: bring the ROI floor to approximately zero so that
    # (a) the initial amplitude guess A0 = max(roi) is unbiased by a large
    # additive offset, and (b) _radial_symmetry_3d operates on gradients that
    # are dominated by the bead rather than background slope.  The Gaussian fit
    # still includes a free 'bg' parameter to absorb any residual offset that
    # the 5th-percentile estimate did not remove (e.g. when the bead is bright
    # enough to contaminate the lower percentile bins of a small ROI).
    roi -= np.percentile(roi, 5)
    roi  = np.clip(roi, 0, None)

    if fit_mode == '3d':
        ok, sz, sy, sx, offset_px, amp, bg, r2 = _quality_check_3d(
            roi, dx, dz, sigma_xy_bounds, sigma_z_bounds
        )
    else:
        ok, sz, sy, sx, offset_px, amp, bg, r2 = _quality_check_1d(
            roi, dx, dz, sigma_xy_bounds, sigma_z_bounds
        )

    if not ok:
        return {'status': 'fit_failed', 'pos': (int(cz), int(cy), int(cx))}

    # Peak pixel relative to ROI geometric centre (for sanity filter)
    iz, iy, ix = np.unravel_index(np.argmax(roi), roi.shape)
    peak_offset_px = (float(iz - rz), float(iy - ry), float(ix - rx))

    outer_shell = np.concatenate([
        roi[0].ravel(), roi[-1].ravel(),
        roi[:, 0, :].ravel(), roi[:, -1, :].ravel(),
        roi[:, :, 0].ravel(), roi[:, :, -1].ravel(),
    ])
    snr = float(np.max(roi)) / max(float(np.std(outer_shell)), 1e-6)

    return {
        'status':         'fit_ok',
        'pos':            (int(cz), int(cy), int(cx)),
        'roi':            roi,
        'sz':             sz,
        'sy':             sy,
        'sx':             sx,
        'offset_px':      offset_px,
        'snr':            snr,
        'amplitude':      amp,
        'background':     bg,
        'r2':             r2,
        'peak_offset_px': peak_offset_px,
    }


# =============================================================================
# Sequential bead filter stack
# =============================================================================

def _filter_edge(candidates, vol_shape, rz, ry, rx, margin_px=2):
    """
    Retain candidates whose ROI + margin fits strictly inside the volume.

    This extends the basic ROI-bounds check: margin_px (default 2) adds a
    safety buffer beyond the half-ROI so that shifted ROIs from sub-pixel
    alignment never read outside the array.

    Orthogonal to σ: selection is based only on spatial position.

    Returns
    -------
    keep : ndarray (M, 3)  — surviving candidate positions
    """
    nz, ny, nx = vol_shape
    mz, my, mx = rz + margin_px, ry + margin_px, rx + margin_px
    mask = (
        (candidates[:, 0] >= mz) & (candidates[:, 0] + mz < nz) &
        (candidates[:, 1] >= my) & (candidates[:, 1] + my < ny) &
        (candidates[:, 2] >= mx) & (candidates[:, 2] + mx < nx)
    )
    return candidates[mask]


def _filter_isolation(candidates, min_sep_um, dx, dz):
    """
    Retain candidates whose nearest neighbour is at least min_sep_um away.

    This filter is complementary to the NMS footprint used in peak_local_max:
    - NMS prevents double-detection during peak finding (threshold ~1×FWHM).
    - This filter enforces a stricter threshold (typically ~3×FWHM_lateral)
      to discard beads whose PSFs may optically overlap even if they were
      detected as separate peaks.  Overlapping PSFs produce systematic biases
      in the Gaussian fit (broader sigma, distorted shape) that degrade the
      averaged PSF.

    Orthogonal to σ: selection is based only on spatial position.

    Returns
    -------
    keep_idx : list[int]  — indices into candidates that survive
    """
    n = len(candidates)
    if n <= 1:
        return list(range(n))

    pos_um = candidates.astype(float) * np.array([dz, dx, dx])
    keep   = []
    for i in range(n):
        diff  = pos_um - pos_um[i]
        dists = np.sqrt((diff ** 2).sum(axis=1))
        dists[i] = np.inf
        if dists.min() >= min_sep_um:
            keep.append(i)
    return keep


def _filter_amplitude(bead_list):
    """
    Discard beads whose fitted amplitude is an outlier by MAD criterion.

    Hampel identifier: |amplitude − median| > 3 × 1.4826 × MAD.
    The factor 1.4826 makes MAD a consistent estimator of the Gaussian sigma,
    so the threshold is equivalent to ±3σ for a Gaussian distribution.

    Orthogonal to σ: filters on bead brightness, not on PSF width.
    Catches: photobleached beads (dim), bead clusters or saturated pixels
    (very bright).

    Returns
    -------
    keep_idx : list[int]
    """
    if len(bead_list) < 3:
        return list(range(len(bead_list)))
    amplitudes = np.array([b['amplitude'] for b in bead_list])
    med = np.median(amplitudes)
    mad = np.median(np.abs(amplitudes - med))
    thr = 3.0 * 1.4826 * mad
    return [i for i, a in enumerate(amplitudes) if abs(a - med) <= thr]


def _filter_r2(bead_list, r2_threshold=0.9):
    """
    Discard beads where the Gaussian fit quality R² < r2_threshold.

    R² measures how well the Gaussian model describes the bead PSF,
    independently of the sigma value.  Low R² indicates: non-Gaussian shape
    (doublets, debris), fitting artefacts, or insufficient SNR.

    Returns
    -------
    keep_idx : list[int]
    """
    return [i for i, b in enumerate(bead_list) if b['r2'] >= r2_threshold]


def _filter_sanity(bead_list, sigma_xy_bounds, sigma_z_bounds,
                   max_offset_px=1.5):
    """
    Discard beads failing any of three sanity checks.

    1. Centre displacement — |fit_centre − intensity_peak| > max_offset_px (px)
       in any axis.  Indicates convergence far from the bead, typically due to
       asymmetric noise or an adjacent undetected bead.

    2. Background — background < 0 or background > max(ROI).
       Unphysical fit, likely from a degenerate data region.

    3. Sigma at upper bound — any σ within 5 % of the curve_fit upper bound.
       The fit hit the constraint wall; the sigma value is an artefact, not a
       physical measurement.

    Returns
    -------
    keep_idx : list[int]
    """
    keep = []
    for i, b in enumerate(bead_list):
        oz, oy, ox   = b['offset_px']
        pz, py, px   = b['peak_offset_px']

        # 1. Centre displacement between fit and initial peak
        if max(abs(oz - pz), abs(oy - py), abs(ox - px)) > max_offset_px:
            continue

        # 2. Unphysical background
        max_roi = float(np.max(b['roi']))
        if b['background'] < 0 or b['background'] > max_roi:
            continue

        # 3. Sigma at upper bound (within 5 %)
        if b['sz'] >= sigma_z_bounds[1]  * 0.95:
            continue
        if b['sy'] >= sigma_xy_bounds[1] * 0.95:
            continue
        if b['sx'] >= sigma_xy_bounds[1] * 0.95:
            continue

        keep.append(i)
    return keep


# =============================================================================
# FWHM measurement helpers
# =============================================================================

# =============================================================================
# Public API
# =============================================================================

def measure_fwhm_from_averaged_psf(psf_3d, voxel_size_nm):
    """
    Measure FWHM along each axis from the central 1-D profiles of an averaged PSF.

    Does not assume Gaussian shape.  Extracts profiles through the voxel of
    maximum intensity, subtracts the 5th-percentile background, interpolates
    at 10× resolution with CubicSpline, then finds the half-maximum crossings
    by linear interpolation between consecutive interpolated samples.

    Parameters
    ----------
    psf_3d : ndarray, shape (nz, ny, nx)
        Averaged, normalised PSF (any dtype; converted to float internally).
    voxel_size_nm : tuple (dz_nm, dy_nm, dx_nm)
        Physical voxel size in nanometres per axis.

    Returns
    -------
    dict with keys:
        fwhm_z_nm : float  — FWHM along Z [nm], or nan if measurement fails
        fwhm_y_nm : float  — FWHM along Y [nm], or nan if measurement fails
        fwhm_x_nm : float  — FWHM along X [nm], or nan if measurement fails
    """
    psf = np.asarray(psf_3d, dtype=float)
    nz, ny, nx = psf.shape
    dz_nm, dy_nm, dx_nm = float(voxel_size_nm[0]), float(voxel_size_nm[1]), float(voxel_size_nm[2])

    iz, iy, ix = np.unravel_index(np.argmax(psf), psf.shape)

    def _fwhm_1d(profile, vox_nm):
        n = len(profile)
        if n < 3:
            return float('nan')
        coords = np.arange(n) * vox_nm
        bg     = float(np.percentile(profile, 5))
        peak   = float(np.max(profile)) - bg
        if peak <= 0:
            return float('nan')
        half_max = bg + 0.5 * peak

        coords_fine  = np.linspace(coords[0], coords[-1], n * 10)
        profile_fine = CubicSpline(coords, profile)(coords_fine)

        above   = profile_fine >= half_max
        diffs   = np.diff(above.astype(np.int8))
        rising  = np.where(diffs ==  1)[0]
        falling = np.where(diffs == -1)[0]

        if len(rising) == 0 or len(falling) == 0:
            return float('nan')

        def _cross(idx):
            y0, y1 = profile_fine[idx], profile_fine[idx + 1]
            x0, x1 = coords_fine[idx], coords_fine[idx + 1]
            if y1 == y0:
                return x0
            return x0 + (half_max - y0) / (y1 - y0) * (x1 - x0)

        left  = _cross(rising[0])
        right = _cross(falling[-1])
        return right - left if right > left else float('nan')

    return {
        'fwhm_z_nm': _fwhm_1d(psf[:, iy, ix], dz_nm),
        'fwhm_y_nm': _fwhm_1d(psf[iz, :, ix], dy_nm),
        'fwhm_x_nm': _fwhm_1d(psf[iz, iy, :], dx_nm),
    }


def _fit_psf_from_histogram_diagnostic(fwhm_values, window_fraction=0.5):
    """
    Fit a Gaussian to the histogram mode of per-bead FWHM values.

    DIAGNOSTIC ONLY — not used for reporting.  Retained for optional JSON
    output when diagnostic_histogram_fit=True.  This estimator is not standard
    in SOLS/OPM literature and is not robust for small N or skewed distributions.

    Returns
    -------
    dict with keys: mu_fit, sigma_fit, n_used, r2, bin_centers, counts,
                    fit_mask, mode_kde
    """
    from scipy.stats import gaussian_kde as _gaussian_kde
    fwhm_values = np.asarray(fwhm_values, dtype=float)
    fwhm_values = fwhm_values[np.isfinite(fwhm_values)]
    n = len(fwhm_values)

    _nan = float('nan')
    if n < 4:
        mu = float(np.median(fwhm_values)) if n > 0 else _nan
        return dict(mu_fit=mu, sigma_fit=_nan, n_used=n, r2=_nan,
                    bin_centers=np.array([mu] if n > 0 else []),
                    counts=np.array([n] if n > 0 else []),
                    fit_mask=np.array([True] if n > 0 else [], dtype=bool),
                    mode_kde=mu)

    # Step 1: Freedman-Diaconis histogram
    counts, edges = np.histogram(fwhm_values, bins='fd')
    bin_centers   = 0.5 * (edges[:-1] + edges[1:])

    # Step 2: KDE mode on fine grid — avoids bin-size dependence
    try:
        kde      = _gaussian_kde(fwhm_values)
        grid     = np.linspace(fwhm_values.min(), fwhm_values.max(), 1000)
        mode_kde = float(grid[np.argmax(kde(grid))])
    except np.linalg.LinAlgError:
        # All values identical (singular covariance); fall back to mean
        mode_kde = float(fwhm_values.mean())

    # Step 3: connected window around the KDE mode
    # Use histogram peak bin as anchor when KDE mode diverges (skewed distributions,
    # small N).  KDE provides sub-bin precision only when it stays within one
    # bin-width of the histogram peak.
    count_max  = float(counts.max())
    peak_bin   = int(np.argmax(counts))
    bin_width  = float(edges[1] - edges[0])
    if abs(mode_kde - bin_centers[peak_bin]) > 1.5 * bin_width:
        anchor = bin_centers[peak_bin]
    else:
        anchor = mode_kde
    mode_bin = int(np.clip(np.searchsorted(edges[1:], anchor, side='left'),
                           0, len(counts) - 1))

    def _grow_window(thr):
        m = np.zeros(len(counts), dtype=bool)
        m[mode_bin] = True
        for j in range(mode_bin - 1, -1, -1):
            if counts[j] >= thr:
                m[j] = True
            else:
                break
        for j in range(mode_bin + 1, len(counts)):
            if counts[j] >= thr:
                m[j] = True
            else:
                break
        return m

    fit_mask = _grow_window(window_fraction * count_max)

    # Expand window to at least 5 bins for a well-conditioned 3-parameter fit.
    # Progressively lower the threshold; fall back to the nearest-bin expansion.
    n_min_bins = 5
    for frac in (0.30, 0.15, 0.05, 0.0):
        if fit_mask.sum() >= n_min_bins:
            break
        fit_mask = _grow_window(frac * count_max)
    if fit_mask.sum() < n_min_bins and len(counts) >= n_min_bins:
        dist = np.abs(np.arange(len(counts)) - mode_bin)
        for idx in np.argsort(dist):
            fit_mask[idx] = True
            if fit_mask.sum() >= n_min_bins:
                break

    x_fit = bin_centers[fit_mask]
    y_fit = counts[fit_mask].astype(float)

    if len(x_fit) < 3:
        return dict(mu_fit=anchor, sigma_fit=_nan, n_used=int(fit_mask.sum()),
                    r2=_nan, bin_centers=bin_centers, counts=counts,
                    fit_mask=fit_mask, mode_kde=mode_kde)

    # Step 4: Poisson-weighted Gaussian fit
    w   = np.sqrt(y_fit + 1.0)
    A0  = count_max
    mu0 = float(x_fit[np.argmax(y_fit)])   # initial guess from window peak
    s0  = max((x_fit[-1] - x_fit[0]) / (2.0 * 2.355),
              bin_width * 0.5)
    try:
        popt, _ = curve_fit(
            lambda x, A, mu, sig: A * np.exp(-(x - mu) ** 2 / (2 * sig ** 2)),
            x_fit, y_fit,
            p0=[A0, mu0, s0],
            sigma=w, absolute_sigma=True,
            bounds=([0,    x_fit.min(), 1e-6],
                    [np.inf, x_fit.max(), np.inf]),
            maxfev=4000,
        )
        A_f, mu_f, sig_f = popt
        y_pred = A_f * np.exp(-(x_fit - mu_f) ** 2 / (2 * sig_f ** 2))
        r2     = _r2_score(y_fit, y_pred)
    except Exception:
        mu_f, sig_f, r2 = anchor, _nan, _nan

    return dict(mu_fit=float(mu_f), sigma_fit=float(abs(sig_f)),
                n_used=int(fit_mask.sum()), r2=float(r2) if np.isfinite(r2) else _nan,
                bin_centers=bin_centers, counts=counts, fit_mask=fit_mask,
                mode_kde=mode_kde)


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
    margin_px=2,
    r2_threshold=0.9,
    reporting_mode='averaged_psf',   # str or list[str]
    diagnostic_histogram_fit=False,
    save_path=None,
    verbose=True,
    progress_callback=None,
    return_bead_data=False,
    fit_mode='1d',
    n_jobs=1,
    compare_theoretical=False,
    na=None,
    wavelength_um=None,
    ni=1.333,
    psf_model='vectorial',
):
    """
    Estimate the experimental PSF by averaging isolated sub-diffraction beads.

    Filter stack (applied after per-bead Gaussian fit)
    ---------------------------------------------------
    1. Edge      — discard beads within (ROI_half + margin_px) of the border.
    2. Isolation — discard beads whose nearest neighbour is closer than
                   min_sep_um (3D Euclidean distance in µm).
    3. Fit       — discard beads where the Gaussian fit failed to converge.
    4. R²        — discard beads with Gaussian fit quality R² < r2_threshold.
    5. Sanity    — discard beads with unphysical fit parameters (centre
                   displacement, background, sigma at constraint wall).
    6. Amplitude — MAD outlier filter on fitted bead amplitude (Hampel
                   identifier; runs last so the amplitude distribution is
                   computed only from beads with reliable, physically sensible
                   fits).

    Log: "Beads: detected=N0 → edge=N1 → isolation=N2 → fit_ok=N2b →
          r²=N3 → sanity=N4 → amplitude=N5"

    FWHM reporting
    --------------
    reporting_mode='averaged_psf' (default):
        Headline = FWHM of the averaged PSF measured from its central 1-D
        profiles (no Gaussian assumption, see measure_fwhm_from_averaged_psf).
        Per-bead mean ± SD and median ± MAD are always logged as context.
        Equivalent to PSFj / Huygens averaged-PSF approach.
    reporting_mode='per_bead_mean':
        Headline = mean(per-bead FWHMs) ± SD.
        Equivalent to Sapoznik et al. eLife 2020.
    reporting_mode='per_bead_median':
        Headline = median(per-bead FWHMs) ± MAD.
        Equivalent to dOPM and PSFj per-bead median style.

    All three estimators are computed regardless of reporting_mode and stored
    in bead_data['fwhm_axes'] for downstream comparison.

    Parameters
    ----------
    tif_path : str
        Deskewed bead volume (ZYX, any integer or float dtype).
    dx : float
        Lateral (XY) pixel size in µm.
    dz : float
        Axial (Z) voxel size in µm.
    threshold : float or None
        DoG threshold.  None → automatic (mean + 5·std of positive DoG).
    min_sep_um : float
        Minimum bead separation [µm].
    roi_um : tuple (rz, ry, rx)  — ROI half-size in µm per axis.
    sigma_xy_bounds : tuple (min, max) [µm] — curve_fit lateral sigma bounds.
    sigma_z_bounds  : tuple (min, max) [µm] — curve_fit axial sigma bounds.
    dog_sigma_small_um, dog_sigma_large_um : float
    margin_px : int
        Extra border margin beyond the ROI half-size (default 2).
    r2_threshold : float
        Minimum Gaussian fit R² to accept a bead (default 0.9).
    reporting_mode : {'averaged_psf', 'per_bead_mean', 'per_bead_median'}
        Controls which FWHM estimator is shown as the headline result.
    diagnostic_histogram_fit : bool
        If True, also runs _fit_psf_from_histogram_diagnostic on each axis and
        stores the result in bead_data['fwhm_axes'][axis]['_diagnostic_histogram_fit_nm'].
        Logged with a warning; not used for reporting.
    save_path, verbose, progress_callback, return_bead_data,
    fit_mode, n_jobs, compare_theoretical, na, wavelength_um, ni, psf_model
        Same as before.

    Returns
    -------
    psf : ndarray, float32  — normalised PSF (sum = 1)
    save_path : str
    bead_data : dict  (only when return_bead_data=True)
        fwhm_averaged_psf_z/y/x   float [nm] — FWHM of averaged PSF
        fwhm_per_bead_mean_z/y/x  float [nm] — mean of per-bead FWHMs
        fwhm_per_bead_sd_z/y/x    float [nm] — SD of per-bead FWHMs
        fwhm_median_z/y/x         float [nm] — median of per-bead FWHMs
        fwhm_mad_z/y/x            float [nm] — MAD of per-bead FWHMs
        fwhm_axes                 dict — JSON-ready nested structure per axis
        reporting_mode            str
        n_edge, n_isolation, n_fit_ok, n_fit_failed,
        n_amplitude, n_r2, n_sanity  — filter stage counts
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
        print(f"[PSF] ROI: {roi_shape} px  ({roi_um[0]:.2f}×{roi_um[1]:.2f}×{roi_um[2]:.2f} µm)")

    # --- Anisotropic ellipsoidal footprint for NMS detection ---
    z_sep_px  = max(1, int(np.ceil(min_sep_um / dz)))
    xy_sep_px = max(1, int(np.ceil(min_sep_um / dx)))
    zz, yy, xx = np.ogrid[
        -z_sep_px  : z_sep_px  + 1,
        -xy_sep_px : xy_sep_px + 1,
        -xy_sep_px : xy_sep_px + 1,
    ]
    footprint = (
        (zz / z_sep_px) ** 2 + (yy / xy_sep_px) ** 2 + (xx / xy_sep_px) ** 2
    ) <= 1.0

    # --- Candidate detection (N0) ---
    candidates = peak_local_max(
        dog, footprint=footprint, threshold_abs=threshold, exclude_border=True,
    )
    n0 = len(candidates)
    if verbose:
        print(f"[PSF] Candidates detected: {n0}")
    _cb(0.18, f"{n0} candidates detected")

    nz, ny, nx = volume.shape

    # --- Filter 1: Edge (pre-fit, cheap) ---
    edge_surv = _filter_edge(candidates, volume.shape, rz, ry, rx, margin_px)
    n1        = len(edge_surv)
    border_px = candidates[~np.isin(np.arange(n0),
                                     np.where(np.all(candidates[:, None] ==
                                                      edge_surv[None], axis=2).any(axis=1))[0])]

    # --- Filter 2: Isolation (pre-fit, uses only peak positions) ---
    iso_idx  = _filter_isolation(edge_surv, min_sep_um, dx, dz)
    iso_surv = edge_surv[iso_idx]
    n2       = len(iso_surv)
    _cb(0.20, f"Edge: {n1}  Isolation: {n2}")

    # --- Fit loop on isolation survivors ---
    n_cand    = n2
    n_workers = (os.cpu_count() or 1) if n_jobs == -1 else max(1, n_jobs)

    def _worker(czyx):
        cz, cy, cx = int(czyx[0]), int(czyx[1]), int(czyx[2])
        return _process_one_bead(
            cz, cy, cx, volume, nz, ny, nx, rz, ry, rx,
            dx, dz, sigma_xy_bounds, sigma_z_bounds, fit_mode,
        )

    if n_workers == 1:
        raw_results = []
        for i, czyx in enumerate(iso_surv):
            if i % max(1, n_cand // 20) == 0:
                _cb(0.20 + 0.55 * i / max(n_cand, 1),
                    f"Fitting bead {i+1}/{n_cand} ...")
            raw_results.append(_worker(czyx))
    else:
        _done = [0]
        _lock = threading.Lock()

        def _worker_tracked(czyx):
            result = _worker(czyx)
            with _lock:
                _done[0] += 1
                d = _done[0]
            if d % max(1, n_cand // 20) == 0 or d == n_cand:
                _cb(0.20 + 0.55 * d / max(n_cand, 1),
                    f"Fitted {d}/{n_cand} ...")
            return result

        if verbose:
            print(f"[PSF] Parallel bead processing: {n_workers} threads")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            raw_results = list(executor.map(_worker_tracked, iso_surv))

    fit_ok_list   = [r for r in raw_results if r['status'] == 'fit_ok']
    fit_fail_list = [r for r in raw_results if r['status'] == 'fit_failed']
    n2b = len(fit_ok_list)

    # --- Filter 3: R² (fit quality gate; runs before physical outlier checks) ---
    r2_idx   = _filter_r2(fit_ok_list, r2_threshold)
    r2_surv  = [fit_ok_list[i] for i in r2_idx]
    n3       = len(r2_surv)

    # --- Filter 4: Sanity (specific fit failure modes) ---
    san_idx  = _filter_sanity(r2_surv, sigma_xy_bounds, sigma_z_bounds)
    san_surv = [r2_surv[i] for i in san_idx]
    n4       = len(san_surv)

    # --- Filter 5: Amplitude (Hampel/MAD; last so the distribution is built
    #     from beads with reliable, physically sensible fits only) ---
    amp_idx     = _filter_amplitude(san_surv)
    final_beads = [san_surv[i] for i in amp_idx]
    n5          = len(final_beads)

    filter_log = (f"Beads: detected={n0} → edge={n1} → isolation={n2} → "
                  f"fit_ok={n2b} → r²={n3} → sanity={n4} → amplitude={n5}")
    if verbose:
        print(f"[PSF] {filter_log}")
    _cb(0.78, filter_log)

    if n5 == 0:
        raise RuntimeError(
            "No valid beads after filtering. Try:\n"
            "  - Lowering 'threshold' (or set to None for automatic)\n"
            "  - Reducing 'min_sep_um' or 'margin_px'\n"
            "  - Lowering 'r2_threshold'\n"
            "  - Widening 'sigma_xy_bounds' / 'sigma_z_bounds'\n"
            "  - Verifying the input is a deskewed bead volume"
        )

    # --- Per-bead FWHM statistics ---
    k = 2.355 * 1000   # σ → FWHM in nm
    fwhm_z_nm = np.array([b['sz'] * k for b in final_beads])
    fwhm_y_nm = np.array([b['sy'] * k for b in final_beads])
    fwhm_x_nm = np.array([b['sx'] * k for b in final_beads])

    def _mad(v):
        a = np.asarray(v)
        return float(np.median(np.abs(a - np.median(a))))

    mean_z, sd_z = float(np.mean(fwhm_z_nm)), float(np.std(fwhm_z_nm))
    mean_y, sd_y = float(np.mean(fwhm_y_nm)), float(np.std(fwhm_y_nm))
    mean_x, sd_x = float(np.mean(fwhm_x_nm)), float(np.std(fwhm_x_nm))
    med_z,  mad_z = float(np.median(fwhm_z_nm)), _mad(fwhm_z_nm)
    med_y,  mad_y = float(np.median(fwhm_y_nm)), _mad(fwhm_y_nm)
    med_x,  mad_x = float(np.median(fwhm_x_nm)), _mad(fwhm_x_nm)

    # --- Sub-pixel alignment and averaging ---
    _cb(0.80, "Averaging beads ...")
    valid_rois   = [b['roi']       for b in final_beads]
    offsets_list = [b['offset_px'] for b in final_beads]
    psf = _center_and_average(valid_rois, offsets_list)

    # --- Normalise ---
    # Clip before normalising rather than subtracting nanmin: cubic-spline
    # shifts (order=3) can introduce small negative ringing artefacts adjacent
    # to the NaN border.  Subtracting nanmin when it is negative would add an
    # artificial positive floor to every voxel, biasing the sum-normalised
    # kernel and degrading deconvolution.  Clipping removes only the artefacts.
    psf = np.clip(psf, 0, None)
    total = np.nansum(psf)
    if total > 0:
        psf /= total
    psf = np.nan_to_num(psf, nan=0.0).astype(np.float32)
    _cb(0.87, "PSF normalised — measuring FWHM ...")

    # --- FWHM of averaged PSF (non-parametric, no Gaussian assumption) ---
    avg_fwhm = measure_fwhm_from_averaged_psf(psf, (dz * 1000.0, dx * 1000.0, dx * 1000.0))
    avg_z = avg_fwhm['fwhm_z_nm']
    avg_y = avg_fwhm['fwhm_y_nm']
    avg_x = avg_fwhm['fwhm_x_nm']

    # Normalise reporting_mode to a list so callers can pass a str or a list
    _modes = [reporting_mode] if isinstance(reporting_mode, str) else list(reporting_mode)
    _valid = {'averaged_psf', 'per_bead_mean', 'per_bead_median'}
    for _m in _modes:
        if _m not in _valid:
            raise ValueError(f"Unknown reporting_mode '{_m}'. Valid: {_valid}")

    if verbose:
        print(f"[PSF] N = {n5} beads used in PSF")
        for axis, avg_nm, mn, sd, med, mad in [
            ('z', avg_z, mean_z, sd_z, med_z, mad_z),
            ('y', avg_y, mean_y, sd_y, med_y, mad_y),
            ('x', avg_x, mean_x, sd_x, med_x, mad_x),
        ]:
            avg_str = f"{avg_nm:.0f}" if np.isfinite(avg_nm) else "?"
            if len(_modes) == 1:
                mode = _modes[0]
                if mode == 'averaged_psf':
                    line = (f"[PSF] FWHM_{axis} = {avg_str} nm (avg-PSF)   "
                            f"per-bead: mean={mn:.0f}±{sd:.0f} SD, "
                            f"median={med:.0f} ±{mad:.0f} MAD, N={n5}")
                elif mode == 'per_bead_mean':
                    line = (f"[PSF] FWHM_{axis} = {mn:.0f} ± {sd:.0f} nm "
                            f"(mean ± SD per-bead, N={n5})   avg-PSF: {avg_str} nm")
                else:  # per_bead_median
                    line = (f"[PSF] FWHM_{axis} = {med:.0f} ± {mad:.0f} nm "
                            f"(median ± MAD per-bead, N={n5})   avg-PSF: {avg_str} nm")
            else:
                parts = []
                if 'averaged_psf'    in _modes: parts.append(f"avg-PSF={avg_str}")
                if 'per_bead_mean'   in _modes: parts.append(f"mean±SD={mn:.0f}±{sd:.0f}")
                if 'per_bead_median' in _modes: parts.append(f"median±MAD={med:.0f}±{mad:.0f}")
                line = f"[PSF] FWHM_{axis}   {'   '.join(parts)}   nm   (N={n5})"
            print(line)

    if diagnostic_histogram_fit:
        if verbose:
            print("[PSF] ⚠ Diagnostic histogram fits (not used for reporting):")
        _diag_hist = {}
        for axis, fwhm_nm in [('z', fwhm_z_nm), ('y', fwhm_y_nm), ('x', fwhm_x_nm)]:
            hd = _fit_psf_from_histogram_diagnostic(fwhm_nm)
            _diag_hist[axis] = hd
            if verbose:
                r2_str = f"{hd['r2']:.3f}" if np.isfinite(hd.get('r2', float('nan'))) else "?"
                print(f"[PSF]   FWHM_{axis} histogram mode = {hd['mu_fit']:.0f} nm  "
                      f"(R²={r2_str}, diagnostic only)")
    else:
        _diag_hist = {}

    _cb(0.93, "FWHM measured")

    # --- Theoretical PSF comparison (optional) ---
    if compare_theoretical:
        if na is None or wavelength_um is None:
            raise ValueError(
                "compare_theoretical=True requires 'na' and 'wavelength_um'."
            )
        try:
            psf_theory = _theoretical_psf(
                psf.shape, dx, dz, wavelength_um, na, ni, psf_model
            )
            metrics = _psf_comparison_metrics(psf.astype(float), psf_theory)
            if verbose:
                print(f"[PSF] Theory vs empirical  (model: {psf_model}):")
                print(f"[PSF]   MSE={metrics['mse']:.6f}  "
                      f"NCC={metrics['ncc']:.4f}  r={metrics['pearson_r']:.4f}")
            _cb(0.97, f"Theory NCC={metrics['ncc']:.3f}  MSE={metrics['mse']:.2e}")
        except Exception as exc:
            if verbose:
                print(f"[PSF] Warning: theory comparison failed: {exc}")
            psf_theory = None
            metrics    = {'mse': None, 'ncc': None, 'pearson_r': None}
    else:
        psf_theory = None
        metrics    = {'mse': None, 'ncc': None, 'pearson_r': None}

    # --- Save ---
    if save_path is None:
        save_path = os.path.splitext(tif_path)[0] + '_psf.tif'
    imwrite(save_path, psf, imagej=True, metadata={'axes': 'ZYX'})
    if verbose:
        print(f"[PSF] Saved: {save_path}  shape={psf.shape}")
    _cb(1.00, f"Saved: {os.path.basename(save_path)}")

    if not return_bead_data:
        return psf, save_path

    # --- Build bead_data ---
    # accepted_* arrays cover all fit_ok beads (n2b); accepted_used marks the n5 final ones.
    all_sigma_z      = [b['sz']  for b in fit_ok_list]
    all_sigma_y      = [b['sy']  for b in fit_ok_list]
    all_sigma_x      = [b['sx']  for b in fit_ok_list]
    all_sigma_xy     = [(b['sy'] + b['sx']) / 2.0 for b in fit_ok_list]
    all_ellipticity  = [((b['sx'] - b['sy']) / ((b['sy'] + b['sx']) / 2.0)
                          if (b['sy'] + b['sx']) > 0 else 0.0)
                         for b in fit_ok_list]
    all_snr          = [b['snr'] for b in fit_ok_list]
    all_accepted_pos = [b['pos'] for b in fit_ok_list]

    final_set   = set(id(b) for b in final_beads)
    used_mask   = np.array([id(b) in final_set for b in fit_ok_list], dtype=bool)

    acc_px = (np.array(all_accepted_pos, dtype=int).reshape(-1, 3)
              if all_accepted_pos else np.empty((0, 3), dtype=int))

    # Rejected positions: fit_failed + all post-fit filter rejects
    rejected_pos = [r['pos'] for r in fit_fail_list]
    for b in fit_ok_list:
        if id(b) not in final_set:
            rejected_pos.append(b['pos'])

    # Border positions: candidates that failed the edge filter
    edge_set     = set(map(tuple, edge_surv.tolist()))
    border_pos   = [tuple(c) for c in candidates.tolist() if tuple(c) not in edge_set]

    def _px_arr(lst):
        return (np.array(lst, dtype=int).reshape(-1, 3)
                if lst else np.empty((0, 3), dtype=int))

    bead_data = {
        # Geometry
        'volume_shape': volume.shape,
        'dx': dx, 'dz': dz, 'roi_shape': roi_shape,
        # Detection
        'candidates_px': candidates,
        'border_px':     _px_arr(border_pos),
        'rejected_px':   _px_arr(rejected_pos),
        # Per-bead arrays (all fit_ok beads, n2b)
        'accepted_px':         acc_px,
        'accepted_sigma_z':    np.array(all_sigma_z),
        'accepted_sigma_y':    np.array(all_sigma_y),
        'accepted_sigma_x':    np.array(all_sigma_x),
        'accepted_sigma_xy':   np.array(all_sigma_xy),
        'accepted_ellipticity':np.array(all_ellipticity),
        'accepted_snr':        np.array(all_snr),
        'accepted_used':       used_mask,
        # Filter-stage counts
        'n_total':        n0,
        'n_edge':         n1,
        'n_border':       n0 - n1,
        'n_isolation':    n2,
        'n_fit_failed':   len(fit_fail_list),
        'n_fit_ok':       n2b,
        'n_amplitude':    n3,
        'n_r2':           n4,
        'n_sanity':       n5,
        'n_quality_rejected': (n0 - n1) + (n1 - n2) + len(fit_fail_list),
        'n_accepted':     n2b,
        'n_used':         n5,
        # FWHM reporting — per-bead statistics
        'fwhm_per_bead_mean_z': mean_z, 'fwhm_per_bead_mean_y': mean_y, 'fwhm_per_bead_mean_x': mean_x,
        'fwhm_per_bead_sd_z':   sd_z,   'fwhm_per_bead_sd_y':   sd_y,   'fwhm_per_bead_sd_x':   sd_x,
        'fwhm_median_z': med_z, 'fwhm_median_y': med_y, 'fwhm_median_x': med_x,
        'fwhm_mad_z':    mad_z, 'fwhm_mad_y':    mad_y, 'fwhm_mad_x':    mad_x,
        # FWHM reporting — averaged PSF (non-parametric)
        'fwhm_averaged_psf_z': avg_z, 'fwhm_averaged_psf_y': avg_y, 'fwhm_averaged_psf_x': avg_x,
        # JSON-ready combined structure (all estimators, independent of reporting_mode)
        'fwhm_axes': {
            'axis_z': {
                'fwhm_averaged_psf_nm':    round(avg_z,  1) if np.isfinite(avg_z)  else None,
                'fwhm_per_bead_mean_nm':   round(mean_z, 1),
                'fwhm_per_bead_sd_nm':     round(sd_z,   1),
                'fwhm_per_bead_median_nm': round(med_z,  1),
                'fwhm_per_bead_mad_nm':    round(mad_z,  1),
                'n_beads_used': n5, 'fit_mode': fit_mode.upper(),
                **({'_diagnostic_histogram_fit_nm': round(_diag_hist['z']['mu_fit'], 1)}
                   if _diag_hist.get('z') else {}),
            },
            'axis_y': {
                'fwhm_averaged_psf_nm':    round(avg_y,  1) if np.isfinite(avg_y)  else None,
                'fwhm_per_bead_mean_nm':   round(mean_y, 1),
                'fwhm_per_bead_sd_nm':     round(sd_y,   1),
                'fwhm_per_bead_median_nm': round(med_y,  1),
                'fwhm_per_bead_mad_nm':    round(mad_y,  1),
                'n_beads_used': n5, 'fit_mode': fit_mode.upper(),
                **({'_diagnostic_histogram_fit_nm': round(_diag_hist['y']['mu_fit'], 1)}
                   if _diag_hist.get('y') else {}),
            },
            'axis_x': {
                'fwhm_averaged_psf_nm':    round(avg_x,  1) if np.isfinite(avg_x)  else None,
                'fwhm_per_bead_mean_nm':   round(mean_x, 1),
                'fwhm_per_bead_sd_nm':     round(sd_x,   1),
                'fwhm_per_bead_median_nm': round(med_x,  1),
                'fwhm_per_bead_mad_nm':    round(mad_x,  1),
                'n_beads_used': n5, 'fit_mode': fit_mode.upper(),
                **({'_diagnostic_histogram_fit_nm': round(_diag_hist['x']['mu_fit'], 1)}
                   if _diag_hist.get('x') else {}),
            },
        },
        'reporting_mode': ','.join(_modes),
        # Theoretical
        'psf_theoretical': psf_theory,
        'psf_mse':         metrics['mse'],
        'psf_ncc':         metrics['ncc'],
        'psf_pearson_r':   metrics['pearson_r'],
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
    p.add_argument("--margin-px",      type=int,   default=2,
                   help="Minimum margin in pixels between a bead ROI and the "
                        "volume edge (edge filter).")
    p.add_argument("--r2-threshold",  type=float, default=0.9,
                   help="Minimum R² for accepting a Gaussian fit (R² filter).")
    p.add_argument("--reporting-mode", type=str, nargs='+',
                   default=["averaged_psf"],
                   choices=["averaged_psf", "per_bead_mean", "per_bead_median"],
                   metavar="MODE",
                   help="One or more FWHM reporting modes (space-separated). "
                        "Choices: averaged_psf (default), per_bead_mean, "
                        "per_bead_median.  Example: --reporting-mode averaged_psf "
                        "per_bead_mean")
    p.add_argument("--diagnostic-histogram-fit", action="store_true",
                   help="Run histogram Gaussian fit on per-bead FWHMs and store "
                        "the result in the JSON output (diagnostic only).")
    p.add_argument("--output",        type=str, default=None,
                   help="Output PSF TIFF path (default: <input>_psf.tif)")
    p.add_argument("--fit-mode",      type=str, default="1d",
                   choices=["1d", "3d"],
                   help="Fitting mode: '1d' (fast, sequential) or '3d' (accurate, slow)")
    p.add_argument("--n-jobs",        type=int, default=1,
                   help="Worker threads for the bead loop. "
                        "1 = sequential; -1 = all CPU cores. "
                        "Most useful with --fit-mode 3d.")
    p.add_argument("--compare-theoretical", action="store_true",
                   help="Compare empirical PSF to a theoretical model (requires psfmodels).")
    p.add_argument("--na",            type=float, default=None,
                   help="Numerical aperture (required with --compare-theoretical).")
    p.add_argument("--wavelength",    type=float, default=None,
                   help="Emission wavelength in µm (required with --compare-theoretical).")
    p.add_argument("--ni",            type=float, default=1.333,
                   help="Immersion medium refractive index.")
    p.add_argument("--psf-model",     type=str,   default="vectorial",
                   choices=["vectorial", "scalar"],
                   help="Theoretical PSF model (vectorial = Richards-Wolf; "
                        "scalar = Gibson-Lanni).")
    return p.parse_args()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        from psf_gui import launch_gui
        launch_gui()
    else:
        args = _parse_args()
        estimate_psf_from_beads(
            tif_path             = args.input,
            dx                   = args.dx,
            dz                   = args.dz,
            threshold            = args.threshold,
            min_sep_um           = args.min_sep,
            roi_um               = tuple(args.roi_um),
            sigma_xy_bounds      = tuple(args.sigma_xy),
            sigma_z_bounds       = tuple(args.sigma_z),
            margin_px            = args.margin_px,
            r2_threshold         = args.r2_threshold,
            reporting_mode           = args.reporting_mode,
            diagnostic_histogram_fit = args.diagnostic_histogram_fit,
            save_path            = args.output,
            fit_mode             = args.fit_mode,
            n_jobs               = args.n_jobs,
            compare_theoretical  = args.compare_theoretical,
            na                   = args.na,
            wavelength_um        = args.wavelength,
            ni                   = args.ni,
            psf_model            = args.psf_model,
            verbose              = True,
        )
