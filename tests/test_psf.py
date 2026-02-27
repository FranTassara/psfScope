"""
Tests for estimate_psf_from_beads using synthetic bead volumes.

Beads are generated as 3-D Gaussians at known positions, which allows
verifying:
  - Candidate detection
  - Gaussian fitting and sigma estimation accuracy
  - PSF normalisation
  - Structure of the returned bead_data dictionary
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from tifffile import imwrite

# Make the parent package importable when running tests directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from postprocess_psf import estimate_psf_from_beads


# =============================================================================
# Synthetic bead generator
# =============================================================================

def _make_bead_volume(
    shape=(60, 128, 128),
    n_beads=6,
    sigma_xy_um=0.15,
    sigma_z_um=0.35,
    dx=0.127,
    dz=0.110,
    amplitude=1000,
    noise_std=20,
    seed=42,
):
    """
    Generate a ZYX volume containing *n_beads* well-separated 3-D Gaussians.

    Returns
    -------
    volume    : np.ndarray, uint16
    positions : list of (z, y, x) in pixels
    sigma_z_um, sigma_xy_um : float — ground-truth sigmas in µm
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    sigma_z_px  = sigma_z_um  / dz
    sigma_xy_px = sigma_xy_um / dx

    volume  = np.zeros(shape, dtype=np.float32)
    volume += rng.normal(0, noise_std, shape).astype(np.float32)

    margin_xy = max(20, int(4 * sigma_xy_px))
    margin_z  = max(10, int(4 * sigma_z_px))
    positions = []

    attempts = 0
    while len(positions) < n_beads and attempts < 5000:
        attempts += 1
        cz = rng.integers(margin_z,  nz - margin_z)
        cy = rng.integers(margin_xy, ny - margin_xy)
        cx = rng.integers(margin_xy, nx - margin_xy)

        # Enforce a minimum separation of 4 µm between beads
        min_sep_px = 4.0 / dx
        ok = all(
            np.sqrt(((cz - pz) * dz / dz) ** 2 +
                    ((cy - py) * dx / dx) ** 2 +
                    ((cx - px) * dx / dx) ** 2) > min_sep_px
            for pz, py, px in positions
        )
        if ok:
            positions.append((cz, cy, cx))

    # Add the 3-D Gaussians
    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    for cz, cy, cx in positions:
        g = amplitude * np.exp(
            -(((zz - cz) / sigma_z_px)  ** 2 +
              ((yy - cy) / sigma_xy_px) ** 2 +
              ((xx - cx) / sigma_xy_px) ** 2) / 2.0
        )
        volume += g

    volume = np.clip(volume, 0, None).astype(np.uint16)
    return volume, positions, sigma_z_um, sigma_xy_um


# =============================================================================
# Test suite
# =============================================================================

class TestEstimatePSF:

    def _run(self, volume, positions, sigma_z_um, sigma_xy_um,
             dx=0.127, dz=0.110, **kwargs):
        """Save the volume to a temporary TIFF and run the estimation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = os.path.join(tmpdir, "beads.tif")
            imwrite(tif_path, volume)

            result = estimate_psf_from_beads(
                tif_path         = tif_path,
                dx               = dx,
                dz               = dz,
                sigma_xy_bounds  = (0.05, 0.50),
                sigma_z_bounds   = (0.05, 0.80),
                min_sep_um       = 2.0,
                roi_um           = (1.5, 1.5, 1.5),
                best_fraction    = 0.8,
                verbose          = False,
                return_bead_data = True,
                **kwargs,
            )
        return result   # (psf, save_path, bead_data)

    def test_psf_shape_and_normalisation(self):
        """PSF must be 3-D, float32, non-negative, and sum to 1."""
        volume, positions, sz, sxy = _make_bead_volume()
        psf, _, _ = self._run(volume, positions, sz, sxy)

        assert psf.ndim   == 3,           "PSF must be 3-D"
        assert psf.dtype  == np.float32,  "PSF must be float32"
        assert np.isfinite(psf).all(),    "PSF must not contain NaN or Inf"
        assert psf.min()  >= 0,           "PSF must be non-negative"
        np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-4,
                                   err_msg="PSF must be normalised (sum = 1)")

    def test_detects_beads(self):
        """At least one valid bead must be detected and used."""
        volume, positions, sz, sxy = _make_bead_volume(n_beads=4)
        _, _, bd = self._run(volume, positions, sz, sxy)

        assert bd['n_accepted'] > 0, "Must detect at least one valid bead"
        assert bd['n_used']     > 0, "Must use at least one bead in the PSF"

    def test_bead_data_structure(self):
        """bead_data must contain all expected keys with consistent shapes."""
        volume, positions, sz, sxy = _make_bead_volume(n_beads=4)
        _, _, bd = self._run(volume, positions, sz, sxy)

        required_keys = [
            'volume_shape', 'dx', 'dz',
            'candidates_px', 'border_px', 'rejected_px', 'accepted_px',
            'accepted_sigma_z', 'accepted_sigma_y', 'accepted_sigma_x',
            'accepted_sigma_xy', 'accepted_used',
            'n_total', 'n_border', 'n_quality_rejected',
            'n_accepted', 'n_used',
        ]
        for key in required_keys:
            assert key in bd, f"Missing key in bead_data: '{key}'"

        n_acc = bd['n_accepted']
        assert bd['accepted_px'].shape      == (n_acc, 3)
        assert len(bd['accepted_sigma_z'])  == n_acc
        assert len(bd['accepted_sigma_xy']) == n_acc
        assert len(bd['accepted_used'])     == n_acc

    def test_sigma_estimate_accuracy(self):
        """Estimated sigma must be within 30% of the ground-truth value."""
        dx, dz    = 0.127, 0.110
        true_sxy  = 0.15    # µm
        true_sz   = 0.35    # µm

        volume, positions, sz, sxy = _make_bead_volume(
            sigma_xy_um=true_sxy, sigma_z_um=true_sz, dx=dx, dz=dz, n_beads=5
        )
        _, _, bd = self._run(volume, positions, sz, sxy, dx=dx, dz=dz)

        if bd['n_used'] == 0:
            pytest.skip("No beads used — adjust test parameters")

        used     = bd['accepted_used']
        mean_sxy = bd['accepted_sigma_xy'][used].mean()
        mean_sz  = bd['accepted_sigma_z'][used].mean()

        assert abs(mean_sxy - true_sxy) / true_sxy < 0.30, (
            f"σ_xy estimate ({mean_sxy:.3f} µm) deviates >30% from ground truth "
            f"({true_sxy} µm)"
        )
        assert abs(mean_sz - true_sz) / true_sz < 0.30, (
            f"σ_z estimate ({mean_sz:.3f} µm) deviates >30% from ground truth "
            f"({true_sz} µm)"
        )

    def test_progress_callback(self):
        """progress_callback must be called with fractions in [0, 1]."""
        volume, positions, sz, sxy = _make_bead_volume(n_beads=3)
        calls = []

        def _cb(frac, msg):
            calls.append((frac, msg))

        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = os.path.join(tmpdir, "beads.tif")
            imwrite(tif_path, volume)
            estimate_psf_from_beads(
                tif_path          = tif_path,
                dx                = 0.127,
                dz                = 0.110,
                roi_um            = (1.5, 1.5, 1.5),
                verbose           = False,
                progress_callback = _cb,
            )

        assert len(calls) > 0, "progress_callback was never called"
        fracs = [f for f, _ in calls]
        assert all(0.0 <= f <= 1.0 for f in fracs), (
            f"Fractions out of [0, 1]: {[f for f in fracs if not 0 <= f <= 1]}"
        )
        assert fracs[-1] == 1.0, "Last fraction must be 1.0"

    def test_backward_compatible_return(self):
        """Without return_bead_data, must return exactly (psf, save_path)."""
        volume, positions, sz, sxy = _make_bead_volume(n_beads=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = os.path.join(tmpdir, "beads.tif")
            imwrite(tif_path, volume)
            result = estimate_psf_from_beads(
                tif_path = tif_path,
                dx=0.127, dz=0.110,
                roi_um=(1.5, 1.5, 1.5),
                verbose=False,
            )
        assert len(result) == 2, (
            f"Without return_bead_data expected 2 return values, got {len(result)}"
        )
        psf, save_path = result
        assert isinstance(psf, np.ndarray)
        assert isinstance(save_path, str)

    def test_psf_peak_near_centre(self):
        """The PSF peak must lie in the central 25–75% region of each axis."""
        volume, positions, sz, sxy = _make_bead_volume(n_beads=5)
        psf, _, _ = self._run(volume, positions, sz, sxy)

        nz, ny, nx = psf.shape
        iz, iy, ix = np.unravel_index(np.argmax(psf), psf.shape)

        assert nz * 0.25 < iz < nz * 0.75, \
            f"PSF peak Z ({iz}) is far from centre ({nz//2})"
        assert ny * 0.25 < iy < ny * 0.75, \
            f"PSF peak Y ({iy}) is far from centre ({ny//2})"
        assert nx * 0.25 < ix < nx * 0.75, \
            f"PSF peak X ({ix}) is far from centre ({nx//2})"

    def test_invalid_volume_raises(self):
        """A 2-D input must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = os.path.join(tmpdir, "bad.tif")
            imwrite(tif_path, np.zeros((64, 64), dtype=np.uint16))
            with pytest.raises(ValueError, match="3-D"):
                estimate_psf_from_beads(
                    tif_path=tif_path, dx=0.127, dz=0.110, verbose=False
                )


# =============================================================================
# Quick smoke test (no pytest required)
# =============================================================================

if __name__ == "__main__":
    print("Generating synthetic bead volume ...")
    vol, pos, sz, sxy = _make_bead_volume(n_beads=8, seed=0)
    print(f"  Volume shape: {vol.shape}   Beads placed: {len(pos)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tif_path = os.path.join(tmpdir, "smoke_beads.tif")
        imwrite(tif_path, vol)

        psf, save_path, bd = estimate_psf_from_beads(
            tif_path         = tif_path,
            dx               = 0.127,
            dz               = 0.110,
            roi_um           = (1.5, 1.5, 1.5),
            return_bead_data = True,
            verbose          = True,
        )

    print(f"\nPSF shape: {psf.shape}   sum = {psf.sum():.6f}")
    print(f"Beads used: {bd['n_used']} / {bd['n_accepted']} accepted")
    if bd['n_used']:
        used = bd['accepted_used']
        print(f"FWHM_xy (mean): {bd['accepted_sigma_xy'][used].mean()*2355:.0f} nm  "
              f"(ground truth: {sxy*2355:.0f} nm)")
        print(f"FWHM_z  (mean): {bd['accepted_sigma_z'][used].mean()*2355:.0f} nm  "
              f"(ground truth: {sz*2355:.0f} nm)")
    print("\n✓ Smoke test passed")
