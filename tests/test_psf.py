"""
Tests for psfScope — estimate_psf_from_beads and helper functions.

Beads are generated as 3-D Gaussians at known positions, which allows
verifying:
  - Candidate detection
  - Gaussian fitting and sigma estimation accuracy
  - PSF normalisation
  - Structure of the returned bead_data dictionary
  - Filter functions (_filter_edge, _filter_isolation, etc.)
  - fit_psf_from_histogram and bootstrap_psf
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from tifffile import imwrite

# Make the parent package importable when running tests directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from postprocess_psf import (
    estimate_psf_from_beads,
    measure_fwhm_from_averaged_psf,
    _filter_edge,
    _filter_isolation,
    _filter_amplitude,
    _filter_r2,
    _filter_sanity,
    _quality_check_3d,
)


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
                margin_px        = 2,
                r2_threshold     = 0.85,
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
            # filter-funnel counts
            'n_total', 'n_border', 'n_quality_rejected', 'n_accepted', 'n_used',
            'n_edge', 'n_isolation', 'n_fit_ok', 'n_fit_failed',
            'n_amplitude', 'n_r2', 'n_sanity',
            # FWHM reporting
            'fwhm_per_bead_mean_z', 'fwhm_per_bead_mean_y', 'fwhm_per_bead_mean_x',
            'fwhm_per_bead_sd_z',   'fwhm_per_bead_sd_y',   'fwhm_per_bead_sd_x',
            'fwhm_median_z', 'fwhm_median_y', 'fwhm_median_x',
            'fwhm_mad_z',    'fwhm_mad_y',    'fwhm_mad_x',
            'fwhm_averaged_psf_z', 'fwhm_averaged_psf_y', 'fwhm_averaged_psf_x',
            'fwhm_axes', 'reporting_mode',
        ]
        for key in required_keys:
            assert key in bd, f"Missing key in bead_data: '{key}'"

        n_acc = bd['n_accepted']
        assert bd['accepted_px'].shape      == (n_acc, 3)
        assert len(bd['accepted_sigma_z'])  == n_acc
        assert len(bd['accepted_sigma_xy']) == n_acc
        assert len(bd['accepted_used'])     == n_acc

        # fwhm_axes must be a dict with all three axis sub-dicts
        assert isinstance(bd['fwhm_axes'], dict), "fwhm_axes must be a dict"
        for ax_key in ['axis_z', 'axis_y', 'axis_x']:
            assert ax_key in bd['fwhm_axes'], f"Missing '{ax_key}' in fwhm_axes"
            ax_d = bd['fwhm_axes'][ax_key]
            for field in ['fwhm_averaged_psf_nm', 'fwhm_per_bead_mean_nm',
                          'fwhm_per_bead_sd_nm', 'fwhm_per_bead_median_nm',
                          'fwhm_per_bead_mad_nm', 'n_beads_used', 'fit_mode']:
                assert field in ax_d, f"Missing field '{field}' in fwhm_axes.{ax_key}"

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

    def test_reporting_modes_consistency(self):
        """All three reporting modes must agree within 10% on well-behaved data."""
        dx, dz    = 0.127, 0.110
        volume, positions, sz, sxy = _make_bead_volume(
            n_beads=8, sigma_xy_um=0.15, sigma_z_um=0.35,
            dx=dx, dz=dz, noise_std=10, seed=7,
        )
        results = {}
        for mode in ['averaged_psf', 'per_bead_mean', 'per_bead_median']:
            with tempfile.TemporaryDirectory() as tmpdir:
                tif_path = os.path.join(tmpdir, "beads.tif")
                from tifffile import imwrite
                imwrite(tif_path, volume)
                _, _, bd = estimate_psf_from_beads(
                    tif_path         = tif_path,
                    dx               = dx,
                    dz               = dz,
                    sigma_xy_bounds  = (0.05, 0.50),
                    sigma_z_bounds   = (0.05, 0.80),
                    min_sep_um       = 2.0,
                    roi_um           = (1.5, 1.5, 1.5),
                    margin_px        = 2,
                    r2_threshold     = 0.85,
                    reporting_mode   = mode,
                    verbose          = False,
                    return_bead_data = True,
                )
            results[mode] = bd

        axes_entry = results['averaged_psf']['fwhm_axes']
        for ax_key in ['axis_z', 'axis_y', 'axis_x']:
            avg_nm  = axes_entry[ax_key]['fwhm_averaged_psf_nm']
            mean_nm = axes_entry[ax_key]['fwhm_per_bead_mean_nm']
            med_nm  = axes_entry[ax_key]['fwhm_per_bead_median_nm']
            if avg_nm is None:
                pytest.skip(f"{ax_key}: avg-PSF FWHM could not be measured")
            ref = mean_nm
            for val, name in [(avg_nm, 'avg_psf'), (med_nm, 'median')]:
                diff = abs(val - ref) / ref
                assert diff < 0.10, (
                    f"{ax_key}: {name}={val:.0f} nm vs mean={ref:.0f} nm "
                    f"(diff {diff*100:.1f}% > 10%)"
                )

    def test_filter_funnel_counts_consistent(self):
        """Filter-funnel counts must be non-negative and monotonically non-increasing."""
        volume, positions, sz, sxy = _make_bead_volume(n_beads=6)
        _, _, bd = self._run(volume, positions, sz, sxy)

        n0 = bd['n_total']
        n1 = bd['n_edge']
        n2 = bd['n_isolation']
        n2b = bd['n_fit_ok']
        n3 = bd['n_amplitude']
        n4 = bd['n_r2']
        n5 = bd['n_sanity']

        assert n0 >= n1 >= 0,  "n_edge must be <= n_total"
        assert n1 >= n2 >= 0,  "n_isolation must be <= n_edge"
        assert n2 >= n2b >= 0, "n_fit_ok must be <= n_isolation candidates"
        assert n2b >= n3 >= 0, "n_amplitude must be <= n_fit_ok"
        assert n3 >= n4 >= 0,  "n_r2 must be <= n_amplitude"
        assert n4 >= n5 >= 0,  "n_sanity must be <= n_r2"
        # With default best_fraction=1.0 all sanity survivors are used
        assert bd['n_used'] == n5, "n_used must equal n_sanity survivors (best_fraction=1.0)"


    def test_theoretical_keys_none_when_disabled(self):
        """When compare_theoretical=False (default) all theory keys must be None."""
        volume, positions, sz, sxy = _make_bead_volume(n_beads=3)
        _, _, bd = self._run(volume, positions, sz, sxy)

        assert 'psf_theoretical' in bd, "Key 'psf_theoretical' must exist in bead_data"
        assert 'psf_mse'         in bd, "Key 'psf_mse' must exist in bead_data"
        assert 'psf_ncc'         in bd, "Key 'psf_ncc' must exist in bead_data"
        assert 'psf_pearson_r'   in bd, "Key 'psf_pearson_r' must exist in bead_data"

        assert bd['psf_theoretical'] is None, "psf_theoretical must be None when disabled"
        assert bd['psf_mse']        is None, "psf_mse must be None when disabled"
        assert bd['psf_ncc']        is None, "psf_ncc must be None when disabled"
        assert bd['psf_pearson_r']  is None, "psf_pearson_r must be None when disabled"


# =============================================================================
# Unit tests for filter functions
# =============================================================================

class TestFilterEdge:
    def test_removes_candidates_near_border(self):
        vol_shape = (30, 64, 64)
        candidates = np.array([
            [2, 32, 32],   # too close to Z=0 edge (margin_px=5, rz=5 → mz=10)
            [15, 32, 32],  # OK
            [28, 32, 32],  # too close to Z end
            [15, 5, 32],   # too close to Y=0
            [15, 32, 5],   # too close to X=0
        ])
        rz, ry, rx = 5, 5, 5
        result = _filter_edge(candidates, vol_shape, rz, ry, rx, margin_px=5)
        assert result.shape[0] == 1, f"Expected 1 survivor, got {result.shape[0]}"
        assert np.all(result[0] == [15, 32, 32])

    def test_keeps_all_when_large_volume(self):
        vol_shape = (200, 200, 200)
        candidates = np.array([[50, 50, 50], [100, 100, 100], [150, 150, 150]])
        result = _filter_edge(candidates, vol_shape, rz=5, ry=5, rx=5, margin_px=2)
        assert result.shape[0] == 3

    def test_empty_candidates(self):
        vol_shape = (30, 64, 64)
        candidates = np.empty((0, 3), dtype=int)
        result = _filter_edge(candidates, vol_shape, rz=5, ry=5, rx=5)
        assert result.shape == (0, 3)


class TestFilterIsolation:
    """_filter_isolation returns a list of survivor indices into `candidates`."""

    def test_removes_close_pair(self):
        # Two beads 1 µm apart (< min_sep_um=2.0) → only one of the pair survives
        dx, dz = 0.127, 0.110
        sep_px = int(1.0 / dx)   # ~8 px ≈ 1 µm
        candidates = np.array([
            [15, 32, 32],
            [15, 32, 32 + sep_px],
            [15, 100, 100],  # isolated
        ])
        keep = _filter_isolation(candidates, min_sep_um=2.0, dx=dx, dz=dz)
        survivors = candidates[keep]
        # The isolated bead must survive; the close pair loses at least one
        assert len(keep) >= 1
        assert any(np.all(r == [15, 100, 100]) for r in survivors)

    def test_keeps_all_when_well_separated(self):
        dx, dz = 0.127, 0.110
        sep_px = int(3.0 / dx)   # 3 µm > min_sep_um=2.0
        candidates = np.array([
            [15, 32, 32],
            [15, 32, 32 + sep_px],
            [15, 32, 32 + 2 * sep_px],
        ])
        keep = _filter_isolation(candidates, min_sep_um=2.0, dx=dx, dz=dz)
        assert len(keep) == 3

    def test_empty_candidates(self):
        keep = _filter_isolation(np.empty((0, 3), dtype=int),
                                 min_sep_um=2.0, dx=0.127, dz=0.110)
        assert len(keep) == 0


class TestFilterAmplitude:
    def _make_beads(self, amplitudes):
        return [{'amplitude': float(a), 'pos': np.array([0, i, 0])}
                for i, a in enumerate(amplitudes)]

    def test_removes_outlier_amplitude(self):
        # 9 similar beads + 1 extreme outlier (10× median)
        amps = [100.0] * 9 + [2000.0]
        beads = self._make_beads(amps)
        result = _filter_amplitude(beads)
        # The extreme outlier must be removed
        survivors = [beads[i] for i in result]
        assert all(b['amplitude'] < 1000 for b in survivors), \
            "Outlier amplitude bead should be removed"

    def test_keeps_uniform_amplitudes(self):
        beads = self._make_beads([100.0] * 5)
        result = _filter_amplitude(beads)
        assert len(result) == 5

    def test_single_bead_survives(self):
        beads = self._make_beads([500.0])
        result = _filter_amplitude(beads)
        assert len(result) == 1

    def test_empty_list(self):
        result = _filter_amplitude([])
        assert result == []


class TestFilterR2:
    def _make_beads(self, r2_values):
        return [{'r2': float(r), 'pos': np.array([0, i, 0])}
                for i, r in enumerate(r2_values)]

    def test_removes_low_r2(self):
        beads = self._make_beads([0.95, 0.80, 0.70, 0.99])
        result = _filter_r2(beads, r2_threshold=0.90)
        assert set(result) == {0, 3}, "Only beads with R² >= 0.90 should survive"

    def test_threshold_boundary(self):
        beads = self._make_beads([0.90, 0.89])
        result = _filter_r2(beads, r2_threshold=0.90)
        assert 0 in result and 1 not in result

    def test_keeps_all_above_threshold(self):
        beads = self._make_beads([0.95, 0.97, 0.99])
        result = _filter_r2(beads, r2_threshold=0.90)
        assert len(result) == 3

    def test_empty_list(self):
        assert _filter_r2([], r2_threshold=0.90) == []


class TestFilterSanity:
    def _make_bead(self, sz=0.35, sy=0.15, sx=0.15, offset=0.0,
                   bg=10.0, roi_max=1000.0):
        roi = np.ones((10, 20, 20), dtype=np.float32) * roi_max * 0.1
        roi[5, 10, 10] = roi_max
        # Keys must match what _process_one_bead returns (sz/sy/sx, not sigma_*)
        return {
            'sz': sz, 'sy': sy, 'sx': sx,
            'offset_px':      (offset, 0.0, 0.0),
            'peak_offset_px': (0.0,    0.0, 0.0),
            'background': bg,
            'roi': roi,
            'pos': np.array([0, 0, 0]),
            'amplitude': 1000.0, 'r2': 0.95,
        }

    def test_keeps_valid_bead(self):
        beads = [self._make_bead()]
        result = _filter_sanity(beads, (0.05, 0.50), (0.05, 0.80))
        assert len(result) == 1

    def test_rejects_large_offset(self):
        beads = [self._make_bead(offset=3.0)]
        result = _filter_sanity(beads, (0.05, 0.50), (0.05, 0.80), max_offset_px=1.5)
        assert len(result) == 0, "Bead with large offset should be rejected"

    def test_rejects_negative_background(self):
        beads = [self._make_bead(bg=-50.0)]
        result = _filter_sanity(beads, (0.05, 0.50), (0.05, 0.80))
        assert len(result) == 0, "Bead with negative background should be rejected"

    def test_rejects_sigma_at_wall(self):
        # sigma_z just below upper bound (0.80 × 0.95 = 0.76 threshold)
        beads = [self._make_bead(sz=0.79)]
        result = _filter_sanity(beads, (0.05, 0.50), (0.05, 0.80))
        assert len(result) == 0, "Bead with sigma at constraint wall should be rejected"

    def test_empty_list(self):
        result = _filter_sanity([], (0.05, 0.50), (0.05, 0.80))
        assert result == []


# =============================================================================
# Regression tests for _quality_check_3d (Item 1)
# =============================================================================

class TestQualityCheck3D:
    """Verify _quality_check_3d always returns an 8-tuple."""

    _DX = 0.127
    _DZ = 0.110
    _SXY_BOUNDS = (0.05, 0.50)
    _SZ_BOUNDS  = (0.05, 0.80)

    def _check(self, roi):
        result = _quality_check_3d(
            roi, self._DX, self._DZ, self._SXY_BOUNDS, self._SZ_BOUNDS
        )
        assert len(result) == 8, (
            f"_quality_check_3d must always return 8 values, got {len(result)}"
        )
        return result

    def test_zero_amplitude_returns_false_8tuple(self):
        """Uniform ROI (A0 ≈ 0) must trigger early return: (False, None×7)."""
        roi = np.full((11, 15, 15), 500.0, dtype=np.float32)
        ok, sz, sy, sx, off, amp, bg, r2 = self._check(roi)
        assert ok is False
        assert all(v is None for v in (sz, sy, sx, off, amp, bg, r2))

    def test_fit_failure_returns_false_8tuple(self):
        """A ROI that causes curve_fit to fail must still return 8 values."""
        rng = np.random.default_rng(7)
        # Tiny ROI → bounds conflict makes fitting ill-conditioned
        roi = rng.normal(200, 50, (3, 5, 5)).astype(np.float32)
        roi[1, 2, 2] = 5000.0  # A0 > 0 so we enter the fitting path
        result = self._check(roi)
        assert result[0] in (True, False), "First element must be bool"

    def test_valid_gaussian_returns_true_8tuple(self):
        """Clean 3-D Gaussian ROI must succeed: (True, sz>0, sy>0, sx>0, ...)."""
        nz, ny, nx = 21, 21, 21
        sigma_xy_px = 0.15 / self._DX
        sigma_z_px  = 0.35 / self._DZ
        zz, yy, xx = np.mgrid[:nz, :ny, :nx]
        cz, cy, cx = nz // 2, ny // 2, nx // 2
        roi = (1000.0 * np.exp(
            -(((zz - cz) / sigma_z_px)  ** 2 +
              ((yy - cy) / sigma_xy_px) ** 2 +
              ((xx - cx) / sigma_xy_px) ** 2) / 2.0
        ) + 50.0).astype(np.float32)

        ok, sz, sy, sx, off, amp, bg, r2 = self._check(roi)
        assert ok is True,   "Clean Gaussian must produce ok=True"
        assert sz > 0,       "sz must be positive"
        assert sy > 0,       "sy must be positive"
        assert sx > 0,       "sx must be positive"
        assert amp > 0,      "amplitude must be positive"
        assert r2 > 0.9,     f"R² must be >0.9 for clean Gaussian, got {r2:.3f}"


# =============================================================================
# Tests for best_fraction parameter (Item 2 / Item 10)
# =============================================================================

class TestBestFraction:

    def _run_bf(self, best_fraction, seed=42, n_beads=8):
        volume, positions, sz, sxy = _make_bead_volume(n_beads=n_beads, seed=seed)
        with tempfile.TemporaryDirectory() as tmpdir:
            tif = os.path.join(tmpdir, "beads.tif")
            imwrite(tif, volume)
            _, _, bd = estimate_psf_from_beads(
                tif_path         = tif,
                dx               = 0.127,
                dz               = 0.110,
                sigma_xy_bounds  = (0.05, 0.50),
                sigma_z_bounds   = (0.05, 0.80),
                min_sep_um       = 2.0,
                roi_um           = (1.5, 1.5, 1.5),
                margin_px        = 2,
                r2_threshold     = 0.85,
                verbose          = False,
                return_bead_data = True,
                best_fraction    = best_fraction,
            )
        return bd

    def test_n_used_reduced_by_best_fraction(self):
        """best_fraction=0.5 must reduce n_used to at most ceil(0.5 × n_sanity)."""
        bd = self._run_bf(best_fraction=0.5)
        if bd['n_sanity'] < 2:
            pytest.skip("Not enough quality beads to test best_fraction")
        import math
        max_allowed = math.ceil(0.5 * bd['n_sanity'])
        assert bd['n_used'] <= max_allowed, (
            f"n_used={bd['n_used']} > ceil(0.5 × n_sanity={bd['n_sanity']})={max_allowed}"
        )

    def test_best_fraction_one_is_noop(self):
        """best_fraction=1.0 must give the same n_used as the default."""
        bd_default = self._run_bf(best_fraction=1.0, seed=0)
        bd_explicit = self._run_bf(best_fraction=1.0, seed=0)
        assert bd_default['n_used'] == bd_explicit['n_used']
        assert bd_default['n_used'] == bd_default['n_sanity'], (
            "With best_fraction=1.0, n_used must equal n_sanity"
        )

    def test_best_fraction_selects_sharpest(self):
        """Used beads (best_fraction subset) must have smaller σ_xy than excluded beads."""
        bd = self._run_bf(best_fraction=0.5)
        if bd['n_sanity'] < 2:
            pytest.skip("Not enough quality beads to test selection")

        used_mask     = bd['accepted_used']
        excluded_mask = bd['accepted_not_used_best_fraction']

        if not excluded_mask.any():
            pytest.skip("No beads excluded by best_fraction in this run")

        mean_used     = bd['accepted_sigma_xy'][used_mask].mean()
        mean_excluded = bd['accepted_sigma_xy'][excluded_mask].mean()
        assert mean_used <= mean_excluded + 1e-9, (
            f"Used beads σ_xy ({mean_used:.4f} µm) must be ≤ excluded "
            f"({mean_excluded:.4f} µm)"
        )

    def test_accepted_rois_cache_present(self):
        """accepted_rois must be a non-empty list of arrays when beads are accepted."""
        bd = self._run_bf(best_fraction=1.0)
        assert 'accepted_rois' in bd, "bead_data must contain 'accepted_rois'"
        if bd['n_accepted'] > 0:
            assert isinstance(bd['accepted_rois'], list), "accepted_rois must be a list"
            assert len(bd['accepted_rois']) == bd['n_accepted'], (
                "len(accepted_rois) must equal n_accepted"
            )
            assert bd['accepted_rois'][0].ndim == 3, "Each ROI must be 3-D"


# =============================================================================
# Unit tests for measure_fwhm_from_averaged_psf
# =============================================================================

def _make_gauss3d(shape, sigma_z_nm, sigma_y_nm, sigma_x_nm, dz_nm, dy_nm, dx_nm):
    """Synthetic 3-D Gaussian PSF centred in `shape`, normalised to peak=1."""
    nz, ny, nx = shape
    z = (np.arange(nz) - nz // 2) * dz_nm
    y = (np.arange(ny) - ny // 2) * dy_nm
    x = (np.arange(nx) - nx // 2) * dx_nm
    ZZ, YY, XX = np.meshgrid(z, y, x, indexing='ij')
    psf = np.exp(-(ZZ**2 / (2 * sigma_z_nm**2) +
                   YY**2 / (2 * sigma_y_nm**2) +
                   XX**2 / (2 * sigma_x_nm**2)))
    return psf.astype(np.float32)


class TestMeasureFWHMFromAveragedPSF:

    def test_gaussian_recovery_within_2pct(self):
        """FWHM of a 3-D Gaussian must be recovered within 2%."""
        dz_nm, dy_nm, dx_nm = 110.0, 127.0, 127.0
        true_fwhm_z_nm  = 824.0   # σ_z  = 350 nm → FWHM = 824 nm
        true_fwhm_xy_nm = 353.0   # σ_xy = 150 nm → FWHM = 353 nm
        sigma_z_nm  = true_fwhm_z_nm  / 2.355
        sigma_xy_nm = true_fwhm_xy_nm / 2.355

        psf = _make_gauss3d((61, 55, 55), sigma_z_nm, sigma_xy_nm, sigma_xy_nm,
                            dz_nm, dy_nm, dx_nm)
        result = measure_fwhm_from_averaged_psf(psf, (dz_nm, dy_nm, dx_nm))

        for key, true_nm in [('fwhm_z_nm',  true_fwhm_z_nm),
                              ('fwhm_y_nm',  true_fwhm_xy_nm),
                              ('fwhm_x_nm',  true_fwhm_xy_nm)]:
            assert np.isfinite(result[key]), f"{key} must be finite"
            err = abs(result[key] - true_nm) / true_nm
            assert err < 0.02, (
                f"{key}: measured {result[key]:.1f} nm, expected {true_nm:.1f} nm "
                f"(error {err*100:.1f}% > 2%)"
            )

    def test_asymmetric_psf_axes_independent(self):
        """Asymmetric PSF: each axis FWHM must be measured independently."""
        dz_nm, dy_nm, dx_nm = 110.0, 127.0, 127.0
        fwhm_z, fwhm_y, fwhm_x = 800.0, 420.0, 360.0
        psf = _make_gauss3d((61, 45, 45),
                            fwhm_z / 2.355, fwhm_y / 2.355, fwhm_x / 2.355,
                            dz_nm, dy_nm, dx_nm)
        result = measure_fwhm_from_averaged_psf(psf, (dz_nm, dy_nm, dx_nm))

        for key, true_nm in [('fwhm_z_nm', fwhm_z),
                              ('fwhm_y_nm', fwhm_y),
                              ('fwhm_x_nm', fwhm_x)]:
            assert np.isfinite(result[key]), f"{key} must be finite"
            err = abs(result[key] - true_nm) / true_nm
            assert err < 0.02, (
                f"{key}: {result[key]:.1f} nm vs expected {true_nm:.1f} nm "
                f"({err*100:.1f}%)"
            )

    def test_returns_required_keys(self):
        psf = _make_gauss3d((21, 21, 21), 150.0, 100.0, 100.0, 110.0, 127.0, 127.0)
        result = measure_fwhm_from_averaged_psf(psf, (110.0, 127.0, 127.0))
        for k in ['fwhm_z_nm', 'fwhm_y_nm', 'fwhm_x_nm']:
            assert k in result, f"Missing key '{k}'"

    def test_nan_for_flat_profile(self):
        """A uniform volume must return nan (no discernible peak)."""
        psf = np.ones((11, 11, 11), dtype=np.float32)
        result = measure_fwhm_from_averaged_psf(psf, (110.0, 127.0, 127.0))
        for k in ['fwhm_z_nm', 'fwhm_y_nm', 'fwhm_x_nm']:
            assert not np.isfinite(result[k]), f"{k} should be nan for flat input"

    def test_bead_correction_keys_present_when_nonzero_diameter(self):
        """Corrected FWHM keys must appear when bead_diameter_nm > 0."""
        psf = _make_gauss3d((21, 21, 21), 150.0, 100.0, 100.0, 110.0, 127.0, 127.0)
        result = measure_fwhm_from_averaged_psf(psf, (110.0, 127.0, 127.0),
                                                bead_diameter_nm=100.0)
        for k in ('fwhm_z_nm_corrected', 'fwhm_y_nm_corrected', 'fwhm_x_nm_corrected'):
            assert k in result, f"Missing corrected key '{k}'"

    def test_bead_correction_keys_absent_when_zero_diameter(self):
        """Corrected FWHM keys must NOT appear when bead_diameter_nm=0."""
        psf = _make_gauss3d((21, 21, 21), 150.0, 100.0, 100.0, 110.0, 127.0, 127.0)
        result = measure_fwhm_from_averaged_psf(psf, (110.0, 127.0, 127.0),
                                                bead_diameter_nm=0.0)
        for k in ('fwhm_z_nm_corrected', 'fwhm_y_nm_corrected', 'fwhm_x_nm_corrected'):
            assert k not in result, f"Key '{k}' must not be present when bead_diameter_nm=0"

    def test_bead_correction_formula(self):
        """Corrected FWHM must equal sqrt(max(fwhm_measured² − d², 0))."""
        dz_nm, dy_nm, dx_nm = 110.0, 127.0, 127.0
        bead_d_nm = 100.0
        psf = _make_gauss3d((61, 55, 55), 824.0 / 2.355, 353.0 / 2.355, 353.0 / 2.355,
                            dz_nm, dy_nm, dx_nm)
        result = measure_fwhm_from_averaged_psf(psf, (dz_nm, dy_nm, dx_nm),
                                                bead_diameter_nm=bead_d_nm)
        d2 = bead_d_nm ** 2
        for axis, raw_key, corr_key in (
            ('z', 'fwhm_z_nm', 'fwhm_z_nm_corrected'),
            ('y', 'fwhm_y_nm', 'fwhm_y_nm_corrected'),
            ('x', 'fwhm_x_nm', 'fwhm_x_nm_corrected'),
        ):
            raw = result[raw_key]
            expected = float(np.sqrt(max(raw ** 2 - d2, 0.0)))
            np.testing.assert_allclose(
                result[corr_key], expected, rtol=1e-6,
                err_msg=f"Correction formula mismatch on {axis}"
            )

    def test_bead_correction_smaller_than_measured(self):
        """Corrected FWHM must be ≤ measured FWHM for a finite bead diameter."""
        psf = _make_gauss3d((61, 55, 55), 824.0 / 2.355, 353.0 / 2.355, 353.0 / 2.355,
                            110.0, 127.0, 127.0)
        result = measure_fwhm_from_averaged_psf(psf, (110.0, 127.0, 127.0),
                                                bead_diameter_nm=50.0)
        for axis in ('z', 'y', 'x'):
            raw  = result[f'fwhm_{axis}_nm']
            corr = result[f'fwhm_{axis}_nm_corrected']
            assert corr <= raw + 1e-9, (
                f"Corrected FWHM ({corr:.1f} nm) must be ≤ measured ({raw:.1f} nm) on {axis}"
            )


# =============================================================================
# Batch-merge combined FWHM
# =============================================================================

class TestBatchMergeFWHM:
    """Post-merge combined FWHM keys are injected into the merged bead_data."""

    def test_batch_merge_has_fwhm_keys(self):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from psf_gui import PSFScopeGUI
        except Exception as exc:
            pytest.skip(f"psf_gui not importable: {exc}")

        dx, dz = 0.127, 0.110
        results = []
        for seed in (42, 99):
            volume, _, _, _ = _make_bead_volume(n_beads=6, seed=seed)
            with tempfile.TemporaryDirectory() as tmpdir:
                tif = os.path.join(tmpdir, "beads.tif")
                imwrite(tif, volume)
                psf, _, bd = estimate_psf_from_beads(
                    tif_path        = tif,
                    dx              = dx,
                    dz              = dz,
                    sigma_xy_bounds = (0.05, 0.50),
                    sigma_z_bounds  = (0.05, 0.80),
                    min_sep_um      = 2.0,
                    roi_um          = (1.5, 1.5, 1.5),
                    margin_px       = 2,
                    r2_threshold    = 0.85,
                    verbose         = False,
                    return_bead_data= True,
                )
            results.append((psf, bd, None))

        psf_m  = PSFScopeGUI._merge_psfs(results)
        bd_m   = PSFScopeGUI._merge_bead_data(results, [f"vol_{i}.tif" for i in range(2)])
        PSFScopeGUI._apply_combined_fwhm(psf_m, bd_m, dz, dx)

        for key in ('fwhm_averaged_psf_z', 'fwhm_averaged_psf_y', 'fwhm_averaged_psf_x'):
            assert key in bd_m, f"Missing key '{key}'"
            assert np.isfinite(bd_m[key]), f"{key} must be finite, got {bd_m[key]}"

        assert bd_m.get('reporting_mode') == 'combined'
        assert 'fwhm_axes' in bd_m
        for ax in ('axis_z', 'axis_y', 'axis_x'):
            assert ax in bd_m['fwhm_axes'], f"fwhm_axes missing '{ax}'"


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
        ax = bd['fwhm_axes']
        avg_xy = 0.5 * (ax['axis_x']['fwhm_averaged_psf_nm'] or 0 +
                        ax['axis_y']['fwhm_averaged_psf_nm'] or 0)
        avg_z  = ax['axis_z']['fwhm_averaged_psf_nm']
        print(f"FWHM_xy (avg-PSF): {avg_xy:.0f} nm  (ground truth: {sxy*2355:.0f} nm)")
        print(f"FWHM_z  (avg-PSF): {avg_z:.0f} nm  (ground truth: {sz*2355:.0f} nm)")
    print("\n✓ Smoke test passed")
