"""
Generate a synthetic bead volume for manual GUI testing of psfScope.

Writes a uint16 TIFF to the same directory as this script.
The volume contains 3-D Gaussians at random positions with a configurable
PSF size, noise level, and number of beads.

Usage
-----
    python generate_test_beads.py
    python generate_test_beads.py --n-beads 20 --noise 40 --output my_beads.tif
"""

import argparse
import os

import numpy as np
from tifffile import imwrite


def make_bead_volume(
    shape=(80, 256, 256),
    n_beads=15,
    sigma_xy_um=0.15,
    sigma_z_um=0.35,
    dx=0.127,
    dz=0.110,
    amplitude=3000,
    noise_std=30,
    seed=42,
):
    """
    Generate a ZYX volume with *n_beads* anisotropic 3-D Gaussians.

    Parameters
    ----------
    shape       : (nz, ny, nx) — volume size in pixels
    n_beads     : number of beads to place
    sigma_xy_um : lateral Gaussian sigma [µm]
    sigma_z_um  : axial Gaussian sigma [µm]
    dx          : lateral pixel size [µm]
    dz          : axial voxel size [µm]
    amplitude   : peak intensity above background
    noise_std   : standard deviation of additive Gaussian noise
    seed        : random seed for reproducibility

    Returns
    -------
    volume      : np.ndarray, uint16, shape = (nz, ny, nx)
    positions   : list of (z, y, x) in pixels
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    sigma_z_px  = sigma_z_um  / dz
    sigma_xy_px = sigma_xy_um / dx

    volume = rng.normal(0, noise_std, shape).astype(np.float32)

    margin_xy = max(20, int(4 * sigma_xy_px))
    margin_z  = max(10, int(4 * sigma_z_px))
    positions = []

    # Minimum bead separation: 4 µm in the lateral plane
    min_sep_px = 4.0 / dx

    attempts = 0
    while len(positions) < n_beads and attempts < 10_000:
        attempts += 1
        cz = int(rng.integers(margin_z,  nz - margin_z))
        cy = int(rng.integers(margin_xy, ny - margin_xy))
        cx = int(rng.integers(margin_xy, nx - margin_xy))
        ok = all(
            np.sqrt((cy - py) ** 2 + (cx - px) ** 2) > min_sep_px
            for _, py, px in positions
        )
        if ok:
            positions.append((cz, cy, cx))

    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    for cz, cy, cx in positions:
        volume += amplitude * np.exp(
            -(((zz - cz) / sigma_z_px)  ** 2 +
              ((yy - cy) / sigma_xy_px) ** 2 +
              ((xx - cx) / sigma_xy_px) ** 2) / 2.0
        )

    volume = np.clip(volume, 0, None).astype(np.uint16)
    return volume, positions


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        description="Generate a synthetic bead TIFF for psfScope GUI testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output",    default=os.path.join(here, "test_beads.tif"),
                   help="Output TIFF path")
    p.add_argument("--shape",     type=int, nargs=3, default=[80, 256, 256],
                   metavar=("NZ", "NY", "NX"), help="Volume shape in pixels")
    p.add_argument("--n-beads",   type=int,   default=15, help="Number of beads")
    p.add_argument("--sigma-xy",  type=float, default=0.15, help="σ_xy [µm]")
    p.add_argument("--sigma-z",   type=float, default=0.35, help="σ_z  [µm]")
    p.add_argument("--dx",        type=float, default=0.127, help="Lateral pixel size [µm]")
    p.add_argument("--dz",        type=float, default=0.110, help="Axial voxel size [µm]")
    p.add_argument("--amplitude", type=int,   default=3000,  help="Bead peak intensity")
    p.add_argument("--noise",     type=float, default=30,    help="Noise std")
    p.add_argument("--seed",      type=int,   default=42,    help="Random seed")
    args = p.parse_args()

    print(f"Generating {args.n_beads} beads in volume {tuple(args.shape)} ...")
    volume, positions = make_bead_volume(
        shape       = tuple(args.shape),
        n_beads     = args.n_beads,
        sigma_xy_um = args.sigma_xy,
        sigma_z_um  = args.sigma_z,
        dx          = args.dx,
        dz          = args.dz,
        amplitude   = args.amplitude,
        noise_std   = args.noise,
        seed        = args.seed,
    )

    imwrite(args.output, volume, imagej=True, metadata={'axes': 'ZYX'})

    print(f"Placed beads: {len(positions)}")
    print(f"True FWHM_xy: {args.sigma_xy * 2355:.0f} nm")
    print(f"True FWHM_z : {args.sigma_z  * 2355:.0f} nm")
    print(f"Saved → {args.output}")
    print()
    print("Suggested psfScope parameters for this volume:")
    print(f"  dx = {args.dx}   dz = {args.dz}")
    print(f"  ROI Z/Y/X = 1.5 µm   Min separation = 3.0 µm")
    print(f"  Best fraction = 0.8   Threshold = auto")


if __name__ == "__main__":
    main()
