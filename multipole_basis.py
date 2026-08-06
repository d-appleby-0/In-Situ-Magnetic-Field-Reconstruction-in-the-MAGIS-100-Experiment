"""
multipole_basis.py
==================
Pure mathematical machinery for cylindrical multipole expansion of the
magnetic field inside the MAGIS-100 vacuum tube.

Physics
-------
In a current-free region, H satisfies Laplace's equation. In cylindrical
coordinates (r, phi, z) the 2D transverse solution at each z-slice is,
following g-2 Eq. (6), normalised by the tube inner radius r0:

    Hx(r,phi) = sum_{n=0}^{N} (r/r0)^n [ Cn cos(n*phi) - Sn sin(n*phi) ]
    Hy(r,phi) = sum_{n=0}^{N} (r/r0)^n [ Cn sin(n*phi) + Sn cos(n*phi) ]
    Hz         = fitted independently (axial component)

Normalising by r0 (default: tube inner radius, 2.75 in) means every
coefficient Cn, Sn comes out in the same units (A/m) regardless of order n,
so mode strengths can be compared directly (as in g-2 Table V) and the
design matrix doesn't mix numerically tiny/huge scales.

The coefficient vector for one slice is:
    theta = [C0, C1, S1, C2, S2, ..., Cn, Sn, Hz_const]
    length = 2*N + 2

n=0: uniform, n=1: dipole, n=2: quadrupole, n=3: sextupole,
n=4: octupole, n=5: decapole, n=6: dodecapole (g-2 tracked to this order)
"""

import warnings
import numpy as np
from typing import Sequence

R0_DEFAULT = 2.75  # inches — MAGIS-100 vacuum tube inner radius


def cart_to_cyl(x: float, y: float) -> tuple[float, float]:
    return np.sqrt(x*x + y*y), np.arctan2(y, x) % (2 * np.pi)


def unit_vector(theta_deg: float, phi_deg: float) -> np.ndarray:
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    return np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])


def phi_x_row(x: float, y: float, n_max: int, r0: float = R0_DEFAULT) -> np.ndarray:
    zeta = complex(x, y) / r0
    row = np.zeros(2 * n_max + 2)
    row[0] = 1.0
    for n in range(1, n_max + 1):
        zn = zeta ** n
        idx = 2*n - 1
        row[idx]   =  zn.real
        row[idx+1] = -zn.imag
    row[-1] = 0.0
    return row


def phi_y_row(x: float, y: float, n_max: int, r0: float = R0_DEFAULT) -> np.ndarray:
    zeta = complex(x, y) / r0
    row = np.zeros(2 * n_max + 2)
    row[0] = 0.0
    for n in range(1, n_max + 1):
        zn = zeta ** n
        idx = 2*n - 1
        row[idx]   = zn.imag
        row[idx+1] = zn.real
    row[-1] = 0.0
    return row


def phi_z_row(n_max: int) -> np.ndarray:
    row = np.zeros(2 * n_max + 2)
    row[-1] = 1.0
    return row

# ── Transverse derivative basis rows (Part 2a: gradient / uniformity) ────────
# Hx, Hy are Re/Im parts of an analytic function of zeta=(x+iy)/r0, so their
# x,y derivatives follow directly from d(zeta^n)/dx = n*zeta^(n-1)/r0 and
# d(zeta^n)/dy = i*n*zeta^(n-1)/r0. These give the transverse gradient of the
# ALREADY-FITTED field (theta) at any point -- no new fitting required.
 
def phi_x_dx_row(x: float, y: float, n_max: int, r0: float = R0_DEFAULT) -> np.ndarray:
    zeta = complex(x, y) / r0
    row = np.zeros(2 * n_max + 2)
    for n in range(1, n_max + 1):
        w = zeta ** (n - 1)
        idx = 2*n - 1
        factor = n / r0
        row[idx]   =  factor * w.real
        row[idx+1] = -factor * w.imag
    return row
 
 
def phi_x_dy_row(x: float, y: float, n_max: int, r0: float = R0_DEFAULT) -> np.ndarray:
    zeta = complex(x, y) / r0
    row = np.zeros(2 * n_max + 2)
    for n in range(1, n_max + 1):
        w = zeta ** (n - 1)
        idx = 2*n - 1
        factor = n / r0
        row[idx]   = -factor * w.imag
        row[idx+1] = -factor * w.real
    return row
 
 
def phi_y_dx_row(x: float, y: float, n_max: int, r0: float = R0_DEFAULT) -> np.ndarray:
    zeta = complex(x, y) / r0
    row = np.zeros(2 * n_max + 2)
    for n in range(1, n_max + 1):
        w = zeta ** (n - 1)
        idx = 2*n - 1
        factor = n / r0
        row[idx]   = factor * w.imag
        row[idx+1] = factor * w.real
    return row
 
 
def phi_y_dy_row(x: float, y: float, n_max: int, r0: float = R0_DEFAULT) -> np.ndarray:
    zeta = complex(x, y) / r0
    row = np.zeros(2 * n_max + 2)
    for n in range(1, n_max + 1):
        w = zeta ** (n - 1)
        idx = 2*n - 1
        factor = n / r0
        row[idx]   =  factor * w.real
        row[idx+1] = -factor * w.imag
    return row
 
 
def reconstruct_gradient(theta: np.ndarray, target_xy: Sequence[tuple],
                         n_max: int, r0: float = R0_DEFAULT) -> np.ndarray:
    """
    Transverse gradient tensor at each target point, using the SAME fitted
    theta as the field reconstruction (no new fitting/data needed):
 
        G[i] = [[dHx/dx, dHx/dy],
                [dHy/dx, dHy/dy]]
 
    Hz is treated as uniform within a slice by this basis (the z-gradient
    is a separate, axial question -- see the project notes on per-pair vs
    cross-pair axial trend estimation).
    """
    G = np.zeros((len(target_xy), 2, 2))
    for i, (x, y) in enumerate(target_xy):
        G[i, 0, 0] = phi_x_dx_row(x, y, n_max, r0=r0) @ theta
        G[i, 0, 1] = phi_x_dy_row(x, y, n_max, r0=r0) @ theta
        G[i, 1, 0] = phi_y_dx_row(x, y, n_max, r0=r0) @ theta
        G[i, 1, 1] = phi_y_dy_row(x, y, n_max, r0=r0) @ theta
    return G
 
 
# ── Plug-and-play measurement -> field composition (Part 1) ─────────────────
 
def build_measurement_to_field_map(A: np.ndarray, target_xy: Sequence[tuple],
                                   n_max: int, r0: float = R0_DEFAULT,
                                   rcond: float = 1e-10) -> dict:
    """
    Compose the fit (theta = pinv(A) @ m) with field reconstruction
    (H(target) = Phi(target) @ theta) into a single matrix per target point,
    so the reader can go directly from raw sensor readings to field value
    without an intermediate fitting step:
 
        Hx(target) = M_x @ m,   Hy(target) = M_y @ m,   Hz(target) = M_z @ m
 
    Returns
    -------
    dict with M_x, M_y, M_z ((n_targets, n_measurements) arrays), the pinv(A)
    used, and dof_effective (rank estimate) so callers can flag rank-deficient
    (non-unique, minimum-norm) cases.
    """
    J = np.linalg.pinv(A, rcond=rcond)
    Phi_x = np.vstack([phi_x_row(x, y, n_max, r0=r0) for x, y in target_xy])
    Phi_y = np.vstack([phi_y_row(x, y, n_max, r0=r0) for x, y in target_xy])
    Phi_z = np.tile(phi_z_row(n_max), (len(target_xy), 1))
    M_x = Phi_x @ J
    M_y = Phi_y @ J
    M_z = Phi_z @ J
    sv = np.linalg.svd(A, compute_uv=False)
    dof = int(np.sum(sv > rcond * sv[0])) if len(sv) else 0
    return {"M_x": M_x, "M_y": M_y, "M_z": M_z, "J": J,
            "dof_effective": dof, "n_coeffs": A.shape[1]}


def sensor_basis_row(x: float, y: float, u: np.ndarray, n_max: int,
                     r0: float = R0_DEFAULT) -> np.ndarray:
    px = phi_x_row(x, y, n_max, r0=r0)
    py = phi_y_row(x, y, n_max, r0=r0)
    pz = phi_z_row(n_max)
    return u[0]*px + u[1]*py + u[2]*pz


def check_design_matrix_rank(n_measurements: int, n_coeffs: int) -> None:
    if n_coeffs <= 0:
        return
    ratio = n_measurements / n_coeffs
    if n_measurements < n_coeffs:
        warnings.warn(
            f"Underdetermined system: {n_measurements} measurements for "
            f"{n_coeffs} coefficients (ratio={ratio:.2f}). No unique solution "
            f"exists without regularisation — reduce n_max, add sensors, or "
            f"use a regularised/pseudoinverse fit.", stacklevel=3)
    elif ratio < 2:
        warnings.warn(
            f"Poorly conditioned system: {n_measurements} measurements for "
            f"{n_coeffs} coefficients (ratio={ratio:.2f} < 2). Consider more "
            f"sensors, fewer multipole orders, or regularisation.", stacklevel=3)


def build_design_matrix(sensor_positions: Sequence[tuple],
                        sensor_orientations: Sequence,
                        n_max: int,
                        r0: float = R0_DEFAULT) -> np.ndarray:
    rows = []
    for (x, y), orient in zip(sensor_positions, sensor_orientations):
        if orient is None:
            axes = np.eye(3)
        elif np.array(orient).ndim == 1:
            axes = np.array(orient).reshape(1, 3)
        else:
            axes = np.array(orient)
        for u in axes:
            rows.append(sensor_basis_row(x, y, u, n_max, r0=r0))
    A = np.vstack(rows)
    check_design_matrix_rank(A.shape[0], 2 * n_max + 2)
    return A


def reconstruct_field(theta: np.ndarray, target_xy: Sequence[tuple],
                      n_max: int, r0: float = R0_DEFAULT) -> np.ndarray:
    H = np.zeros((len(target_xy), 3))
    for i, (x, y) in enumerate(target_xy):
        H[i, 0] = phi_x_row(x, y, n_max, r0=r0) @ theta
        H[i, 1] = phi_y_row(x, y, n_max, r0=r0) @ theta
        H[i, 2] = phi_z_row(n_max)               @ theta
    return H


def reconstruct_magnitude(theta: np.ndarray, target_xy: Sequence[tuple],
                          n_max: int, r0: float = R0_DEFAULT) -> np.ndarray:
    H = reconstruct_field(theta, target_xy, n_max, r0=r0)
    return np.linalg.norm(H, axis=1)


def coefficient_names(n_max: int) -> list[str]:
    order_names = {0: 'uniform', 1: 'dipole', 2: 'quad', 3: 'sextupole',
                   4: 'octupole', 5: 'decapole', 6: 'dodecapole'}
    names = [f"C0({order_names.get(0,'n=0')})"]
    for n in range(1, n_max + 1):
        label = order_names.get(n, f'n={n}')
        names += [f"C{n}({label})", f"S{n}({label})"]
    names.append("Hz(axial)")
    return names


if __name__ == "__main__":
    import warnings as _w
    print("=== multipole_basis self-test ===\n")
    n_max = 2
    r0 = R0_DEFAULT
    names = coefficient_names(n_max)
    print(f"Coefficient names (n_max={n_max}): {names}")
    print(f"r0 = {r0} in\n")

    theta_uniform = np.zeros(2*n_max + 2); theta_uniform[0] = 1.0
    pts = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
    H = reconstruct_field(theta_uniform, pts, n_max, r0=r0)
    print("Uniform field (C0=1): Hx=1 everywhere")
    for (x,y), h in zip(pts, H):
        print(f"  ({x:+.1f},{y:+.1f}): Hx={h[0]:.4f} Hy={h[1]:.4f} Hz={h[2]:.4f}")

    theta_dipole = np.zeros(2*n_max + 2); theta_dipole[1] = 1.0
    H2 = reconstruct_field(theta_dipole, pts, n_max, r0=r0)
    print(f"\nDipole field (C1=1, r0={r0}): Hx=x/r0, Hy=y/r0")
    for (x,y), h in zip(pts, H2):
        print(f"  ({x:+.1f},{y:+.1f}): Hx={h[0]:.4f}(expect {x/r0:.4f}) Hy={h[1]:.4f}(expect {y/r0:.4f})")

    names6 = coefficient_names(6)
    print(f"\nn_max=6 names ({len(names6)} total): {names6}")
    assert len(names6) == 14

    print("\n── Rank check tests ──")
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        build_design_matrix([(-4.,0.),(0.,4.)], [None,None], n_max=3, r0=r0)
        for x in w: print(f"  WARNING: {x.message}")

    print("\n── Gradient self-test (analytic vs finite difference) ──")
    rng = np.random.default_rng(0)
    n_max_g = 3
    theta_rand = rng.normal(size=2*n_max_g+2)
    test_pt = (0.9, -0.4)
    eps = 1e-6
    G_analytic = reconstruct_gradient(theta_rand, [test_pt], n_max_g, r0=R0_DEFAULT)[0]
    x0, y0 = test_pt
    Hx0 = reconstruct_field(theta_rand, [(x0,y0)], n_max_g, r0=R0_DEFAULT)[0]
    Hx_px = reconstruct_field(theta_rand, [(x0+eps,y0)], n_max_g, r0=R0_DEFAULT)[0]
    Hx_py = reconstruct_field(theta_rand, [(x0,y0+eps)], n_max_g, r0=R0_DEFAULT)[0]
    G_fd = np.array([[(Hx_px[0]-Hx0[0])/eps, (Hx_py[0]-Hx0[0])/eps],
                     [(Hx_px[1]-Hx0[1])/eps, (Hx_py[1]-Hx0[1])/eps]])
    max_err = np.max(np.abs(G_analytic - G_fd))
    print(f"  analytic:\n{G_analytic}")
    print(f"  finite-diff:\n{G_fd}")
    print(f"  max abs error: {max_err:.2e}  {'✓ PASS' if max_err < 1e-4 else '✗ FAIL'}")
    assert max_err < 1e-4, "Gradient basis rows do not match finite-difference check!"

    print("\n✓ All basis tests passed")
