"""
One-dimensional non-linear Reynolds equation solver in cylindrical coordinates.
"""

# %% Imports and Setup
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.special import i0

from skfem import *
from skfem.element import ElementQuad1
from skfem.autodiff import *
from skfem.autodiff.helpers import *
from skfem.visuals.matplotlib import plot, show

import openairbearing as ab

# %% Constants and Parameters
bearing = ab.CircularBearing(error=1e-6, error_type="quadratic")
mu = bearing.mu
kappa = bearing.kappa
hp = bearing.hp
ps = bearing.ps
pa = bearing.pa
h = 5e-6
nzs = [1, 5, 25, 50, 100]

# %% Helper Functions

def get_beta(bearing, h0):
    """
    Calculate the porous feeding parameter.

    Parameters
    ----------
    bearing : object
        CircularBearing object
    h0 : float
        Air gap height

    Returns
    -------
    float
        Porous feeding parameter
    """
    b = bearing
    beta = 6 * b.kappa * b.xa**2 / (b.hp * h0**3)
    return beta

def analytic_sol(bearing, ps, h0, xvals=None):
    """
    Calculate the analytic pressure solution.

    Parameters
    ----------
    bearing : object
        CircularBearing object
    ps : float
        Supply pressure
    h0 : float
        Air gap height
    xvals : array_like, optional
        Radial positions

    Returns
    -------
    np.ndarray
        Analytic pressure solution
    """
    b = bearing
    beta = get_beta(b, h0)
    if xvals is None:
        xvals = b.x
    p0 = (
        ps * (1 - (1 - b.pa**2 / ps**2) * i0(np.outer(xvals / b.xa, (2 * beta)**0.5)) / i0((2 * beta)**0.5)) ** 0.5
    )
    return p0.ravel()

@NonlinearForm
def porous(pp, pr, vp, vr, w):
    r = w.x[0]
    return -r * pp / mu * kappa * (pp.grad[0] * vp.grad[0] + pp.grad[1] * vp.grad[1])

@NonlinearForm
def reynolds(pp, pr, vp, vr, w):
    r = w.x[0]
    eps = 1e-8
    return (
        + r / eps * (pp - pr) * (vp - vr)
        - r * h**3 / (12 * mu) * pr * pr.grad[0] * vr.grad[0]
        - r * kappa / (2 * mu / h) * pp * dot(grad(pp), w.n) * vr
    )

def calculate_pressure_force(basis, p):
    @LinearForm
    def load_form(v, w):
        r = w.x[0]
        return 2 * np.pi * r * w.p * v
    pressure_field = basis.interpolate(p)
    W = load_form.assemble(basis, p=pressure_field).sum()
    return W

# %% Pre-allocate plot settings
plt.figure()
colors = cycle(['r', 'g', 'm', 'c', 'orange', 'darkblue', 'k'])
markers = cycle(['x', 'o', 's', '^', 'v', '+', '*'])

# %% Main Loop Over nz Values
for nz in nzs:
    nz_porous = nz
    nr = 60
    z_vals = np.concatenate([np.array([0.0, h]), np.linspace(h, h + hp, nz_porous + 1)[1:]])
    r_vals = np.linspace(0, bearing.xa, nr)
    m = MeshQuad.init_tensor(r_vals, z_vals).with_defaults()
    e = ElementQuad1() * ElementQuad1()
    basis = Basis(m, e)
    fbasis = basis.boundary('bottom')

    p = basis.ones() * (ps + pa) / 2.0
    D1 = np.setdiff1d(basis.get_dofs("top").all('u^1'), basis.get_dofs("left").all('u^1'))
    Dright = np.intersect1d(basis.get_dofs('right').all('u^2'), basis.get_dofs('bottom').all('u^2'))
    Dzero = np.setdiff1d(basis.get_dofs(elements=True).all('u^2'), basis.get_dofs('bottom').all('u^2'))
    p[Dright] = pa
    p[D1] = ps
    p[Dzero] = 0
    D = np.concatenate((D1, Dright, Dzero))

    for itr in range(30):
        J1, rhs1 = porous.assemble(basis, x=p)
        J2, rhs2 = reynolds.assemble(fbasis, x=p)
        dp = solve(*condense(J1 + J2, rhs1 + rhs2, D=D))
        p_prev = p.copy()
        p += 0.9 * dp
        print(np.linalg.norm(p - p_prev))
        if np.linalg.norm(p - p_prev) < 1:
            break

    (pp, ppbasis), (pr, prbasis) = basis.split(p)
    print(pp.shape, pr.shape)

    x_plot = np.linspace(0, bearing.xa, 100)
    p_ref = analytic_sol(bearing, ps, h, xvals=x_plot)

    dofs_u2 = basis.get_dofs('bottom').all('u^2')
    p_num = p[dofs_u2]
    x_num = basis.doflocs[0][dofs_u2]

    color = next(colors)
    marker = next(markers)
    plt.plot(x_num, p_num, color=color, linestyle='-', label=f"Numeric (nr = {nr},nz = {nz})")
    plt.scatter(x_num, p_num, color=color, marker=marker, s=20, label=f"DOFs (nr = {nr},nz = {nz})")

plt.plot(x_plot, p_ref, "b--", label="Analytic (100 pts)")
plt.xlabel("x")
plt.ylabel("Pressure")
plt.legend()
plt.grid(True)
plt.title("Pressure Comparison (Constant Penalty Parameter 1e-8)")
plt.tight_layout()
plt.show()

fig1 = ppbasis.plot(pp, colorbar=True, shading='gouraud')
plt.title(f"Porous Pressure Field (nz = {nz})")
fig2 = prbasis.plot(pr, colorbar=True, shading='gouraud')
plt.title(f"Reynolds Pressure Field (nz = {nz})")
plt.show()


# if __name__ == "__main__":
#     from skfem.visuals.matplotlib import plot, show

#     p_ref = analytic_sol(bearing, ps, h)

#     W = calculate_pressure_force(basis, p)
#     print(f"Total load: {W:.2f} N")

#     fig, ax = plt.subplots()

#     plot(basis, p, ax=ax, Nrefs=0, color="red", label="Numeric")
#     plt.plot(bearing.x, p_ref, "b--", label="Analytic")



#     show()