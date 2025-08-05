# Reynolds FEM Solver for Journal Bearing using 
#Model and assumptions from Constantinescus book Gas Lubrication
import numpy as np
from skfem import *
from skfem.helpers import dot, grad
from skfem.autodiff import *
from skfem.autodiff.helpers import *
import jax.numpy as jnp
import matplotlib.pyplot as plt
from skfem.visuals.matplotlib import draw
from skfem.autodiff.helpers import *
from skfem.utils import solver_iter_krylov, build_pc_ilu, condense
from scipy.sparse.linalg import gmres
# -------------------
# PARAMETERS
# -------------------

def solve_pressure_distribution(
    R=0.05,
    Lz=0.25,
    hp=0.0045,
    h0=30e-6,
    epsilon=-0.4,
    phi0=0.0,
    Kp=8e-16,
    pa=1e5,
    ps=7e5,
    nphi=20,
    nz=200,
    phi_min=0 ,
    phi_max=np.pi
):
    """
    Solves the simplified 2D Reynolds-Darcy pressure equation in (phi, z) coordinates.

    Parameters
    ----------
    R : float
        Journal radius (m)
    Lz : float
        Journal length in z-direction (m)
    hp : float
        Porous thickness (m)
    h0 : float
        Mean airgap (m)
    epsilon : float
        Eccentricity ratio
    phi0 : float
        Eccentricity angle (rad) — not used in this simplified model
    mu : float
        Viscosity (Pa·s)
    Kp : float
        Permeability (m²)
    pa : float
        Ambient pressure (Pa)
    ps : float
        Supply pressure (Pa)
    nphi : int
        Number of points in φ direction
    nz : int
        Number of points in z direction

    Returns
    -------
    p : np.ndarray
        Pressure field array of shape (nphi * nz,)
    mesh : skfem.MeshTri
        The FEM mesh
    basis : skfem.Basis
        The FEM basis
    """
    # Mesh in (phi, z)
    phi_vals = np.linspace(phi_min, phi_max, nphi)
    z_vals = np.linspace(-Lz / 2, Lz / 2, nz)

    mesh = MeshTri.init_tensor(phi_vals, z_vals).with_boundaries({
        'left': lambda x: x[1] == z_vals[0],   # z = -Lz/2
        'right': lambda x: x[1] == z_vals[-1]  # z = +Lz/2
    })

    element = ElementTriP1()
    basis = Basis(mesh, element)

    # Bilinear forms
    @BilinearForm
    def lhs(u, v, w):
        phi = w.x[0]
        h = h0 * (1 + epsilon * jnp.cos(phi))
        h3 = h ** 3

        z_term = -h3 * R * grad(u)[1] * grad(v)[1]
        phi_term = -h3 / R * grad(u)[0] * grad(v)[0]
        return z_term + phi_term

    @BilinearForm
    def rhs(u, v, w):
        phi = w.x[0]
        h = h0 * (1 + epsilon * jnp.cos(phi))
        h3 = h ** 3
        r2 = R + h
        C1 = 12 * Kp
        delta = r2 * jnp.log(1 + hp / r2)
        Kp_eff = C1 / delta
        return Kp_eff * R * u * v

    # Assemble system
    A = lhs.assemble(basis)
    M = rhs.assemble(basis)

    # Dirichlet BCs at z = ±Lz/2
    D_left = basis.get_dofs('left').all()
    D_right = basis.get_dofs('right').all()
    D = np.concatenate([D_left, D_right])

    x = basis.ones() * (ps**2 + pa**2) / 2.0
    x[D] = ps**2 - pa**2

    η = solve(*condense(A - M, x=x, D=D))
    p = np.sqrt(ps**2 - η)

    return p, mesh, basis
