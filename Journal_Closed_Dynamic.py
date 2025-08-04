# Coupled Time Dependent Reynolds–Darcy FEM Solver 
###TODO Implement Rotordynamic spring effect KX and CX_dot and air film damping to dynamic equation 

import numpy as np
from skfem import *
from skfem.helpers import dot, grad
from skfem.autodiff import *
from skfem.autodiff.helpers import *
import jax.numpy as jnp
import matplotlib.pyplot as plt
from skfem.visuals.matplotlib import draw
from skfem.utils import solver_iter_krylov, build_pc_ilu, condense
from scipy.sparse.linalg import gmres
from pyamg import smoothed_aggregation_solver, ruge_stuben_solver
from scipy.linalg import eigvals

import time
import sys
import os

# Add parent directories to Python path for local imports


# ----------------------------------------------------------------------------
# Configuration Parameters
# ----------------------------------------------------------------------------

# Solver options
SOLVER = 'Krylov_gmres'         # Options: 'basic', 'Krylov_gmres'
PRECONDITIONER = 'RUBEN'        # Options: 'ILU', 'AMG', 'RUBEN', 'JACOBI'

# Geometry and physics
PHI_PERIODIC = True             # Enable/disable periodicity in phi-direction
R = 0.05                        # Journal radius [m]
Lz = 0.25                       # Journal length in z-direction [m]
hp = 0.0045                     # Thickness of the porous layer [m]
h0 = 30e-6                      # Mean air gap height [m]
epsilon = -0.2                  # Eccentricity ratio
phi0 = 0                        # Eccentricity orientation [rad]
mu = 1.8e-5                     # Dynamic viscosity [Pa.s]
Kp = 8e-16                      # Permeability of the porous layer [m^2]

# Rotational speeds [RPM]
RPM_journal = 0
RPM_bearing = 0
omega = RPM_journal / (2 * np.pi) / 60  # Angular speed [rad/s]
Omega = RPM_bearing / (2 * np.pi) / 60
Us = R * (omega + Omega)                       # Surface speed [m/s]
Vz = 0                                  # Axial velocity (assumed zero)

# Pressures
pa = 1e5                                # Ambient pressure [Pa]
ps = 7e5                                # Supply pressure [Pa]

# ----------------------------------------------------------------------------
# External Excitation Parameters (used for dynamic forcing)
# ----------------------------------------------------------------------------

# Mesh discretization
n_r = 11                                # Number of radial layers (incl. airgap)
n_phi = 40                              # Azimuthal resolution
n_z = 11                                # Axial resolution

# Newton solver parameters
maxiterNewton = 50

# Initial Newton tolerances (static and dynamic)
toleranceNewton_static_init = 1e-4

# Dynamic solve tolerances (during rigid body motion)
toleranceNewton_dynamic_init = 2       # High tolerance for initial transient solves
toleranceNewton = 2                    # Global Newton residual tolerance

# Dual-stopping dynamic convergence criteria  
tol_dp_inf = 1e-3
tol_res = 1e-6
eps_p = 1e-12

# Time stepping configuration
Nt = 5000                              # Number of time steps
dt = 1e-5                              # Time step size [s]
#time step 1e-5 ok wrt to film timescale?CFL condition not calculated, 1e-3 has some definitely too large esp without implicit time integration

# Sinusoidal excitation force applied in the y-direction.
# These can be adjusted or extended to other force models (e.g. chirp, square wave, etc.).
f = 500  # Excitation frequency [Hz]
A = 4    # Amplitude [N]; peak force applied in sinusoidal forcing
step_excitation=5

# Penalty method coupling terms (interface + periodic)
eps_coupling = 1e-5                    # Coupling penalty between porous and Reynolds
eps_periodic = 1 / R * eps_coupling    # Penalty for enforcing phi-periodicity 
#this should probably made to same order to avoid illconditioning


# ----------------------------------------------------------------------------
# Radial Mesh Generation with Geometric Grading
# ----------------------------------------------------------------------------

def make_rvals_with_fixed_first_layer(R, h0, hp, N):
    """
    Create radial coordinates for a layered mesh, preserving the first layer as h0
    and distributing the remaining thickness hp geometrically.

    Parameters
    ----------
    R : float
        Inner radius of the bearing (journal radius).
    h0 : float
        Thickness of the first layer (air gap).
    hp : float
        Total thickness of the porous layer.
    N : int
        Total number of layers (including the air gap).

    Returns
    -------
    r_vals : np.ndarray
        Array of radial coordinates.
    dzs : np.ndarray
        Thickness of each layer.
    """
    N_remaining = N - 1
    h_remaining = hp

    # Search for geometric growth factor
    g_low, g_high = 1.01, 100.0
    for _ in range(100):
        g = (g_low + g_high) / 2
        dzs_rest = h0 * g ** np.arange(N_remaining)
        total = dzs_rest.sum()
        if total < h_remaining:
            g_low = g
        else:
            g_high = g
        if abs(total - h_remaining) < 1e-12:
            break

    dzs_rest *= h_remaining / dzs_rest.sum()
    dzs = np.concatenate([[h0], dzs_rest])
    r_vals = R + np.cumsum(np.concatenate([[0], dzs]))

    return r_vals, dzs

# Generate radial mesh points and thicknesses
r_vals, dzs = make_rvals_with_fixed_first_layer(R, h0, hp, n_r)
phi_vals = np.linspace(0, 2 * np.pi, n_phi)
z_vals = np.linspace(-Lz / 2, Lz / 2, n_z)

# ----------------------------------------------------------------------------
# Mesh Definition and Boundary Labeling
# ----------------------------------------------------------------------------

# Define boundary functions for labeling mesh facets
def boundary_top(x):
    return np.isclose(x[0], R + hp + h0)

def boundary_bottom(x):
    return np.isclose(x[0], R)

def boundary_left(x):
    return np.isclose(x[2], z_vals[0])  # z = -Lz/2

def boundary_right(x):
    return np.isclose(x[2], z_vals[-1])  # z = +Lz/2

# Create 3D tensor product mesh and assign boundary markers
mesh = MeshTet.init_tensor(r_vals, phi_vals, z_vals).with_boundaries({
    'reynolds_surface': lambda x: np.isclose(x[0], R, atol=1e-9),
    'top': boundary_top,
    'bottom': boundary_bottom,
    'left': boundary_left,
    'right': boundary_right,
    'phi0': lambda x: np.isclose(x[1], 0.0, atol=1e-10),
    'phi2pi': lambda x: np.isclose(x[1], 2 * np.pi, atol=1e-10),
})

# Define element type for each component (P1 scalar elements)
element = ElementTetP1() * ElementTetP1()

# Build basis over entire mesh for both pressure fields (porous and Reynolds)
basis = Basis(mesh, element)

# Identify Reynolds surface (bottom facet of porous region)
bottom_facets = mesh.boundaries['bottom']
## Build boundary basis used for Reynolds Jacobian assembly
# (Note: prbasis here is not interpolated from a split)
fbasis = basis.boundary('bottom')

# This fbasis is used for Reynolds Jacobian assembly.
# It does not share the same structure as a basis derived from a split solution.
# Caution: projecting split pressure fields directly to this fbasis may be incompatible.

# ----------------------------------------------------------------------------
# Nonlinear Weak Forms: Porous and Reynolds Layers
# ----------------------------------------------------------------------------

@NonlinearForm
def porous(pp, pr, vp, vr, w):
    """
    Nonlinear form for Darcy flow in the porous region.
    
    Parameters
    ----------
    pp, pr : Trial functions (pp: porous)
    vp, vr : Test functions (vp: porous)
    w : Autodiff context

    Returns
    -------
    Expression representing Darcy flow weak form.
    """
    r = w.x[0]
    return -r * pp / mu * Kp * (
        pp.grad[0] * vp.grad[0] +
        1 / r**2 * pp.grad[1] * vp.grad[1] +
        pp.grad[2] * vp.grad[2]
    )

@NonlinearForm
def reynolds(pp, pr, vp, vr, w):
    """
    Nonlinear form for Reynolds flow in the air gap.

    Includes Couette, Poiseuille, axial, pressure-coupling,
    and penalty terms for interface continuity.
    """
    r = w.x[0]
    phi = w.x[1]
    z = w.x[2]

    # Gap height based on eccentricity and angle
    h = h0 + epsilon * h0 * jnp.cos(phi - phi0)

    # Slip parameters 
    # where 𝛼 is a dimensionless sli p coefficient
    # that depends on the material characteristics of the permeable material
    # and not on the physical properties of the fluid (Beavers et al. 1970)
    # PHI and PSI Set to zero here for negligible slip
    #alpha normal value around 0.1
    alpha=0.1
    sqrt_kp = jnp.sqrt(Kp)
    PHI = 3 * (sqrt_kp * h + 2 * alpha * Kp) / (h * (sqrt_kp + alpha * h))
    PSI = sqrt_kp / (sqrt_kp + alpha * h) 
    PHI = 0.0
    PSI = 0.0

    # Journal surface tangential velocity
    Uphi = (omega + Omega) * R

    # Azimuthal (phi) terms: Couette + Poiseuille
    PhiTerm = (
        (Uphi * h / 2 * (1 + PSI)) * pr * vr.grad[1] -
        (h**3 / (12 * mu * r) * (1 + PHI)) * pr * pr.grad[1] * vr.grad[1]
    )

    # Axial (z) terms
    ZTerm = r * (
        Vz * h / 2 * (1 + PSI) * pr * vr.grad[2] -
        h**3 / (12 * mu) * (1 + PHI) * pr * pr.grad[2] * vr.grad[2]
    )

    # Coupling pressure gradient from porous region
    PressureGradientTerm = -r * Kp / (2 * mu / h) * pp * dot(grad(pp), w.n) * vr

    # Penalty enforcement of pp ≈ pr at interface
    PenaltyTerm = r / eps_coupling * (pp - pr) * (vp - vr)

    return PhiTerm + ZTerm + PressureGradientTerm + PenaltyTerm

# ----------------------------------------------------------------------------
# DOF Masking
# ----------------------------------------------------------------------------

# Identify Reynolds-layer DOFs (located at radius R)
r_coords = basis.doflocs[0]
phi_coords = basis.doflocs[1]
z_coords = basis.doflocs[2]
reynolds_dofs = np.where(np.isclose(r_coords, R))[0]

# ----------------------------------------------------------------------------
# Boundary Conditions and Initial Pressure Field
# ----------------------------------------------------------------------------

# Initialize pressure field with average of supply and ambient
p = basis.ones() * (ps + pa) / 2.0

# Dirichlet DOFs for porous pressure (u^1)
top1   = basis.get_dofs("top").all("u^1")
left1  = basis.get_dofs("left").all("u^1")
right1 = basis.get_dofs("right").all("u^1")

# Feed zone DOFs = top minus edges
D_feeding = np.setdiff1d(top1, np.union1d(left1, right1))

# Dirichlet DOFs for Reynolds layer (u^2) at left/right (z ends)
Dleft = np.intersect1d(
    basis.get_dofs('left').all('u^2'),
    basis.get_dofs('bottom').all('u^2')
)
Dright = np.intersect1d(
    basis.get_dofs('right').all('u^2'),
    basis.get_dofs('bottom').all('u^2')
)

# Kill Reynolds DOFs not on interface (R)
non_reynolds_dofs = np.setdiff1d(
    basis.get_dofs(elements=True).all('u^2'),
    basis.get_dofs('bottom').all('u^2')
)

# Combine all Dirichlet indices
D = np.concatenate((D_feeding, Dleft, Dright, non_reynolds_dofs))

# Assign Dirichlet values
p[D_feeding] = ps       # Supply pressure at feed
p[Dleft] = pa            # Ambient at left edge
p[Dright] = pa           # Ambient at right edge
p[non_reynolds_dofs] = 0 # Kill Reynolds everywhere else

# ----------------------------------------------------------------------------
# Periodicity Enforcement (φ = 0 ↔ φ = 2π)
# ----------------------------------------------------------------------------

from scipy.sparse import lil_matrix

if PHI_PERIODIC:
    # Retrieve DOFs along φ=0 and φ=2π boundaries
    phi0_dofs = basis.get_dofs('phi0').all()
    phi2pi_dofs = basis.get_dofs('phi2pi').all()

    # Create sorting keys to match φ=0 and φ=2π nodes
    val0 = basis.doflocs[2, phi0_dofs] + 0.1 * basis.doflocs[0, phi0_dofs]
    val2 = basis.doflocs[2, phi2pi_dofs] + 0.1 * basis.doflocs[0, phi2pi_dofs]

    # Sort the DOFs by a compound key to ensure 1-to-1 matching
    phi0_sorted = phi0_dofs[np.argsort(val0)]
    phi2pi_sorted = phi2pi_dofs[np.argsort(val2)]

    # Build sparse penalty matrix enforcing u(φ=0) ≈ u(φ=2π)
    P_lil = lil_matrix((basis.N, basis.N))

    # Diagonal contributions
    P_lil[phi0_sorted, phi0_sorted] = +1 / eps_periodic
    P_lil[phi2pi_sorted, phi2pi_sorted] = +1 / eps_periodic

    # Off-diagonal couplings
    P_lil[phi0_sorted, phi2pi_sorted] = -1 / eps_periodic
    P_lil[phi2pi_sorted, phi0_sorted] = -1 / eps_periodic

    # Convert to efficient CSR format for later addition
    P = P_lil.tocsr()


# ----------------------------------------------------------------------------
# Static Solve: Newton Iteration
# ----------------------------------------------------------------------------

if SOLVER == 'basic':
    for itr in range(maxiterNewton):
        t0 = time.perf_counter()

        # Assemble Jacobians and residuals
        J1, rhs1 = porous.assemble(basis, x=p)
        J2, rhs2 = reynolds.assemble(fbasis, x=p)

        # Combine into full Jacobian and residual
        J = J1 + J2
        rhs = rhs1 + rhs2

        if PHI_PERIODIC:
            J = J + P  # Add periodic penalty

        # Solve condensed system
        dp = solve(*condense(J, rhs, D=D))
        p_prev = p.copy()
        p += 0.9 * dp  # Relaxed update

        norm = np.linalg.norm(p - p_prev)
        print(norm)

        t1 = time.perf_counter()
        print(f"[Newton {itr}] Direct solve took {(t1 - t0)*1000:.2f} ms")

        if norm < toleranceNewton_static_init:
            break

    # Split pressure into porous and Reynolds parts
    (pp, ppbasis), (pr, prbasis) = basis.split(p)
    print(pp.shape, pr.shape)

    from skfem.utils import build_pc_ilu, build_pc_diag

elif SOLVER == 'Krylov_gmres':
    def make_preconditioner(Acond):
        if PRECONDITIONER == 'ILU':
            return build_pc_ilu(Acond)
        elif PRECONDITIONER == 'JACOBI':
            return build_pc_diag(Acond)
        elif PRECONDITIONER == 'AMG':
            ml = smoothed_aggregation_solver(Acond)
            return ml.aspreconditioner()
        elif PRECONDITIONER == 'RUBEN':
            ml = ruge_stuben_solver(Acond)
            return ml.aspreconditioner()
        else:
            raise ValueError(f"Unknown preconditioner: {PRECONDITIONER}")

    def make_solver(Acond):
        M = make_preconditioner(Acond)
        return solver_iter_krylov(
            krylov=gmres,
            M=M,
            rtol=1e-7,
            atol=1e-8,
            maxiter=1000,
            restart=500,
            verbose=False
        )

    for itr in range(maxiterNewton):
        t0 = time.perf_counter()

        # 🔹 Assemble porous system
        t_asm0 = time.perf_counter()
        J1, rhs1 = porous.assemble(basis, x=p)
        t_asm1 = time.perf_counter()
        print(f"[Newton {itr}] ⏳ porous.assemble      = {(t_asm1 - t_asm0)*1e3:.2f} ms")

        # 🔹 Assemble Reynolds system
        t_asm2 = time.perf_counter()
        J2, rhs2 = reynolds.assemble(fbasis, x=p)
        t_asm3 = time.perf_counter()
        print(f"[Newton {itr}] ⏳ reynolds.assemble    = {(t_asm3 - t_asm2)*1e3:.2f} ms")

        # 🔹 Matrix sum
        t_sum0 = time.perf_counter()
        J = J1 + J2
        t_sum1 = time.perf_counter()
        print(f"[Newton {itr}] ⏳ Jacobian sum (J1+J2) = {(t_sum1 - t_sum0)*1e3:.2f} ms")

        if PHI_PERIODIC:
            t_pen0 = time.perf_counter()
            J = J + P  # Add periodic penalty
            t_pen1 = time.perf_counter()
            print(f"[Newton {itr}] ⏳ Penalty add         = {(t_pen1 - t_pen0)*1e3:.2f} ms")

        # 🔹 Combine RHS
        b = rhs1 + rhs2

        # 🔹 Condense BCs
        t_cond0 = time.perf_counter()
        Acond, bcond, p0, I = condense(J, b, D=D)
        t_cond1 = time.perf_counter()
        print(f"[Newton {itr}] ⏳ condense()           = {(t_cond1 - t_cond0)*1e3:.2f} ms")

        # 🔹 Build solver and preconditioner
        t_pc0 = time.perf_counter()
        solver = make_solver(Acond)
        t_pc1 = time.perf_counter()
        print(f"[Newton {itr}] 🔧 Preconditioner build = {(t_pc1 - t_pc0)*1e3:.2f} ms")

        # 🔹 Solve linear system
        t_solve0 = time.perf_counter()
        dp_I = solver(Acond, bcond)
        t_solve1 = time.perf_counter()
        print(f"[Newton {itr}] 🧠 GMRES solve          = {(t_solve1 - t_solve0)*1e3:.2f} ms")

        # 🔹 Update full pressure field
        dp = p0.copy()
        dp[I] = dp_I

        delta = np.linalg.norm(dp)
        p_prev = p.copy()

        if itr == maxiterNewton - 1 or delta < toleranceNewton:
            p += dp
        else:
            p += 0.9 * dp

        print(f"[Newton {itr}] ✅ Δp = {delta:.3e}")

        t_end = time.perf_counter()
        print(f"[Newton {itr}] 🕒 Total Newton step    = {(t_end - t0)*1e3:.2f} ms")
        print("-" * 60)

        if delta < toleranceNewton_static_init:
            break

 
# ----------------------------------------------------------------------------
# Load Carrying Capacity (Initial Equilibrium)
# ----------------------------------------------------------------------------

from skfem import FacetBasis, asm, LinearForm

# Split pressure into porous and Reynolds components
(pp0, ppbasis0), (pr0, prbasis0) = basis.split(p)

# Interpolate Reynolds pressure onto its surface facets
fbasis_dynamicLoad = FacetBasis(
    prbasis0.mesh,
    prbasis0.elem,
    facets=prbasis0.mesh.boundaries['reynolds_surface']
)
pressure_field = fbasis_dynamicLoad.interpolate(pr0)

# Define area form 
@LinearForm
def area_form(v, w):
    return w.x[0]*v

area = asm(area_form, fbasis_dynamicLoad).sum()

# Define vertical and horizontal load integrals
@LinearForm
def load_y(v, w):
    return jnp.cos(w.x[1]) * w.x[0] * w.p * v

@LinearForm
def load_x(v, w):
    return -jnp.sin(w.x[1]) * w.x[0] * w.p * v

# Evaluate initial load forces
Fy0 = asm(load_y, fbasis_dynamicLoad, p=pressure_field).sum()
Fx0 = asm(load_x, fbasis_dynamicLoad, p=pressure_field).sum()

# Infer equivalent static shaft mass (supports vertical load)
weight_force = -Fy0
m = Fy0 / 9.80665
print(f"Initial Load Fy0 = {Fy0:.3f} N, Implied mass = {m:.3f} kg")


# ----------------------------------------------------------------------------
# Initial Conditions for Rigid Body Motion
# ----------------------------------------------------------------------------

x = 0.0                      # Horizontal displacement [m]
y = epsilon * h0            # Vertical displacement [m] (from eccentricity)
vx = 0.0                    # Initial horizontal velocity [m/s]
vy = 0.0                    # Initial vertical velocity [m/s]

p_old = p                   # Store last solution (for time stepping)
x_old, y_old = x, y         # Save for geometric time lagging
Fx = 0                      # Initialize forces
Fy = 0

# First dynamic update using initial Fx0
x += dt * vx + 0.5 * dt**2 * Fx0 / m
vx += dt * Fx0 / m

y += dt * vy + 0.5 * dt**2 * Fy / m
vy += dt * Fy / m

# History storage for postprocessing
x_hist, y_hist = [], []
Fx_hist, Fy_hist = [], []
t_hist = []

# ----------------------------------------------------------------------------
# Dynamic Forms: Porous and Reynolds (with Time Dependency)
# ----------------------------------------------------------------------------

@NonlinearForm
def porous_dynamic(pp, pr, vp, vr, w):
    """
    Time-dependent nonlinear form for porous domain.

    Adds a backward-Euler time derivative term:
        ∂p/∂t ≈ (p - p_old) / dt

    Parameters
    ----------
    w['prevP'][0] : previous time-step pressure in porous domain.
    """
    r = w.x[0]
    ddt_term = r * ((pp - w['prevP'][0]) * vr) / dt
    darcy_term = -r * pp / mu * Kp * (
        pp.grad[0] * vp.grad[0] +
        1 / r**2 * pp.grad[1] * vp.grad[1] +
        pp.grad[2] * vp.grad[2]
    )
    return darcy_term + ddt_term


def make_reynolds_dynamic(x, y):
    """
    Factory function that defines Reynolds dynamic nonlinear form
    using current (x, y) displacement for geometry update.
    """
    @NonlinearForm
    def reynolds_dynamic(pp, pr, vp, vr, w):
        r = w.x[0]
        phi = w.x[1]

        # Old and current film thickness
        h_old = h0 + y_old * jnp.cos(phi) - x_old * jnp.sin(phi)
        h = h0 + y * jnp.cos(phi) - x * jnp.sin(phi)

        # Time derivative (backward Euler for ∂(ph)/∂t)
        ddt_term = r * ((h_old * w['prevR']) - h * pr) * vr / dt

        # Reynolds terms
        
# Slip parameters 
        # where 𝛼 is a dimensionless sli p coefficient
        # that depends on the material characteristics of the permeable material
        # and not on the physical properties of the fluid (Beavers et al. 1970)
        # PHI and PSI Set to zero here for negligible slip
        #alpha normal value around 0.1
        alpha=0.1
        sqrt_kp = jnp.sqrt(Kp)
        PHI = 3 * (sqrt_kp * h + 2 * alpha * Kp) / (h * (sqrt_kp + alpha * h))
        PSI = sqrt_kp / (sqrt_kp + alpha * h) 

        PHI, PSI = 0.0, 0.0  # Slip terms (neglected)
        Uphi = (omega + Omega) * R

        PhiTerm = (
            (Uphi * h / 2) * pr * vr.grad[1] -
            h**3 / (12 * mu * r) * pr * pr.grad[1] * vr.grad[1]
        )
        ZTerm = r * (
            (Vz * h / 2) * pr * vr.grad[2] -
            h**3 / (12 * mu) * pr * pr.grad[2] * vr.grad[2]
        )
        PressureGradientTerm = -r * Kp / (2 * mu / h) * pp * dot(grad(pp), w.n) * vr
        PenaltyTerm = r / eps_coupling * (pp - pr) * (vp - vr)

        return PhiTerm + ZTerm + PressureGradientTerm + PenaltyTerm + ddt_term

    return reynolds_dynamic

# Precompute reusable trace basis for interpolating old Reynolds pressure
(pp, ppbasis), (pr, prbasis) = basis.split(p_old)
fprbasis = FacetBasis(
    mesh,
    prbasis.elem,
    quadrature=fbasis.quadrature,
    facets=mesh.boundaries['bottom']
)


# ----------------------------------------------------------------------------
# Time-Stepping Loop: Solve Coupled System at Each Step
# ----------------------------------------------------------------------------

for step in range(Nt):
    t = step * dt
    reynolds_dynamic = make_reynolds_dynamic(x, y)

    if SOLVER == 'basic':
        for itr in range(50):
            J1, rhs1 = porous_dynamic.assemble(basis, x=p, prevP=p_old)
            (pp_old, _), (pr_old, _) = basis.split(p_old)
            J2, rhs2 = reynolds_dynamic.assemble(fbasis, x=p, prevR=fprbasis.interpolate(pr_old))

            A = J1 + J2
            b = rhs1 + rhs2

            if PHI_PERIODIC:
                A += P

            dp = solve(*condense(A, b, D=D))
            p_prev = p.copy()
            p += 0.7 * dp

            delta = np.linalg.norm(p - p_prev)
            print(f"[step {step}] Newton {itr}: Δp = {delta:.3e}")
            if delta < 1:
                break

    elif SOLVER == 'Krylov_gmres':
        def make_preconditioner(Acond):
            if PRECONDITIONER == 'ILU':
                return build_pc_ilu(Acond)
            elif PRECONDITIONER == 'JACOBI':
                return build_pc_diag(Acond)
            elif PRECONDITIONER == 'AMG':
                ml = smoothed_aggregation_solver(Acond)
                return ml.aspreconditioner()
            elif PRECONDITIONER == 'RUBEN':
                ml = ruge_stuben_solver(Acond)
                return ml.aspreconditioner()
            else:
                raise ValueError(f"Unknown preconditioner: {PRECONDITIONER}")

        def make_solver(Acond):
            M = make_preconditioner(Acond)
            return solver_iter_krylov(
                krylov=gmres,
                M=M,
                rtol=1e-7,
                atol=1e-8,
                maxiter=1000,
                restart=500,
                verbose=False
            )

        for itr in range(100):
            J1, rhs1 = porous.assemble(basis, x=p)
            (pp_old, _), (pr_old, _) = basis.split(p_old)
            J2, rhs2 = reynolds_dynamic.assemble(fbasis, x=p, prevR=fprbasis.interpolate(pr_old))

            J = J1 + J2
            b = rhs1 + rhs2

            if PHI_PERIODIC:
                J += P

            Acond, bcond, p0, I = condense(J, b, D=D)
            solver = make_solver(Acond)

            dp_I = solver(Acond, bcond)
            dp = p0.copy()
            dp[I] = dp_I

            # Dual-stopping criteria
            """ tol_dp_inf = 1e-3
            tol_res = 1e-6
            eps_p = 1e-12 """

            # Armijo backtracking line search
            res0 = np.linalg.norm(bcond)
            Adp_I = Acond.dot(dp_I)
            alpha = 1.0
            alpha_min = 1e-3
            c = 1e-4

            while alpha > alpha_min:
                res_trial = np.linalg.norm(bcond - alpha * Adp_I)
                if res_trial <= (1 - c * alpha) * res0:
                    break
                alpha *= 0.5

            # Update solution
            p_prev = p.copy()
            p = p_prev + alpha * dp

            # Evaluate update magnitude
            dp_applied = p - p_prev
            dp_rel_inf = np.max(
                np.divide(
                    np.abs(dp_applied),
                    np.abs(p),
                    out=np.zeros_like(dp_applied),
                    where=np.abs(p) > eps_p
                )
            )
            res_cond = np.linalg.norm(bcond - alpha * Adp_I)

            print(f"[step {step}] Newton {itr}: dp_rel_inf={dp_rel_inf:.2e}, res={res_cond:.2e}")

            if (dp_rel_inf < tol_dp_inf) and (res_cond < tol_res):
                print(f"[step {step}] Newton converged in {itr+1} iterations.")
                break
        else:
            print(f"[step {step}] Newton did NOT converge within 100 iterations.")

    # Evaluate pressure and interpolate force field
    (pp, ppbasis), (pr, prbasis) = basis.split(p)
    fbasis_dynamicLoad = FacetBasis(
        prbasis.mesh,
        prbasis.elem,
        facets=prbasis.mesh.boundaries['reynolds_surface']
    )
    pressure_field = fbasis_dynamicLoad.interpolate(pr)
    
    
 

    # ----------------------------------------------------------------------------
    # External Forcing and Rigid Body Update
    # ----------------------------------------------------------------------------

    # External vertical force (can be 0 until excitation starts)
    step_excitation = 5  # Step to begin applying sinusoidal excitation
    if step < step_excitation:
        F_ext_y = 0.0
    else:
        F_ext_y = A * np.sin(2 * np.pi * f * t)  # Sinusoidal vertical excitation

    # Total hydrodynamic force = pressure + body weight + external excitation
    Fy = asm(load_y, fbasis_dynamicLoad, p=pressure_field).sum() + weight_force + F_ext_y
    Fx = asm(load_x, fbasis_dynamicLoad, p=pressure_field).sum()

    print('Forces in X and Y =', Fx, Fy)

    # Save current pressure field as previous
    p_old = p
    x_old, y_old = x, y

    # Rigid body position update (2nd-order time integration)
    x += dt * vx + 0.5 * dt**2 * Fx / m
    vx += dt * Fx / m

    y += dt * vy + 0.5 * dt**2 * Fy / m
    vy += dt * Fy / m

    # Store data for plotting
    x_hist.append(x)
    y_hist.append(y)
    Fx_hist.append(Fx)
    Fy_hist.append(Fy)
    t_hist.append(dt * step)

    # Periodic reporting every 10 steps
    if step % 10 == 0:
        print(f"Step {step:3d}: x = {x:.2e}, Fx = {Fx:.2e}, y = {y:.2e}, Fy = {Fy:.2e}")


# ----------------------------------------------------------------------------
# Visualization of Trajectories and Forces
# ----------------------------------------------------------------------------

plt.figure(figsize=(6, 4))
plt.plot(t_hist, x_hist, label='x(t)')
plt.xlabel("Time (s)")
plt.ylabel("Displacement X (m)")
plt.title("Rigid Body Motion under Pressure Forces")
plt.grid(); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 4))
plt.plot(t_hist, y_hist, label='y(t)')
plt.xlabel("Time (s)")
plt.ylabel("Displacement Y (m)")
plt.title("Rigid Body Motion under Pressure Forces")
plt.grid(); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 4))
plt.plot(t_hist, Fx_hist, label='Fx(t)')
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Total Horizontal Force Over Time")
plt.grid(); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 4))
plt.plot(t_hist, Fy_hist, label='Fy(t)')
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Total Vertical Force Over Time")
plt.grid(); plt.legend(); plt.tight_layout(); plt.show()