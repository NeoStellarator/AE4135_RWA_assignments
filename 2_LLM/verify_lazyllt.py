"""
lazyllt verification script
----------------------------
Parameters:
  span        = 1 m       (rectangular wing, root_chord set for AR=5 as example)
  AoA         = 0 deg
  V_inf       = 60 m/s
  rho         = 1 kg/m³
  horseshoes  = 15 Fourier coefficients (num_coefficients in lazyllt)
  polar       = linear thin-airfoil polar, alpha_0 extracted by linear fit

The "polar" is a user-supplied 2D section Cl vs alpha table.
Here we use a NACA 2412-like polar (alpha_0 ≈ -2.08 deg) so the wing
produces non-trivial lift at AoA = 0, making it a useful verification case.
Swap in your own alpha / Cl_2D arrays to match your actual airfoil.
"""

import numpy as np
import matplotlib.pyplot as plt
from lazyllt import LiftingLineModel, UnsweptWing
from globals import main_dir,data_dir
# ─── 1. USER PARAMETERS ──────────────────────────────────────────────────────

SPAN          = 1.0    # m
ROOT_CHORD    = 1    # m  → AR = 5 for a rectangular wing
AOA           = 0.0    # deg (geometric angle of attack)
V_INF         = 60.0   # m/s
RHO           = 1.0    # kg/m³
N_HORSESHOES  = 15     # Fourier coefficients (odd terms: 1,3,5,…,29)
N_POINTS      = 15    # spanwise discretisation points on the wing

# ─── 2. SECTION POLAR ────────────────────────────────────────────────────────
# Replace these arrays with your actual polar data.
# Linear range only; lazyllt uses classical (linear) LLT.
# NACA 2412: alpha_0 ≈ -2.08 deg, Cl_alpha ≈ 2π/rad
polar_path= data_dir.joinpath("ARAD8pct_polar.txt")
alpha_polar_deg = np.array([-6, -4, -2, 0, 2, 4, 6, 8, 10], dtype=float)
Cl_2D           = np.array([-0.49, -0.28, -0.07, 0.14, 0.35, 0.56, 0.76, 0.95, 1.14], dtype=float)
polar_txt = np.loadtxt(polar_path,skiprows=2)
        

alpha_polar_deg = polar_txt[:,0]
Cl_2D    = polar_txt[:,1]

# Fit a line through the linear range to extract alpha_0 (zero-lift angle)
polar_coeffs = np.polyfit(alpha_polar_deg, Cl_2D, 1)   # [slope, intercept]
alpha_0_fit  = -polar_coeffs[1] / polar_coeffs[0]       # Cl = 0  →  alpha_0

print("═" * 52)
print("  SECTION POLAR FIT")
print(f"  Cl_alpha  = {polar_coeffs[0]:.4f} /deg  "
      f"({np.deg2rad(polar_coeffs[0]):.4f} /rad  ≈ 2π = {2*np.pi:.4f})")
print(f"  alpha_0   = {alpha_0_fit:.4f} deg")
print("═" * 52)

# ─── 3. BUILD WING & SOLVE ───────────────────────────────────────────────────

wing = UnsweptWing(
    span       = SPAN,
    root_chord = ROOT_CHORD,
    alpha_0    = alpha_0_fit,   # zero-lift angle from polar fit [deg]
    aoa        = AOA,
    num_points = N_POINTS,
)

model = LiftingLineModel(num_coefficients=N_HORSESHOES)
model.add_wing(wing)
solution = next(model.solve())

# ─── 4. GLOBAL AERODYNAMIC COEFFICIENTS ──────────────────────────────────────

CL         = float(solution.cl)
CDi        = float(solution.cdi)
e          = float(solution.efficiency)
L          = float(solution.lift(RHO, V_INF))
Di         = float(solution.induced_drag(RHO, V_INF))
solver_err = float(solution.solver_error)
AR         = float(wing.aspect_ratio)
S          = float(wing.wing_area)

print(f"\n  WING GEOMETRY")
print(f"  Span       = {SPAN} m")
print(f"  Root chord = {ROOT_CHORD} m")
print(f"  Wing area  = {S:.4f} m²")
print(f"  AR         = {AR:.4f}")

print(f"\n  LIFTING-LINE SOLUTION  (AoA = {AOA}°, V∞ = {V_INF} m/s, ρ = {RHO} kg/m³)")
print(f"  CL         = {CL:.6f}")
print(f"  CDi        = {CDi:.6f}")
print(f"  e (Oswald) = {e:.6f}")
print(f"  Lift       = {L:.4f} N")
print(f"  Induced D  = {Di:.4f} N")
print(f"  Solver RMS error = {solver_err:.4e} deg")
print("═" * 52)

# ─── 5. SPANWISE CL DISTRIBUTION ─────────────────────────────────────────────
# Kutta-Joukowski: L'(y) = ρ V∞ Γ(y)
# Section Cl(y)  = L'(y) / (½ ρ V∞² c(y))  =  2 Γ(y) / (V∞ c(y))

gamma   = np.array(solution.circulation(V_INF))   # Γ(y),  shape (N,)
chord   = np.array(wing.c)                          # c(y),  shape (N,)
y_nodes = np.array(wing.nodes)                      # y positions, shape (N,)

Cl_dist = 2.0 * gamma / (V_INF * chord)             # local section Cl

# Normalised span coordinate  η = 2y / b
eta = 2.0 * y_nodes / SPAN

# Elliptical reference distribution scaled to same total CL
# CL_elliptic(η) = (4 CL / π) * √(1 - η²)
Cl_elliptic = (4.0 * CL / np.pi) * np.sqrt(np.clip(1.0 - eta**2, 0, None))

# ─── 6. INDUCED ANGLE OF ATTACK ──────────────────────────────────────────────
alpha_i   = np.array(solution.alpha_induced)    # deg, shape (N,)
alpha_eff = np.array(solution.alpha_effective)  # deg, shape (N,)

# ─── 7. PLOT ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"lazyllt — Rectangular Wing  (b={SPAN} m, c={ROOT_CHORD} m, "
    f"AoA={AOA}°, V∞={V_INF} m/s, ρ={RHO} kg/m³, N={N_HORSESHOES})",
    fontsize=12
)

# — Panel A: Section polar fit —
ax = axes[0]
alpha_fit_line = np.linspace(alpha_polar_deg[0], alpha_polar_deg[-1], 100)
Cl_fit_line    = np.polyval(polar_coeffs, alpha_fit_line)
ax.scatter(alpha_polar_deg, Cl_2D, color="royalblue", zorder=5, label="Polar data")
ax.plot(alpha_fit_line, Cl_fit_line, "r--", label=f"Linear fit  α₀={alpha_0_fit:.2f}°")
ax.axhline(0, color="k", linewidth=0.7)
ax.axvline(alpha_0_fit, color="grey", linewidth=0.7, linestyle=":")
ax.set_xlabel("Section α [deg]")
ax.set_ylabel("Section Cl [ ]")
ax.set_title("Section Polar")
ax.legend()
ax.grid(True, alpha=0.3)

# — Panel B: Spanwise Cl distribution —
ax = axes[1]
ax.plot(eta, Cl_dist,    "b-",  linewidth=2,   label="lazyllt  Cl(y)")
ax.plot(eta, Cl_elliptic,"r--", linewidth=1.5, label="Elliptic ref")
ax.set_xlabel("η = 2y/b [ ]")
ax.set_ylabel("Local Cl [ ]")
ax.set_title(f"Spanwise Cl Distribution  (CL={CL:.4f})")
ax.legend()
ax.grid(True, alpha=0.3)

# — Panel C: Induced and effective AoA —
ax = axes[2]
ax.plot(eta, alpha_i,   "g-",  linewidth=2,  label="α_induced [deg]")
ax.plot(eta, alpha_eff, "m--", linewidth=1.5, label="α_effective [deg]")
ax.set_xlabel("η = 2y/b [ ]")
ax.set_ylabel("Angle [deg]")
ax.set_title("Induced & Effective AoA")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lazyllt_cl_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Plot saved → lazyllt_cl_distribution.png")