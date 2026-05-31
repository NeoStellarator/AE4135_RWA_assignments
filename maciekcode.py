"""
LL_convection_wake_2.py  -  Lifting Line Theory for a horizontal-axis wind turbine rotor.

Each blade element is represented by a horseshoe vortex: one bound filament at the
quarter-chord and two trailing filaments that extend into a helical frozen wake.
The wake convection speed is U_wake = Uinf*(1 - a_avg), where a_avg is the
rotor-averaged axial induction updated each outer iteration.
Bound circulation at each element is found by fixed-point iteration with relaxation.
Spanwise panels use cosine distribution for higher resolution near root and tip.

Rotor: DU95-W-180, R=50 m, B=3, TSR=10, Uinf=10 m/s
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# PARAMETERS
# =============================================================================
R       = 50.0
B       = 3
R_ROOT  = 0.2
R_TIP   = 1.0
UINF    = 10.0
RHO     = 1.225
TSR     = 10.0
OMEGA   = UINF * TSR / R

N_ELEMENTS = 30
N_REVS     = 10
D_PSI_DEG  = 5.0
COSINE_DISTRIBUTION = True   # True: cosine spanwise spacing (tutorial recommendation)

RELAX          = 0.01
RELAX_U        = 0.3
MAX_ITER_INNER = 300
MAX_ITER_OUTER = 100
TOL_INNER      = 5e-5   # relative tolerance: max|dGamma| / max|Gamma|
TOL_OUTER      = 5e-5   # absolute tolerance on u_conv change [m/s]

VORTEX_CORE_RC = 0.0001

POLAR_FILE   = os.path.join(os.path.dirname(__file__), "DU95W180_polar.csv")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "LL_convection2_results.txt")
FIGURE_FILE  = os.path.join(os.path.dirname(__file__), "figures", "LL_convection2.png")


# =============================================================================
# POLAR
# =============================================================================
def load_polar(filepath):
    df = pd.read_csv(filepath, skiprows=1)
    return df["Alfa"].values, df["Cl"].values, df["Cd"].values


def interp_polar(alpha_deg, alpha_data, cl_data, cd_data):
    cl = np.interp(alpha_deg, alpha_data, cl_data)
    cd = np.interp(alpha_deg, alpha_data, cd_data)
    return cl, cd


# =============================================================================
# GEOMETRY
# =============================================================================
def blade_geometry(N):
    if COSINE_DISTRIBUTION:
        # Cosine spacing: denser near root and tip where circulation gradients are larger
        theta = np.linspace(0.0, np.pi, N + 1)
        edges = R_ROOT + 0.5 * (R_TIP - R_ROOT) * (1.0 - np.cos(theta))
    else:
        edges = np.linspace(R_ROOT, R_TIP, N + 1)
    ctrl  = 0.5 * (edges[:-1] + edges[1:])
    chord = 3.0 * (1.0 - ctrl) + 1.0
    twist = -14.0 * (1.0 - ctrl) + 2.0
    return edges, ctrl, chord, twist


# =============================================================================
# WAKE
# =============================================================================
def build_wake(r_panel_edges, u_conv):
    """
    Helical frozen wake. Returns WP: (B, N+1, n_steps+1, 3).
    Axial direction x, rotor plane y-z.
    """
    r_edges = r_panel_edges * R
    n_steps = int(round(N_REVS * 2.0 * np.pi / np.radians(D_PSI_DEG)))
    psi     = np.arange(n_steps + 1) * np.radians(D_PSI_DEG)
    x_wake  = (u_conv / OMEGA) * psi       # x_w = t * U_wake

    WP = np.zeros((B, len(r_edges), len(psi), 3))
    for b in range(B):
        psi_b = b * 2.0 * np.pi / B
        for j, r in enumerate(r_edges):
            WP[b, j, :, 0] = x_wake
            WP[b, j, :, 1] = r * np.cos(psi_b + psi)
            WP[b, j, :, 2] = r * np.sin(psi_b + psi)
    return WP


# =============================================================================
# BIOT-SAVART  (Katz & Plotkin algorithm)
# =============================================================================
def biot_savart_segments(P, P1, P2):
    """
    Induced velocity at P from K unit-strength vortex filaments P1->P2.
    P:  (3,)    target point
    P1: (K, 3)  segment start points
    P2: (K, 3)  segment end points
    Returns (3,) velocity — sum over all K filaments.
    """
    r1  = P - P1
    r2  = P - P2
    n1  = np.linalg.norm(r1, axis=1, keepdims=True)
    n2  = np.linalg.norm(r2, axis=1, keepdims=True)
    cx  = np.cross(r1, r2)
    csq = np.einsum("ij,ij->i", cx, cx)
    r0  = P2 - P1
    dot = np.einsum("ij,ij->i", r0,
                    r1 / np.maximum(n1, 1e-12)
                    - r2 / np.maximum(n2, 1e-12))
    denom = np.maximum(csq, VORTEX_CORE_RC**2)
    ok    = (n1[:, 0] > 1e-6) & (n2[:, 0] > 1e-6)
    V     = np.where(ok[:, None], cx * (dot / denom)[:, None], 0.0)
    return V.sum(axis=0) / (4.0 * np.pi)


# =============================================================================
# INFLUENCE MATRICES
# =============================================================================
def build_trailing_matrix(r_ctrl, r_edges, u_conv):
    """
    A[i, j, 3] = velocity at ctrl[i] per unit strength of trailing vortex line j.
    j runs over N+1 panel edges. Summed over all B blades.
    Shape: (N, N+1, 3).
    """
    WP = build_wake(r_edges, u_conv)
    N  = len(r_ctrl)
    ctrl_pts = np.column_stack([np.zeros(N), r_ctrl * R, np.zeros(N)])
    A = np.zeros((N, N + 1, 3))
    print("  Building trailing vortex matrix...")
    for i in range(N):
        for b in range(B):
            for j in range(N + 1):
                A[i, j] += biot_savart_segments(ctrl_pts[i],
                                                WP[b, j, :-1],
                                                WP[b, j, 1:])
    return A


def build_bound_vortex_matrix(r_ctrl, r_edges):
    """
    Bbv[i, j, 3] = velocity at ctrl[i] per unit Gamma of the bound vortex
                   filament of horseshoe j (summed over all B blades).

    Each bound vortex runs from r_edges[j] to r_edges[j+1] at the blade
    azimuthal position in the rotor plane (x = 0), representing the quarter-
    chord filament described in the tutorial.

    Shape: (N, N, 3).
    Note: for blade 0, control points and bound vortices are collinear (both
    on the y-axis), so blade 0 contributes zero — as expected from Biot-Savart.
    The non-zero contribution comes from the other B-1 blades.
    """
    N = len(r_ctrl)
    ctrl_pts = np.column_stack([np.zeros(N), r_ctrl * R, np.zeros(N)])
    Bbv = np.zeros((N, N, 3))
    print("  Building bound vortex matrix...")
    for i in range(N):
        for b in range(B):
            psi_b = b * 2.0 * np.pi / B
            cos_b = np.cos(psi_b)
            sin_b = np.sin(psi_b)
            for j in range(N):
                P1 = np.array([[0.0,
                                r_edges[j]   * R * cos_b,
                                r_edges[j]   * R * sin_b]])
                P2 = np.array([[0.0,
                                r_edges[j+1] * R * cos_b,
                                r_edges[j+1] * R * sin_b]])
                Bbv[i, j] += biot_savart_segments(ctrl_pts[i], P1, P2)
    return Bbv


# =============================================================================
# HELPERS
# =============================================================================
def gamma_to_dG(Gamma):
    """Trailing vortex strength from bound circulation (Helmholtz theorem)."""
    N        = len(Gamma)
    dG       = np.zeros(N + 1)
    dG[0]    = Gamma[0]
    dG[1:-1] = Gamma[1:] - Gamma[:-1]
    dG[-1]   = -Gamma[-1]
    return dG


def total_induced_velocity(A, dG, Bbv, Gamma):
    """Sum trailing and bound vortex contributions."""
    return (np.einsum("ijk,j->ik", A, dG)
            + np.einsum("ijk,j->ik", Bbv, Gamma))


# =============================================================================
# INNER SOLVER  (fixed-point iteration, wake held fixed)
# =============================================================================
def _inner_solve(r_ctrl, chord, twist, A, Bbv, alpha_data, cl_data, cd_data,
                 Gamma_init=None):
    """Iterate Gamma to convergence using relative tolerance."""
    r_ctrl_m = r_ctrl * R
    Gamma    = Gamma_init.copy() if Gamma_init is not None else np.zeros(N_ELEMENTS)

    for _ in range(MAX_ITER_INNER):
        dG    = gamma_to_dG(Gamma)
        u_ind = total_induced_velocity(A, dG, Bbv, Gamma)

        Vn = UINF + u_ind[:, 0]
        Vt = OMEGA * r_ctrl_m + u_ind[:, 2]

        phi       = np.arctan2(Vn, Vt)
        alpha_deg = twist + np.degrees(phi)
        cl, _     = interp_polar(alpha_deg, alpha_data, cl_data, cd_data)

        Gamma_new = 0.5 * np.sqrt(Vn**2 + Vt**2) * chord * cl

        ref_err = max(np.max(np.abs(Gamma_new)), 0.001)
        err     = np.max(np.abs(Gamma_new - Gamma)) / ref_err
        Gamma   = RELAX * Gamma_new + (1.0 - RELAX) * Gamma

        if err < TOL_INNER:
            break

    return Gamma


# =============================================================================
# MAIN SOLVER
# =============================================================================
def solve_lifting_line(alpha_data, cl_data, cd_data):
    """
    Outer loop: rebuild trailing matrix with U_wake = Uinf*(1 - a_avg).
    Inner loop: iterate Gamma for fixed wake geometry.
    Bound vortex matrix is built once (independent of wake speed).
    """
    r_edges, r_ctrl, chord, twist = blade_geometry(N_ELEMENTS)

    print("Building bound vortex matrix (once)...")
    Bbv = build_bound_vortex_matrix(r_ctrl, r_edges)

    u_conv        = UINF
    Gamma         = np.zeros(N_ELEMENTS)
    a_avg         = 0.0
    n_iters_outer = 0

    for outer in range(MAX_ITER_OUTER):
        print(f"Outer iteration {outer + 1}:  u_conv = {u_conv:.4f} m/s")
        A     = build_trailing_matrix(r_ctrl, r_edges, u_conv)
        Gamma = _inner_solve(r_ctrl, chord, twist, A, Bbv,
                             alpha_data, cl_data, cd_data, Gamma_init=Gamma)

        dG    = gamma_to_dG(Gamma)
        u_ind = total_induced_velocity(A, dG, Bbv, Gamma)

        a_dist     = -u_ind[:, 0] / UINF
        a_avg      = np.trapezoid(a_dist * r_ctrl, r_ctrl) / np.trapezoid(r_ctrl, r_ctrl)
        u_conv_new = RELAX_U * UINF * (1.0 - a_avg) + (1.0 - RELAX_U) * u_conv
        n_iters_outer += 1

        if abs(u_conv_new - u_conv) < TOL_OUTER:
            u_conv = u_conv_new
            print(f"Outer loop converged: u_conv={u_conv:.4f} m/s  a_avg={a_avg:.5f}")
            break
        u_conv = u_conv_new

    return _postprocess(r_edges, r_ctrl, chord, twist, Gamma, A, Bbv,
                        alpha_data, cl_data, cd_data, n_iters_outer, u_conv, a_avg)


def _postprocess(r_edges, r_ctrl, chord, twist, Gamma, A, Bbv,
                 alpha_data, cl_data, cd_data, n_iters, u_conv, a_avg):
    r_ctrl_m = r_ctrl * R

    dG    = gamma_to_dG(Gamma)
    u_ind = total_induced_velocity(A, dG, Bbv, Gamma)

    Vn    = UINF + u_ind[:, 0]
    Vt    = OMEGA * r_ctrl_m + u_ind[:, 2]
    Vmag2 = Vn**2 + Vt**2

    phi       = np.arctan2(Vn, Vt)
    alpha_deg = twist + np.degrees(phi)
    cl, cd    = interp_polar(alpha_deg, alpha_data, cl_data, cd_data)

    a  = -u_ind[:, 0] / UINF
    ap =  u_ind[:, 2] / (OMEGA * r_ctrl_m)

    fn = 0.5 * RHO * Vmag2 * chord * (cl * np.cos(phi) + cd * np.sin(phi))
    ft = 0.5 * RHO * Vmag2 * chord * (cl * np.sin(phi) - cd * np.cos(phi))

    dr = (r_edges[1:] - r_edges[:-1]) * R
    CT = np.sum(fn * dr * B) / (0.5 * RHO * UINF**2 * np.pi * R**2)
    CP = np.sum(ft * r_ctrl_m * OMEGA * dr * B) / (0.5 * RHO * UINF**3 * np.pi * R**2)

    return dict(
        r_R=r_ctrl, a=a, ap=ap,
        phi=np.degrees(phi), alpha=alpha_deg,
        cl=cl, cd=cd, gamma=Gamma,
        fn=fn, ft=ft, dr=dr,
        CT=CT, CP=CP,
        u_conv=u_conv, a_avg=a_avg, n_iters=n_iters,
    )


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================
def run_convection_wake_2(polar_file=POLAR_FILE):
    """Solve and return results dict. Called by comparison_LL.py."""
    alpha_data, cl_data, cd_data = load_polar(polar_file)
    return solve_lifting_line(alpha_data, cl_data, cd_data)


# =============================================================================
# OUTPUT
# =============================================================================
def save_results(res, filepath=RESULTS_FILE):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lines = [
        f"LL convection wake  |  R={R}m  B={B}  TSR={TSR}  Uinf={UINF}m/s",
        f"wake conv. speed = {res['u_conv']:.4f} m/s   a_avg = {res['a_avg']:.5f}",
        f"CT = {res['CT']:.4f}   CP = {res['CP']:.4f}   outer_iters={res['n_iters']}",
        "",
        f"{'r/R':>8} {'a':>8} {'a_prime':>9} {'phi[deg]':>10} {'alpha[deg]':>11}"
        f" {'Cl':>7} {'Cd':>8} {'gamma':>9} {'fn[N/m]':>10} {'ft[N/m]':>10}",
    ]
    for i in range(len(res["r_R"])):
        lines.append(
            f"{res['r_R'][i]:>8.4f} {res['a'][i]:>8.4f} {res['ap'][i]:>9.4f}"
            f" {res['phi'][i]:>10.3f} {res['alpha'][i]:>11.3f}"
            f" {res['cl'][i]:>7.4f} {res['cd'][i]:>8.5f}"
            f" {res['gamma'][i]:>9.4f} {res['fn'][i]:>10.2f} {res['ft'][i]:>10.2f}"
        )
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {filepath}")


def print_results(res):
    print(f"\nLL convection wake  |  TSR={TSR}  CT={res['CT']:.4f}  CP={res['CP']:.4f}")
    print(f"u_conv={res['u_conv']:.4f} m/s   a_avg={res['a_avg']:.5f}")
    print(f"{'r/R':>8} {'a':>8} {'a`':>8} {'alpha[deg]':>11} {'Cl':>7}")
    for i in range(len(res["r_R"])):
        print(f"{res['r_R'][i]:>8.4f} {res['a'][i]:>8.4f} {res['ap'][i]:>8.4f}"
              f" {res['alpha'][i]:>11.3f} {res['cl'][i]:>7.4f}")


# =============================================================================
# PLOT
# =============================================================================
def plot_results(res, filepath=FIGURE_FILE):
    norm = 0.5 * RHO * UINF**2 * R
    r    = res["r_R"]

    fig = plt.figure(figsize=(14, 16))
    fig.suptitle(
        "LL convection wake — spanwise distributions\n"
        f"DU95-W-180,  R={R} m,  B={B},  TSR={TSR},  V₀={UINF} m/s  |  "
        f"CT={res['CT']:.4f}   CP={res['CP']:.4f}   "
        f"u_conv={res['u_conv']:.3f} m/s   a_avg={res['a_avg']:.4f}",
        fontsize=10, fontweight="bold",
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.32)

    COLOR = "#7570b3"
    KW    = dict(color=COLOR, lw=2.0, marker="D", ms=3)

    def panel(ax, y, title, ylabel, hline=None):
        ax.plot(r, y, **KW)
        if hline is not None:
            ax.axhline(hline, color="0.45", lw=1.0, ls=":", label=f"{hline:.3f}")
            ax.legend(fontsize=7)
        ax.set_xlabel("r/R [-]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.35)

    panel(fig.add_subplot(gs[0, 0]), res["phi"],               "Inflow angle",           "phi [deg]")
    panel(fig.add_subplot(gs[0, 1]), res["alpha"],             "Angle of attack",        "alpha [deg]")
    panel(fig.add_subplot(gs[1, 0]), res["a"],                 "Axial induction factor", "a [-]",      hline=1.0/3.0)
    panel(fig.add_subplot(gs[1, 1]), res["ap"],                "Tangential induction",   "a' [-]")
    panel(fig.add_subplot(gs[2, 0]), res["cl"],                "Lift coefficient",       "Cl [-]")
    panel(fig.add_subplot(gs[2, 1]), res["cd"],                "Drag coefficient",       "Cd [-]")
    panel(fig.add_subplot(gs[3, 0]), res["fn"] / norm,         "Normal loading",         "Fn / (0.5ρV²R) [-]")
    panel(fig.add_subplot(gs[3, 1]), res["gamma"],             "Bound circulation",      r"$\Gamma$ [m²/s]")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {filepath}")
    return fig


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("Running LL convection wake solver...")
    res = run_convection_wake_2()
    print_results(res)
    save_results(res)
    plot_results(res)
    plt.show()
