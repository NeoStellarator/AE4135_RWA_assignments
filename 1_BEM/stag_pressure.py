import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from globals import data_dir, plot_dir, res_dir
from Rotor import Rotor

def stag_pressure_spanwise_loc(locations, save=False):
    V0 = 60  # m/s 
    rho = 1.007  # kg/m³
    p0 = 7.9495e4  # N/m² (freestream static pressure)
    

    c_R_func = lambda r_R : 0.18-0.06*r_R
    twst_func = lambda r_R : -50*r_R+35

    rotor = Rotor(
        # Geometry definition
        c_R_func  = c_R_func,
        twst_func = twst_func,
        pitch = 45,
        B     = 6,
        R     = 0.7,
        r_R_H = 0.25,
        polar_path = data_dir.joinpath('ARAD8pct_polar.txt'),
        # Operating condition
        J = 2,
        Vinf = 60,
        rho = 1.067,
        # Discretizaiton
        n_elem = 100,
        dist_elem = 'uniform',
        isPropeller = True)

    fpath = res_dir.joinpath("propeller_radial_data_J_2.csv")
    rotor.export_dist(fpath)
    df = pd.read_csv(fpath)
    
    # Get rotor parameters
    R = 0.7  # Rotor radius in meters
    J = 2  # Advance ratio
    n = J * V0 / (2 * np.pi * R)  # Rotational speed in revolutions per second
    omega = 2 * np.pi * n  # rad/s
    
    # Calculate actual radius
    r_actual = df['r_R'].values * R
    
    # Extract induction factors
    a_values = df['a'].values
    a_prime_values = df['aline'].values
    
    # Calculate dr for area calculations
    dr = np.gradient(r_actual)
    dA = 2 * np.pi * r_actual * dr
    
    def stagnation_pressure(p_static, V_axial, V_tangential, rho):
        """Calculate stagnation pressure from static pressure and velocity components"""
        V_magnitude = np.sqrt(V_axial**2 + V_tangential**2)
        return p_static + 0.5 * rho * V_magnitude**2
    
    # ============= FAR UPWIND =============
    V_axial_far_upwind = V0 * np.ones(len(df))
    V_tang_far_upwind = np.zeros(len(df))
    p_static_far_upwind = p0 * np.ones(len(df))
    p_stag_far_upwind = stagnation_pressure(p_static_far_upwind, V_axial_far_upwind, V_tang_far_upwind, rho)
    
    # Reference stagnation pressure for normalization
    p_stag_ref = p_stag_far_upwind[0]  # Freestream stagnation pressure (constant)
    
    # ============= UPWIND OF ROTOR =============
    V_axial_upwind = V0 * (1 + a_values)
    V_tang_upwind = omega * r_actual * (1 - a_prime_values)
    # Static pressure from Bernoulli (total pressure conserved from far upwind)
    p_static_upwind = p0 + 0.5 * rho * V0**2 - 0.5 * rho * (V_axial_upwind**2 + V_tang_upwind**2)
    p_stag_upwind = stagnation_pressure(p_static_upwind, V_axial_upwind, V_tang_upwind, rho)
    
    # ============= PRESSURE JUMP ACROSS ROTOR USING Ct =============
    # Always use Ct (thrust coefficient) for pressure jump
    if 'Ct' in df.columns and df['Ct'].notna().any():
        # dT = Ct * 0.5 * rho * V0^2 * dA
        dT = df['Ct'].values * 0.5 * rho * V0**2 * dA
        # Static pressure jump = dT / dA (force per area)
        delta_p_static = dT / dA
        method_used = "Ct (thrust coefficient)"
        print(f"Using {method_used} for pressure jump calculation")
    else:
        print("Warning: Ct not found in data")
        method_used = "Not available"
    
    # ============= DOWNWIND OF ROTOR =============
    # Velocities are continuous through rotor
    V_axial_downwind = V0 * (1 + a_values)
    V_tang_downwind = omega * r_actual * (1 - a_prime_values)
    # Static pressure jumps by delta_p_static
    p_static_downwind = p_static_upwind + delta_p_static
    # Stagnation pressure increases by the work added by rotor
    p_stag_downwind = stagnation_pressure(p_static_downwind, V_axial_downwind, V_tang_downwind, rho)
    
    # ============= FAR DOWNWIND =============
    # Wake develops: axial and tangential velocities increase
    V_axial_far_downwind = V0 * (1 + 2*a_values)
    V_tang_far_downwind = 2 * omega * r_actual * (1 - a_prime_values)
    # Static pressure from Bernoulli (stagnation pressure conserved from downwind)
    V_mag_far_downwind = np.sqrt(V_axial_far_downwind**2 + V_tang_far_downwind**2)
    p_static_far_downwind = p_stag_downwind - 0.5 * rho * V_mag_far_downwind**2
    # Stagnation pressure is conserved (no losses)
    p_stag_far_downwind = stagnation_pressure(p_static_far_downwind, V_axial_far_downwind, V_tang_far_downwind, rho)
    
    # Normalize stagnation pressures
    p_stag_far_upwind_norm = p_stag_far_upwind / p_stag_ref
    p_stag_upwind_norm = p_stag_upwind / p_stag_ref
    p_stag_downwind_norm = p_stag_downwind / p_stag_ref
    p_stag_far_downwind_norm = p_stag_far_downwind / p_stag_ref
    
    # Create dictionary for easy access with normalized values
    results = {
        'far upwind': (p_stag_far_upwind_norm, V_axial_far_upwind, V_tang_far_upwind, p_static_far_upwind),
        'upwind of the rotor disk': (p_stag_upwind_norm, V_axial_upwind, V_tang_upwind, p_static_upwind),
        'downwind of the rotor disk': (p_stag_downwind_norm, V_axial_downwind, V_tang_downwind, p_static_downwind),
        'far downwind': (p_stag_far_downwind_norm, V_axial_far_downwind, V_tang_far_downwind, p_static_far_downwind)
    }
    
    
    # Additional plot showing all locations together for comparison
    fig2, ax2 = plt.subplots(figsize=(6,4))
    
    ax2.plot(df['r_R'], p_stag_upwind_norm, 'y-', linewidth=3, label='Upwind of rotor disk')
    ax2.plot(df['r_R'], p_stag_far_upwind_norm, 'k--', linewidth=2, label='Far upwind')
    ax2.plot(df['r_R'], p_stag_downwind_norm, 'b-', linewidth=3, label='Downwind of rotor disk')#,marker='^', markersize=3)
    ax2.plot(df['r_R'], p_stag_far_downwind_norm, 'r--', linewidth=2, label='Far downwind')#,marker='d', markersize=1)
    
    
    ax2.set_xlabel(r'$r/R$', fontsize=12)
    ax2.set_ylabel(r'$p_{\text{t}} / p_{\text{t},\infty}$ [-]', fontsize=12)
    # ax2.set_title('Stagnation Pressure Distribution Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    # ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    if save:
        plt.savefig(plot_dir.joinpath('stag_pressure_comparison.pdf'), bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Method used: {method_used}")
    print(f"Freestream stagnation pressure (reference): {p_stag_ref:.0f} Pa")
    print(f"Normalized freestream stagnation pressure: 1.000")
    
    mid_idx = len(df) // 2
    print(f"\nAt r/R={df['r_R'].values[mid_idx]:.2f}:")
    print(f"  Upwind stagnation (normalized): {p_stag_upwind_norm[mid_idx]:.4f}")
    print(f"  Downwind stagnation (normalized): {p_stag_downwind_norm[mid_idx]:.4f}")
    print(f"  Far downwind stagnation (normalized): {p_stag_far_downwind_norm[mid_idx]:.4f}")
    
    # Check conservation (should be equal)
    diff_norm = p_stag_downwind_norm[mid_idx] - p_stag_far_downwind_norm[mid_idx]
    print(f"\nStagnation pressure conservation (downwind vs far downwind):")
    print(f"  Difference (normalized): {diff_norm:.6f}")
    print(f"  Should be 0 (within numerical precision)")
    
    # Static pressure evolution
    print(f"\nStatic pressure evolution at r/R={df['r_R'].values[mid_idx]:.2f}:")
    print(f"  Far upwind: {p_static_far_upwind[mid_idx]:.0f} Pa")
    print(f"  Upwind: {p_static_upwind[mid_idx]:.0f} Pa")
    print(f"  Downwind: {p_static_downwind[mid_idx]:.0f} Pa")
    print(f"  Far downwind: {p_static_far_downwind[mid_idx]:.0f} Pa")
    
    # Additional info about coefficients used
    if 'Ct' in df.columns:
        print(f"\nCt at r/R={df['r_R'].values[mid_idx]:.2f}: {df['Ct'].values[mid_idx]:.4f}")
    
    # Show peak location
    peak_idx = np.argmax(p_stag_downwind_norm)
    print(f"\nPeak stagnation pressure occurs at r/R={df['r_R'].values[peak_idx]:.3f}")
    print(f"  Peak normalized value: {p_stag_downwind_norm[peak_idx]:.4f}")
    
    return df, results, method_used

# Run the function
locations = ['far upwind', 'upwind of the rotor disk', 'downwind of the rotor disk', 'far downwind']
df_result, results, method = stag_pressure_spanwise_loc(locations, save=False)