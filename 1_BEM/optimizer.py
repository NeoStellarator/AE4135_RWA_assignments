import numpy as np
from scipy.optimize import minimize
from pathlib import Path
from Rotor import Rotor
from globals import data_dir

def normalize(x, bounds):
    x_norm = [(x[i]-bounds[i][0])/(bounds[i][1]-bounds[i][0]) for i in range(len(x))]
    bounds_norm = [[0,1] for i in range(len(x))]
    return x_norm, bounds_norm

def denormalize(x_norm, bounds):
    x = [bounds[i][0] + x_norm[i]*(bounds[i][1]-bounds[i][0]) for i in range(len(x_norm))]
    return x

def objective_harvest(x_norm, args):
    """
    Returns power (W). Negative = harvesting.
    Huge penalty if any alpha outside [-10,25] or BEM fails.
    """
    B, J, r_R_H, polar_path, n_elem, dist_elem, isPropeller, R, Uinf, rho, bounds = args
    x = denormalize(x_norm, bounds)
    at, bt, ac, bc = x

    twst_func = lambda r_R: at*r_R + bt
    c_R_func = lambda r_R: bc + ac*r_R

    try:
        # Updated Rotor initialization for new class
        rotor = Rotor(
            c_R_func=c_R_func,
            twst_func=twst_func,
            pitch=0,
            B=B,
            R=R,  
            r_R_H=r_R_H,
            polar_path=polar_path,
            J=J,
            Vinf=Uinf, 
            rho=rho,
            n_elem=n_elem,
            dist_elem=dist_elem,
            isPropeller=isPropeller
        )

        power = rotor.P

        alphas = rotor.alpha

        # Alpha constraint
        if np.any(alphas < -10) or np.any(alphas > 25):
            return 1e12  

        return power

    except Exception:
        return 1e12

# ---------- Configuration ----------
bounds = [[-90, 90], [10, 120], [-0.10, -0.02], [0.10, 0.30]]
x0 = [-50, 81, -0.06, 0.18]
x0_norm, bounds_norm = normalize(x0, bounds)

B = 6
R = 0.7
r_R_H = 0.25
n_elem = 100
polar_path = data_dir.joinpath('ARAD8pct_polar.txt')
dist_elem = "uniform"
isPropeller = True
Uinf = 60
rho = 1.067
J = 60/20/1.4

args = [B, J, r_R_H, polar_path, n_elem, dist_elem, isPropeller, R, Uinf, rho, bounds]

# ---------- Step 1: Generate many random starting points ----------
print("Generating random initial points...")
np.random.seed(42)  # for reproducibility
n_random = 1000
random_starts = np.random.uniform(0, 1, (n_random, len(x0_norm)))

# Add structured points
structured = [
    [0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1],
    [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1],
    [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1],
    [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1],
    [0.5,0.5,0.5,0.5]
]
random_starts = np.vstack([random_starts, structured])
print(f"Total start points: {len(random_starts)}")

# ---------- Step 2: Evaluate all quickly (no optimization) ----------
print("Evaluating initial points...")
values = []
for i, start in enumerate(random_starts):
    val = objective_harvest(start, args)
    values.append(val)
    if (i+1) % 100 == 0:
        print(f"  Evaluated {i+1}/{len(random_starts)}")

values = np.array(values)

# Keep only feasible points (not huge penalty)
feasible_idx = values < 1e8   # 1e12 is penalty, we want < 1e8
feasible_values = values[feasible_idx]
feasible_starts = random_starts[feasible_idx]

if len(feasible_starts) == 0:
    print("No feasible points found! Check alpha bounds or BEM.")
    exit()

print(f"Found {len(feasible_starts)} feasible points.")
print(f"Best feasible power so far: {np.min(feasible_values):.2f} W")

# Take top N 
n_best = 10
best_indices = np.argsort(feasible_values)[:n_best]
best_starts = feasible_starts[best_indices]
best_start_values = feasible_values[best_indices]

print("\nTop 5 starting powers (W):", best_start_values[:5])

# ---------- Step 3: Local optimization from best starts ----------
print("\nRunning SLSQP from top starts...")
best_results = []   # store (res, is_feasible)
for i, start in enumerate(best_starts):
    print(f"Optimizing start {i+1}/{len(best_starts)}...")
    res = minimize(
        fun=objective_harvest,
        args=args,
        x0=start,
        method="SLSQP",
        bounds=bounds_norm,
        options={'maxiter': 200, 'ftol': 1e-6, 'eps': 0.05, 'disp': False}
    )
    # Check if result is feasible
    final_val = objective_harvest(res.x, args)
    if final_val < 1e8:
        best_results.append((res, final_val))
        print(f"  -> Feasible: power={final_val:.2f} W")
    else:
        print(f"  -> Infeasible (alpha violation)")

if not best_results:
    print("No feasible results after local optimization.")
    exit()

# Find best feasible result
best_res, best_feasible_val = min(best_results, key=lambda x: x[1])
print(f"\nBest feasible after local search: {best_feasible_val:.2f} W")

# ---------- Step 4: Final refinement ----------
print("\nFinal refinement with tighter tolerances...")
final_res = minimize(
    fun=objective_harvest,
    args=args,
    x0=best_res.x,
    method="SLSQP",
    bounds=bounds_norm,
    options={'maxiter': 500, 'ftol': 1e-10, 'eps': 0.02, 'disp': True}
)

# Verify final result
final_obj = objective_harvest(final_res.x, args)
if final_obj > 1e8:
    print("Final refinement produced infeasible design. Using best from local search.")
    final_res = best_res
    final_obj = best_feasible_val
else:
    print(f"Final refinement gave feasible power: {final_obj:.2f} W")

print(f"\nFinal power (objective value): {final_obj:.2f} W")

# Denormalize final parameters
x_best = denormalize(final_res.x, bounds)
at, bt, ac, bc = x_best

print("\n=== Final optimized design ===")
print(f"Twist: {at:.2f}·(r/R) + {bt:.2f}°")
print(f"Chord: c/R = {bc:.4f} + {ac:.4f}·(r/R)")

# Build final rotor for detailed analysis
twst_func = lambda r_R: at*r_R + bt
c_R_func = lambda r_R: bc + ac*r_R

rotor = Rotor(
    c_R_func=c_R_func,
    twst_func=twst_func,
    pitch=0,
    B=B,
    R=R,  # Added R parameter
    r_R_H=r_R_H,
    polar_path=polar_path,
    J=J,
    Vinf=Uinf,  # Changed from Uinf to Vinf
    rho=rho,
    n_elem=n_elem,
    dist_elem=dist_elem,
    isPropeller=isPropeller
)

# Get results directly from rotor attributes
final_power = rotor.P
thrust = rotor.T
torque = rotor.Q

print(f"\nAerodynamic performance (from rotor):")
print(f"  Power: {final_power:.2f} W")
if final_power < 0:
    print(f"  ✓ Harvesting: {abs(final_power):.2f} W extracted")
else:
    print(f"  ✗ Consuming power: {final_power:.2f} W")

# Extract alphas directly from rotor attribute
alphas = rotor.alpha

if len(alphas) > 0:
    print(f"Alpha range: {np.min(alphas):.2f}° to {np.max(alphas):.2f}°")
    # Double-check constraint
    if np.any(alphas < -10) or np.any(alphas > 25):
        print("WARNING: Final design violates alpha bounds!")
    else:
        print("✓ All alphas within [-10°, 25°]")

# Optional plot
rotor.plot_check()