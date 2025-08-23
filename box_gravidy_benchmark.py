"""
GRAVIDY–box benchmark script.

Compares:
- GRAVIDY–box: Main method with implicit reparameterization and damped Newton
- APGD-box: Accelerated PGD with Nesterov acceleration (external baseline)

All methods solve least squares on box constraints: min_{x in [lo,hi]} 0.5 ||Ax - b||^2
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Import all solvers
from utils.box_objective import create_box_test_problem
from solver.gravidy_box import GRAVIDY_box
from solver.apgd_box import APGD_box


def run_box_benchmark(n=100, m=100, eta=10.0, max_iters=250, 
                     seed=20, verbose=False):
    """
    Run benchmark comparing GRAVIDY–box and APGD baseline.
    
    Args:
        n: Problem dimension (number of variables)
        m: Number of constraints/observations  
        eta: Step size for GRAVIDY–box method
        max_iters: Maximum iterations for all methods
        seed: Random seed for reproducibility
        verbose: Print detailed progress
        
    Returns:
        results: Dictionary with trajectories and timing for each method
    """
    print(f"\n=== GRAVIDY–box Benchmark ===")
    print(f"Problem: n={n}, m={m}, eta={eta}, max_iters={max_iters}")
    print(f"Objective: min_{{x in [lo,hi]}} 0.5 ||Ax - b||^2")
    
    # Create test problem
    A, b, lo, hi, x_star, problem = create_box_test_problem(n, m, seed=seed)
    f_star = problem.f(x_star)
    
    print(f"Ground truth objective f* = {f_star:.6e}")
    print(f"Box bounds: lo ∈ [{np.min(lo):.3f}, {np.max(lo):.3f}], hi ∈ [{np.min(hi):.3f}, {np.max(hi):.3f}]")
    print("-" * 70)
    
    results = {}
    
    # ---- GRAVIDY–box (Newton) ----
    print("Running GRAVIDY–box (Newton)...")
    t0 = time.perf_counter()
    x_grav_newton, hist_grav_newton = GRAVIDY_box(problem, eta=eta, max_outer=max_iters, 
                                                 tol_grad=1e-10, inner='newton', verbose=verbose)
    t_grav_newton = time.perf_counter() - t0
    
    # ---- GRAVIDY–box (MGN) ----
    print("Running GRAVIDY–box (MGN)...")
    t0 = time.perf_counter()
    x_grav_mgn, hist_grav_mgn = GRAVIDY_box(problem, eta=eta, max_outer=max_iters, 
                                           tol_grad=1e-10, inner='mgn', verbose=verbose)
    t_grav_mgn = time.perf_counter() - t0
    
    # ---- APGD baseline ----
    print("Running APGD-box (Nesterov accelerated)...")
    t0 = time.perf_counter()
    x_apgd, hist_apgd = APGD_box(problem, max_iters=max_iters, tol_grad=1e-10,
                                 verbose=verbose)
    t_apgd = time.perf_counter() - t0
    
    # Extract final results
    def get_final_stats(x_final, hist):
        err = np.linalg.norm(x_final - x_star)
        f_final = problem.f(x_final)
        f_gap = f_final - f_star
        iters = len(hist)
        return err, f_gap, iters
    
    err_grav_newton, gap_grav_newton, iters_grav_newton = get_final_stats(x_grav_newton, hist_grav_newton)
    err_grav_mgn, gap_grav_mgn, iters_grav_mgn = get_final_stats(x_grav_mgn, hist_grav_mgn)
    err_apgd, gap_apgd, iters_apgd = get_final_stats(x_apgd, hist_apgd)
    
    # Print summary
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Method':<25} {'||x-x*||_2':<12} {'f(x)-f*':<12} {'Iters':<8} {'Time [s]':<10}")
    print("-" * 75)
    print(f"{'GRAVIDY–box (Newton)':<25} {err_grav_newton:<12.3e} {gap_grav_newton:<12.3e} {iters_grav_newton:<8} {t_grav_newton:<10.3f}")
    print(f"{'GRAVIDY–box (MGN)':<25} {err_grav_mgn:<12.3e} {gap_grav_mgn:<12.3e} {iters_grav_mgn:<8} {t_grav_mgn:<10.3f}")
    print(f"{'APGD-box (Nesterov)':<25} {err_apgd:<12.3e} {gap_apgd:<12.3e} {iters_apgd:<8} {t_apgd:<10.3f}")
    print("=" * 75)
    
    # Store results
    results = {
        'problem': {'A': A, 'b': b, 'lo': lo, 'hi': hi, 'x_star': x_star, 'f_star': f_star},
        'gravidy_box_newton': {'x': x_grav_newton, 'hist': hist_grav_newton, 'time': t_grav_newton},
        'gravidy_box_mgn': {'x': x_grav_mgn, 'hist': hist_grav_mgn, 'time': t_grav_mgn},
        'apgd_box': {'x': x_apgd, 'hist': hist_apgd, 'time': t_apgd}
    }
    
    return results


def plot_results(results):
    """Create comprehensive plots of the benchmark results."""
    
    # Extract data
    x_star = results['problem']['x_star']
    f_star = results['problem']['f_star']
    
    # Convert histories to arrays
    def extract_arrays(hist):
        iters = [h[0] for h in hist]
        objs = [h[1] for h in hist] 
        grads = [h[2] for h in hist]
        times = [h[3] for h in hist]
        return np.array(iters), np.array(objs), np.array(grads), np.array(times)
    
    # Note: For simplicity, computing errors properly would require storing x trajectory
    it_grav_newton, f_grav_newton, g_grav_newton, t_grav_newton = extract_arrays(results['gravidy_box_newton']['hist'])
    it_grav_mgn, f_grav_mgn, g_grav_mgn, t_grav_mgn = extract_arrays(results['gravidy_box_mgn']['hist'])
    it_apgd, f_apgd, g_apgd, t_apgd = extract_arrays(results['apgd_box']['hist'])
    
    # Set up plotting
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titleweight': 'bold'
    })
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Objective gap vs iterations
    ax = axes[0, 0]
    ax.semilogy(it_grav_newton, np.abs(f_grav_newton - f_star), 'r-', linewidth=3, label='GRAVIDY–box (Newton)')
    ax.semilogy(it_grav_mgn, np.abs(f_grav_mgn - f_star), 'g-', linewidth=3, label='GRAVIDY–box (MGN)')
    ax.semilogy(it_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='APGD-box (Nesterov)')
    ax.set_xlabel('Iterations', fontweight='bold')
    ax.set_ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    ax.set_title('Objective Gap vs Iterations', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Final KKT residuals (bar chart)
    ax = axes[0, 1]
    # Compute actual KKT residuals for final points
    A, b = results['problem']['A'], results['problem']['b']
    lo, hi = results['problem']['lo'], results['problem']['hi']
    
    # For box constraints: project(x) = clip(x, lo, hi)
    def compute_kkt_residual(x_final):
        grad = A.T @ (A @ x_final - b)  # gradient of 0.5||Ax-b||^2
        projected = np.clip(x_final - grad, lo, hi)  # project onto box
        return np.linalg.norm(x_final - projected)
    
    kkt_values = [
        compute_kkt_residual(results['gravidy_box_newton']['x']),
        compute_kkt_residual(results['gravidy_box_mgn']['x']),
        compute_kkt_residual(results['apgd_box']['x'])
    ]
    
    methods = ['GRAVIDY–box (Newton)', 'GRAVIDY–box (MGN)', 'APGD-box']
    colors = ['red', 'green', 'blue']
    
    bars = ax.bar(methods, kkt_values, color=colors, alpha=0.8)
    ax.set_ylabel(r'KKT residual $\|x - \Pi_C(x - \nabla f(x))\|_2$', fontweight='bold')
    ax.set_title('Final KKT Residuals', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Add value labels on bars
    for bar, value in zip(bars, kkt_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2e}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Final solution comparison
    ax = axes[1, 0]
    n = len(x_star)
    indices = np.arange(n)
    width = 0.35
    
    ax.bar(indices - width, results['gravidy_box_newton']['x'], width, 
           label='GRAVIDY–box (Newton)', alpha=0.8, color='red')
    ax.bar(indices, results['gravidy_box_mgn']['x'], width,
           label='GRAVIDY–box (MGN)', alpha=0.8, color='green')
    ax.bar(indices + width, results['apgd_box']['x'], width,
           label='APGD-box (Nesterov)', alpha=0.8, color='blue')
    ax.plot(indices, x_star, 'ko-', linewidth=2, markersize=4, 
            label=r'$x^*$ (true)', alpha=0.7)
    
    ax.set_xlabel('Component Index', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Final Solutions on Box', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Objective gap vs time (loglog)
    ax = axes[1, 1]
    ax.loglog(t_grav_newton, np.abs(f_grav_newton - f_star), 'r-', linewidth=3, label='GRAVIDY–box (Newton)')
    ax.loglog(t_grav_mgn, np.abs(f_grav_mgn - f_star), 'g-', linewidth=3, label='GRAVIDY–box (MGN)')
    ax.loglog(t_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='APGD-box (Nesterov)')
    ax.set_xlabel('Time [seconds]', fontweight='bold')
    ax.set_ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    ax.set_title('Objective Gap vs Time (loglog)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figures for LaTeX
    import os
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/box_benchmark.pdf", bbox_inches="tight")
    plt.show()
    
    # Create individual plots for systematic reporting
    
    # Figure 1: Objective gap vs iterations
    plt.figure(figsize=(7, 5))
    it_grav_newton, f_grav_newton, g_grav_newton, t_grav_newton = extract_arrays(results['gravidy_box_newton']['hist'])
    it_grav_mgn, f_grav_mgn, g_grav_mgn, t_grav_mgn = extract_arrays(results['gravidy_box_mgn']['hist'])
    it_apgd, f_apgd, g_apgd, t_apgd = extract_arrays(results['apgd_box']['hist'])
    
    plt.semilogy(it_grav_newton, np.abs(f_grav_newton - f_star), 'r-', linewidth=3, label='GRAVIDY–box (Newton)')
    plt.semilogy(it_grav_mgn, np.abs(f_grav_mgn - f_star), 'g-', linewidth=3, label='GRAVIDY–box (MGN)')
    plt.semilogy(it_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='APGD-box (Nesterov)')
    plt.xlabel('Iterations', fontweight='bold')
    plt.ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    plt.title('Box: objective gap vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.savefig("figs/box_err_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 2: Objective gap vs time
    plt.figure(figsize=(7, 5))
    plt.loglog(t_grav_newton, np.abs(f_grav_newton - f_star), 'r-', linewidth=3, label='GRAVIDY–box (Newton)')
    plt.loglog(t_grav_mgn, np.abs(f_grav_mgn - f_star), 'g-', linewidth=3, label='GRAVIDY–box (MGN)')
    plt.loglog(t_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='APGD-box (Nesterov)')
    plt.xlabel('Time [seconds]', fontweight='bold')
    plt.ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    plt.title('Box: objective gap vs time', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.savefig("figs/box_f_vs_time.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Test multiple eta values
    eta_values = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    print("Testing GRAVIDY–box with different eta values:")
    print("=" * 60)
    
    for eta in eta_values:
        print(f"\n--- Testing eta = {eta} ---")
        results = run_box_benchmark(n=100, m=100, eta=eta, 
                                   max_iters=250, seed=20, verbose=False)
        
        # Extract final errors
        err_grav_newton = np.linalg.norm(results['gravidy_box_newton']['x'] - results['problem']['x_star'])
        err_grav_mgn = np.linalg.norm(results['gravidy_box_mgn']['x'] - results['problem']['x_star'])
        err_apgd = np.linalg.norm(results['apgd_box']['x'] - results['problem']['x_star'])
        
        print(f"GRAVIDY–box (Newton) final error: {err_grav_newton:.3e}")
        print(f"GRAVIDY–box (MGN) final error: {err_grav_mgn:.3e}")
        print(f"APGD-box final error: {err_apgd:.3e}")
    
    # Run final benchmark with highest eta for detailed plots
    print(f"\n--- Final detailed run with eta = {eta_values[-1]} ---")
    final_results = run_box_benchmark(n=100, m=100, eta=eta_values[-1], 
                                     max_iters=250, seed=20, verbose=False)
    
    # Create plots
    plot_results(final_results)
