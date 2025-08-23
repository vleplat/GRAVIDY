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
    
    # ---- GRAVIDY–box ----
    print("Running GRAVIDY–box...")
    t0 = time.perf_counter()
    x_grav, hist_grav = GRAVIDY_box(problem, eta=eta, max_outer=max_iters, 
                                   tol_grad=1e-10, verbose=verbose)
    t_grav = time.perf_counter() - t0
    
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
    
    err_grav, gap_grav, iters_grav = get_final_stats(x_grav, hist_grav)
    err_apgd, gap_apgd, iters_apgd = get_final_stats(x_apgd, hist_apgd)
    
    # Print summary
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Method':<25} {'||x-x*||_2':<12} {'f(x)-f*':<12} {'Iters':<8} {'Time [s]':<10}")
    print("-" * 75)
    print(f"{'GRAVIDY–box':<25} {err_grav:<12.3e} {gap_grav:<12.3e} {iters_grav:<8} {t_grav:<10.3f}")
    print(f"{'APGD-box (Nesterov)':<25} {err_apgd:<12.3e} {gap_apgd:<12.3e} {iters_apgd:<8} {t_apgd:<10.3f}")
    print("=" * 75)
    
    # Store results
    results = {
        'problem': {'A': A, 'b': b, 'lo': lo, 'hi': hi, 'x_star': x_star, 'f_star': f_star},
        'gravidy_box': {'x': x_grav, 'hist': hist_grav, 'time': t_grav},
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
    it_grav, f_grav, g_grav, t_grav = extract_arrays(results['gravidy_box']['hist'])
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
    ax.semilogy(it_grav, np.abs(f_grav - f_star), 'r-', linewidth=3, label='GRAVIDY–box')
    ax.semilogy(it_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='APGD-box (Nesterov)')
    ax.set_xlabel('Iterations', fontweight='bold')
    ax.set_ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    ax.set_title('Objective Gap vs Iterations', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Gradient norm vs iterations  
    ax = axes[0, 1]
    ax.semilogy(it_grav, g_grav, 'r-', linewidth=3, label='GRAVIDY–box')
    ax.semilogy(it_apgd, g_apgd, 'b--', linewidth=3, label='APGD-box (Nesterov)')
    ax.set_xlabel('Iterations', fontweight='bold')
    ax.set_ylabel(r'$\|\nabla f(x_k)\|_2$', fontweight='bold')
    ax.set_title('Gradient Norm vs Iterations', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Objective gap vs time (loglog)
    ax = axes[1, 0]
    ax.loglog(t_grav, np.abs(f_grav - f_star), 'r-', linewidth=3, label='GRAVIDY–box')
    ax.loglog(t_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='APGD-box (Nesterov)')
    ax.set_xlabel('Time [seconds]', fontweight='bold')
    ax.set_ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    ax.set_title('Objective Gap vs Time (loglog)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Final solution comparison
    ax = axes[1, 1]
    n = len(x_star)
    indices = np.arange(n)
    width = 0.35
    
    ax.bar(indices - width/2, results['gravidy_box']['x'], width, 
           label='GRAVIDY–box', alpha=0.8, color='red')
    ax.bar(indices + width/2, results['apgd_box']['x'], width,
           label='APGD-box (Nesterov)', alpha=0.8, color='blue')
    ax.plot(indices, x_star, 'ko-', linewidth=2, markersize=4, 
            label=r'$x^*$ (true)', alpha=0.7)
    
    ax.set_xlabel('Component Index', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Final Solutions on Box Constraints', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figures for LaTeX
    import os
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/box_benchmark.pdf", bbox_inches="tight")
    plt.show()
    
    # Create individual plots for systematic reporting
    
    # Figure 1: Error vs iterations
    plt.figure(figsize=(7, 5))
    err_grav = compute_errors(results['gravidy_box']['hist'], results['gravidy_box']['x'])
    err_apgd = compute_errors(results['apgd_box']['hist'], results['apgd_box']['x'])
    
    it_grav, f_grav, g_grav, t_grav = extract_arrays(results['gravidy_box']['hist'])
    it_apgd, f_apgd, g_apgd, t_apgd = extract_arrays(results['apgd_box']['hist'])
    
    plt.semilogy(it_grav, err_grav, 'r-', linewidth=3, label='GRAVIDY–box')
    plt.semilogy(it_apgd, err_apgd, 'b--', linewidth=3, label='APGD-box (Nesterov)')
    plt.xlabel('Iterations', fontweight='bold')
    plt.ylabel(r'$\|x_k - x^*\|_2$', fontweight='bold')
    plt.title('Box: error vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.savefig("figs/box_err_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 2: Objective vs time
    plt.figure(figsize=(7, 5))
    plt.loglog(t_grav, np.abs(f_grav - f_star), 'r-', linewidth=3, label='GRAVIDY–box')
    plt.loglog(t_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='APGD-box (Nesterov)')
    plt.xlabel('Time [seconds]', fontweight='bold')
    plt.ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    plt.title('Box: objective vs time', fontweight='bold')
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
        err_grav = np.linalg.norm(results['gravidy_box']['x'] - results['problem']['x_star'])
        err_apgd = np.linalg.norm(results['apgd_box']['x'] - results['problem']['x_star'])
        
        print(f"GRAVIDY–box final error: {err_grav:.3e}")
        print(f"APGD-box final error: {err_apgd:.3e}")
    
    # Run final benchmark with highest eta for detailed plots
    print(f"\n--- Final detailed run with eta = {eta_values[-1]} ---")
    final_results = run_box_benchmark(n=100, m=100, eta=eta_values[-1], 
                                     max_iters=250, seed=20, verbose=False)
    
    # Create plots
    plot_results(final_results)
