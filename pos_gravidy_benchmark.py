"""
GRAVIDY–pos benchmark script for positive orthant optimization (NNLS).

Compares:
- GRAVIDY–pos: Main method with implicit Euler in log-coordinates and damped Newton
- APGD-pos: Accelerated PGD with Nesterov acceleration (projection by clipping)
- PGD-BB-pos: Projected Barzilai-Borwein with Armijo line search
- MU-pos: Multiplicative updates (A>=0, b>=0 baseline)

All methods solve: min_{x >= 0} 0.5 ||Ax - b||^2
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Import all solvers
from utils.pos_objective import create_pos_test_problem
from solver.gravidy_pos import GRAVIDY_pos
from solver.apgd_pos import APGD_pos
from solver.pgd_bb_pos import PGD_BB_pos
from solver.mu_pos import MU_pos


def run_pos_benchmark(n=120, m=120, density=0.15, eta=30.0, max_iters=400, 
                     seed=100, verbose=False):
    """
    Run benchmark comparing GRAVIDY–pos and baselines.
    
    Args:
        n: Problem dimension (number of variables)
        m: Number of constraints/observations  
        density: Sparsity density of ground truth solution
        eta: Step size for GRAVIDY–pos method
        max_iters: Maximum iterations for all methods
        seed: Random seed for reproducibility
        verbose: Print detailed progress
        
    Returns:
        results: Dictionary with trajectories and timing for each method
    """
    print(f"\n=== GRAVIDY–pos Benchmark ===")
    print(f"Problem: n={n}, m={m}, density={density:.2f}, eta={eta}, max_iters={max_iters}")
    print(f"Objective: min_{{x >= 0}} 0.5 ||Ax - b||^2")
    
    # Create test problem
    A, b, x_star, problem = create_pos_test_problem(n, m, density=density, seed=seed)
    f_star = problem.f(x_star)
    
    print(f"Ground truth objective f* = {f_star:.6e}")
    print(f"Ground truth sparsity: {np.sum(x_star > 1e-10)}/{n} non-zero components")
    print(f"Matrix A: min={np.min(A):.3f}, max={np.max(A):.3f}")
    print(f"Vector b: min={np.min(b):.3f}, max={np.max(b):.3f}")
    print("-" * 70)
    
    results = {}
    
    # ---- GRAVIDY–pos ----
    print("Running GRAVIDY–pos...")
    t0 = time.perf_counter()
    x_grav, hist_grav = GRAVIDY_pos(problem, eta=eta, max_outer=max_iters, 
                                   tol_grad=1e-10, verbose=verbose)
    t_grav = time.perf_counter() - t0
    
    # ---- APGD-pos baseline ----
    print("Running APGD-pos (Nesterov accelerated)...")
    t0 = time.perf_counter()
    x_apgd, hist_apgd = APGD_pos(problem, max_iters=max_iters, tol_grad=1e-10,
                                 verbose=verbose)
    t_apgd = time.perf_counter() - t0
    
    # ---- Proj-BB baseline ----
    print("Running Proj-BB (Barzilai-Borwein)...")
    t0 = time.perf_counter()
    x_bb, hist_bb = PGD_BB_pos(problem, max_iters=max_iters, tol_grad=1e-10,
                               verbose=verbose)
    t_bb = time.perf_counter() - t0
    
    # ---- MU baseline ----
    print("Running MU (Multiplicative Updates)...")
    t0 = time.perf_counter()
    x_mu, hist_mu = MU_pos(problem, max_iters=max_iters, tol_grad=1e-10,
                           verbose=verbose)
    t_mu = time.perf_counter() - t0
    
    # Extract final results
    def get_final_stats(x_final, hist):
        err = np.linalg.norm(x_final - x_star)
        f_final = problem.f(x_final)
        f_gap = f_final - f_star
        iters = len(hist)
        return err, f_gap, iters
    
    err_grav, gap_grav, iters_grav = get_final_stats(x_grav, hist_grav)
    err_apgd, gap_apgd, iters_apgd = get_final_stats(x_apgd, hist_apgd)
    err_bb, gap_bb, iters_bb = get_final_stats(x_bb, hist_bb)
    err_mu, gap_mu, iters_mu = get_final_stats(x_mu, hist_mu)
    
    # Print summary
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Method':<25} {'||x-x*||_2':<12} {'f(x)-f*':<12} {'Iters':<8} {'Time [s]':<10}")
    print("-" * 75)
    print(f"{'GRAVIDY–pos':<25} {err_grav:<12.3e} {gap_grav:<12.3e} {iters_grav:<8} {t_grav:<10.3f}")
    print(f"{'PGD+Nesterov':<25} {err_apgd:<12.3e} {gap_apgd:<12.3e} {iters_apgd:<8} {t_apgd:<10.3f}")
    print(f"{'Proj-BB (Armijo)':<25} {err_bb:<12.3e} {gap_bb:<12.3e} {iters_bb:<8} {t_bb:<10.3f}")
    print(f"{'MU (A≥0,b≥0)':<25} {err_mu:<12.3e} {gap_mu:<12.3e} {iters_mu:<8} {t_mu:<10.3f}")
    print("=" * 75)
    
    # Store results
    results = {
        'problem': {'A': A, 'b': b, 'x_star': x_star, 'f_star': f_star},
        'gravidy_pos': {'x': x_grav, 'hist': hist_grav, 'time': t_grav},
        'apgd_pos': {'x': x_apgd, 'hist': hist_apgd, 'time': t_apgd},
        'pgd_bb_pos': {'x': x_bb, 'hist': hist_bb, 'time': t_bb},
        'mu_pos': {'x': x_mu, 'hist': hist_mu, 'time': t_mu}
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
    it_grav, f_grav, g_grav, t_grav = extract_arrays(results['gravidy_pos']['hist'])
    it_apgd, f_apgd, g_apgd, t_apgd = extract_arrays(results['apgd_pos']['hist'])
    it_bb, f_bb, g_bb, t_bb = extract_arrays(results['pgd_bb_pos']['hist'])
    it_mu, f_mu, g_mu, t_mu = extract_arrays(results['mu_pos']['hist'])
    
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
    ax.semilogy(it_grav, np.abs(f_grav - f_star), 'r-', linewidth=3, label='GRAVIDY–pos (implicit Newton)')
    ax.semilogy(it_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='PGD+Nesterov')
    ax.semilogy(it_bb, np.abs(f_bb - f_star), 'g:', linewidth=3, label='Proj-BB (Armijo)')
    ax.semilogy(it_mu, np.abs(f_mu - f_star), 'm-.', linewidth=3, label='MU (A≥0,b≥0)')
    ax.set_xlabel('Iterations', fontweight='bold')
    ax.set_ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    ax.set_title('Objective Gap vs Iterations', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Gradient norm vs iterations  
    ax = axes[0, 1]
    ax.semilogy(it_grav, g_grav, 'r-', linewidth=3, label='GRAVIDY–pos (implicit Newton)')
    ax.semilogy(it_apgd, g_apgd, 'b--', linewidth=3, label='PGD+Nesterov')
    ax.semilogy(it_bb, g_bb, 'g:', linewidth=3, label='Proj-BB (Armijo)')
    ax.semilogy(it_mu, g_mu, 'm-.', linewidth=3, label='MU (A≥0,b≥0)')
    ax.set_xlabel('Iterations', fontweight='bold')
    ax.set_ylabel(r'$\|\nabla f(x_k)\|_2$', fontweight='bold')
    ax.set_title('Gradient Norm vs Iterations', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Objective gap vs time (loglog)
    ax = axes[1, 0]
    ax.loglog(t_grav, np.abs(f_grav - f_star), 'r-', linewidth=3, label='GRAVIDY–pos (implicit Newton)')
    ax.loglog(t_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='PGD+Nesterov')
    ax.loglog(t_bb, np.abs(f_bb - f_star), 'g:', linewidth=3, label='Proj-BB (Armijo)')
    ax.loglog(t_mu, np.abs(f_mu - f_star), 'm-.', linewidth=3, label='MU (A≥0,b≥0)')
    ax.set_xlabel('Time [seconds]', fontweight='bold')
    ax.set_ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    ax.set_title('Objective Gap vs Time (loglog)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Final solution comparison
    ax = axes[1, 1]
    n = len(x_star)
    indices = np.arange(n)
    width = 0.2
    
    ax.bar(indices - 1.5*width, results['gravidy_pos']['x'], width, 
           label='GRAVIDY–pos', alpha=0.8, color='red')
    ax.bar(indices - 0.5*width, results['apgd_pos']['x'], width,
           label='PGD+Nesterov', alpha=0.8, color='blue')
    ax.bar(indices + 0.5*width, results['pgd_bb_pos']['x'], width,
           label='Proj-BB', alpha=0.8, color='green')
    ax.bar(indices + 1.5*width, results['mu_pos']['x'], width,
           label='MU', alpha=0.8, color='magenta')
    ax.plot(indices, x_star, 'ko-', linewidth=2, markersize=4, 
            label=r'$x^*$ (true)', alpha=0.7)
    
    ax.set_xlabel('Component Index', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Final Solutions on Positive Orthant', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figures for LaTeX
    import os
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/pos_benchmark.pdf", bbox_inches="tight")
    plt.show()
    
    # Create individual plots for systematic reporting
    
    # Figure 1: Objective gap vs iterations
    plt.figure(figsize=(7, 5))
    plt.semilogy(it_grav, np.abs(f_grav - f_star), 'r-', linewidth=3, label='GRAVIDY–pos (implicit Newton)')
    plt.semilogy(it_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='PGD+Nesterov')
    plt.semilogy(it_bb, np.abs(f_bb - f_star), 'g:', linewidth=3, label='Proj-BB (Armijo)')
    plt.semilogy(it_mu, np.abs(f_mu - f_star), 'm-.', linewidth=3, label='MU (A≥0,b≥0)')
    plt.xlabel('Iterations', fontweight='bold')
    plt.ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    plt.title('NNLS (positive): objective gap', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.savefig("figs/pos_f_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 2: Objective vs time
    plt.figure(figsize=(7, 5))
    plt.loglog(t_grav, np.abs(f_grav - f_star), 'r-', linewidth=3, label='GRAVIDY–pos (implicit Newton)')
    plt.loglog(t_apgd, np.abs(f_apgd - f_star), 'b--', linewidth=3, label='PGD+Nesterov')
    plt.loglog(t_bb, np.abs(f_bb - f_star), 'g:', linewidth=3, label='Proj-BB (Armijo)')
    plt.loglog(t_mu, np.abs(f_mu - f_star), 'm-.', linewidth=3, label='MU (A≥0,b≥0)')
    plt.xlabel('Time [seconds]', fontweight='bold')
    plt.ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    plt.title('NNLS (positive): objective vs time', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.savefig("figs/pos_f_vs_time.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Test multiple eta values
    eta_values = [10.0, 30.0, 50.0, 100.0, 200.0]
    
    print("Testing GRAVIDY–pos with different eta values:")
    print("=" * 60)
    
    for eta in eta_values:
        print(f"\n--- Testing eta = {eta} ---")
        results = run_pos_benchmark(n=120, m=120, density=0.15, eta=eta, 
                                   max_iters=400, seed=100, verbose=False)
        
        # Extract final errors
        err_grav = np.linalg.norm(results['gravidy_pos']['x'] - results['problem']['x_star'])
        err_apgd = np.linalg.norm(results['apgd_pos']['x'] - results['problem']['x_star'])
        err_bb = np.linalg.norm(results['pgd_bb_pos']['x'] - results['problem']['x_star'])
        err_mu = np.linalg.norm(results['mu_pos']['x'] - results['problem']['x_star'])
        
        print(f"GRAVIDY–pos final error: {err_grav:.3e}")
        print(f"PGD+Nesterov final error: {err_apgd:.3e}")
        print(f"Proj-BB final error: {err_bb:.3e}")
        print(f"MU final error: {err_mu:.3e}")
    
    # Run final benchmark with highest eta for detailed plots
    print(f"\n--- Final detailed run with eta = {eta_values[-1]} ---")
    final_results = run_pos_benchmark(n=120, m=120, density=0.15, eta=eta_values[-1], 
                                     max_iters=400, seed=100, verbose=False)
    
    # Create plots
    plot_results(final_results)
