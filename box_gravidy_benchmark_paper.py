#!/usr/bin/env python3
"""
GRAVIDY–box Benchmark for Paper
Paper-grade benchmarking with multi-seed averaging and proper metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from utils.box_objective import create_box_test_problem
from solver.gravidy_box import GRAVIDY_box
from solver.apgd_box import APGD_box


def compute_kkt_residual(x, problem):
    """Compute KKT residual: ||x - Π_C(x - ∇f(x))||_2"""
    grad = problem.grad(x)
    projected = problem.project(x - grad)
    return np.linalg.norm(x - projected)


def run_single_trial(problem, x_star, eta=50.0, max_iters=250, seed=0, tol_kkt=1e-6):
    """Run single trial for all methods."""
    np.random.seed(seed)
    
    results = {}
    
    # GRAVIDY–box
    start_time = time.perf_counter()
    x_grav, hist_grav = GRAVIDY_box(problem, eta=eta, max_outer=max_iters, 
                                   tol_grad=tol_kkt, verbose=False)
    time_grav = time.perf_counter() - start_time
    
    kkt_grav = [entry[2] for entry in hist_grav]  # grad_norm approximates KKT
    f_grav = [entry[1] for entry in hist_grav]    # objective values
    times_grav = [entry[3] for entry in hist_grav]  # times
    
    # Compute actual KKT residual for final point
    grad_final = problem.grad(x_grav)
    projected_final = problem.project(x_grav - grad_final)
    kkt_final = np.linalg.norm(x_grav - projected_final)
    
    results['gravidy'] = {
        'kkt': kkt_grav,
        'objective': f_grav,
        'times': times_grav,
        'final_error': np.linalg.norm(x_grav - x_star),
        'final_kkt': kkt_final,
        'converged': kkt_final <= tol_kkt
    }
    
    # APGD-box (Nesterov)
    start_time = time.perf_counter()
    x_apgd, hist_apgd = APGD_box(problem, max_iters=max_iters, 
                                tol_grad=tol_kkt, verbose=False)
    time_apgd = time.perf_counter() - start_time
    
    kkt_apgd = [entry[2] for entry in hist_apgd]
    f_apgd = [entry[1] for entry in hist_apgd]
    times_apgd = [entry[3] for entry in hist_apgd]
    
    # Compute actual KKT residual for final point
    grad_final = problem.grad(x_apgd)
    projected_final = problem.project(x_apgd - grad_final)
    kkt_final = np.linalg.norm(x_apgd - projected_final)
    
    results['apgd'] = {
        'kkt': kkt_apgd,
        'objective': f_apgd,
        'times': times_apgd,
        'final_error': np.linalg.norm(x_apgd - x_star),
        'final_kkt': kkt_final,
        'converged': kkt_final <= tol_kkt
    }
    
    return results


def run_multi_seed_benchmark(n=100, m=100, eta=50.0, max_iters=250, 
                            n_trials=10, tol_kkt=1e-6):
    """Run benchmark with multiple seeds and compute statistics."""
    print(f"\n=== GRAVIDY–box Paper Benchmark ===")
    print(f"Problem: n={n}, m={m}, eta={eta}")
    print(f"Trials: {n_trials}, max_iters={max_iters}, KKT_tol={tol_kkt:.0e}")
    print(f"Objective: min_{{x in [lo,hi]}} 0.5 ||Ax - b||^2")
    print("-" * 70)
    
    # Generate problem (same for all trials)
    A, b, lo, hi, x_star, problem = create_box_test_problem(n, m, seed=42)
    f_star = problem.f(x_star)
    
    print(f"Ground truth objective f* = {f_star:.6e}")
    print(f"Box bounds: lo ∈ [{np.min(lo):.3f}, {np.max(lo):.3f}], hi ∈ [{np.min(hi):.3f}, {np.max(hi):.3f}]")
    print(f"Matrix A: condition number ≈ {np.linalg.cond(A):.1f}")
    print("-" * 70)
    
    # Run trials
    all_results = []
    for trial in range(n_trials):
        if trial % 2 == 0:
            print(f"Running trial {trial+1}/{n_trials}...")
        results = run_single_trial(problem, x_star, eta=eta, max_iters=max_iters, 
                                 seed=trial, tol_kkt=tol_kkt)
        all_results.append(results)
    
    # Compute statistics
    methods = ['gravidy', 'apgd']
    method_names = ['GRAVIDY–box', 'APGD-box (Nesterov)']
    
    stats = {}
    for i, method in enumerate(methods):
        # Collect final metrics
        final_errors = [r[method]['final_error'] for r in all_results]
        final_kkt = [r[method]['final_kkt'] for r in all_results]
        converged = [r[method]['converged'] for r in all_results]
        
        stats[method] = {
            'name': method_names[i],
            'final_error_mean': np.mean(final_errors),
            'final_error_std': np.std(final_errors),
            'final_kkt_mean': np.mean(final_kkt),
            'final_kkt_std': np.std(final_kkt),
            'convergence_rate': np.mean(converged)
        }
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY (averaged over {} trials)".format(n_trials))
    print("="*80)
    print(f"{'Method':<25} {'||x-x*||_2':<15} {'KKT residual':<15} {'Conv. rate':<10}")
    print("-" * 80)
    for method in methods:
        s = stats[method]
        print(f"{s['name']:<25} {s['final_error_mean']:<7.3e}±{s['final_error_std']:<6.2e} "
              f"{s['final_kkt_mean']:<7.3e}±{s['final_kkt_std']:<6.2e} {s['convergence_rate']:<10.2f}")
    print("="*80)
    
    return all_results, stats, problem, x_star, f_star


def plot_paper_results(all_results, stats, problem, x_star, f_star):
    """Create paper-quality plots with error bars."""
    methods = ['gravidy', 'apgd'] 
    colors = ['red', 'blue']
    linestyles = ['-', '--']
    method_names = ['GRAVIDY–box', 'APGD-box (Nesterov)']
    
    # Determine common iteration and time grids
    max_iters = max(len(all_results[0][method]['kkt']) for method in methods)
    max_time = max(all_results[0][method]['times'][-1] for method in methods)
    
    iter_grid = np.arange(max_iters)
    time_grid = np.linspace(0, max_time, 100)
    
    # Interpolate trajectories to common grids and compute statistics
    iter_stats = {}
    time_stats = {}
    
    for method in methods:
        # Iteration-based statistics
        kkt_matrix = np.full((len(all_results), max_iters), np.nan)
        for trial, result in enumerate(all_results):
            kkt_traj = result[method]['kkt']
            kkt_matrix[trial, :len(kkt_traj)] = kkt_traj
        
        # Fill NaN with last value
        for trial in range(len(all_results)):
            mask = ~np.isnan(kkt_matrix[trial, :])
            if np.any(mask):
                last_val = kkt_matrix[trial, mask][-1]
                kkt_matrix[trial, ~mask] = last_val
        
        iter_stats[method] = {
            'kkt_mean': np.nanmean(kkt_matrix, axis=0),
            'kkt_std': np.nanstd(kkt_matrix, axis=0)
        }
        
        # Time-based statistics  
        obj_matrix = np.full((len(all_results), len(time_grid)), np.nan)
        for trial, result in enumerate(all_results):
            times = result[method]['times']
            objectives = result[method]['objective']
            obj_interp = np.interp(time_grid, times, objectives)
            obj_matrix[trial, :] = obj_interp
        
        time_stats[method] = {
            'obj_mean': np.nanmean(obj_matrix, axis=0),
            'obj_std': np.nanstd(obj_matrix, axis=0)
        }
    
    # Create plots
    plt.style.use('default')
    
    # Figure 1: KKT residual vs iterations
    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        mean_kkt = iter_stats[method]['kkt_mean']
        std_kkt = iter_stats[method]['kkt_std']
        
        plt.semilogy(iter_grid, mean_kkt, color=colors[i], linestyle=linestyles[i], 
                    linewidth=3, label=method_names[i])
        plt.fill_between(iter_grid, mean_kkt - std_kkt, mean_kkt + std_kkt, 
                        color=colors[i], alpha=0.2)
    
    plt.xlabel('Iterations', fontweight='bold')
    plt.ylabel(r'KKT residual', fontweight='bold')
    plt.title('Box: KKT residual vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    # Save for paper
    import os
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/box_err_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 2: Objective vs time
    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        mean_obj = time_stats[method]['obj_mean']
        std_obj = time_stats[method]['obj_std']
        
        plt.loglog(time_grid, np.abs(mean_obj - f_star + 1e-16), 
                  color=colors[i], linestyle=linestyles[i], 
                  linewidth=3, label=method_names[i])
        plt.fill_between(time_grid, 
                        np.abs(mean_obj - std_obj - f_star + 1e-16),
                        np.abs(mean_obj + std_obj - f_star + 1e-16),
                        color=colors[i], alpha=0.2)
    
    plt.xlabel('Time [seconds]', fontweight='bold')
    plt.ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    plt.title('Box: objective vs time', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("figs/box_f_vs_time.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Paper-grade benchmark
    all_results, stats, problem, x_star, f_star = run_multi_seed_benchmark(
        n=100, m=100, eta=500.0, max_iters=250, 
        n_trials=10, tol_kkt=1e-6
    )
    
    # Create paper plots
    plot_paper_results(all_results, stats, problem, x_star, f_star)
