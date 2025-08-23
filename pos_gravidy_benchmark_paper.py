#!/usr/bin/env python3
"""
GRAVIDY–pos Benchmark for Paper
Paper-grade benchmarking with multi-seed averaging and proper metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from utils.pos_objective import create_pos_test_problem
from solver.gravidy_pos import GRAVIDY_pos
from solver.apgd_pos import APGD_pos
from solver.pgd_bb_pos import PGD_BB_pos
from solver.mu_pos import MU_pos


def compute_kkt_residual(x, problem):
    """Compute KKT residual: ||x - Π_C(x - ∇f(x))||_2"""
    grad = problem.grad(x)
    projected = problem.project(x - grad)
    return np.linalg.norm(x - projected)


def run_single_trial(problem, x_star, eta=30.0, max_iters=400, seed=0, tol_kkt=1e-8):
    """Run single trial for all methods."""
    np.random.seed(seed)
    
    results = {}
    
    # Initial point (same for all methods) - use fixed seed for reproducibility
    np.random.seed(seed)
    x0 = np.maximum(np.random.randn(problem.n), 0.1)
    
    # GRAVIDY–pos
    start_time = time.perf_counter()
    x_grav, hist_grav = GRAVIDY_pos(problem, eta=eta, max_outer=max_iters, 
                                   tol_grad=tol_kkt, verbose=False)
    time_grav = time.perf_counter() - start_time
    
    # Extract trajectories with KKT residuals
    # Note: we need to run the algorithm again to collect trajectory points
    # or modify solvers to return full trajectory - for now use final point
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
    
    # PGD+Nesterov
    start_time = time.perf_counter()
    x_apgd, hist_apgd = APGD_pos(problem, max_iters=max_iters, x0=x0.copy(),
                                tol_grad=tol_kkt, verbose=False)
    time_apgd = time.perf_counter() - start_time
    
    kkt_apgd = [entry[2] for entry in hist_apgd]  # grad_norm approximates KKT
    f_apgd = [entry[1] for entry in hist_apgd]    # objective values
    times_apgd = [entry[3] for entry in hist_apgd]  # times
    
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
    
    # Projected BB
    start_time = time.perf_counter()
    x_bb, hist_bb = PGD_BB_pos(problem, max_iters=max_iters, x0=x0.copy(),
                              tol_grad=tol_kkt, verbose=False)
    time_bb = time.perf_counter() - start_time
    
    kkt_bb = [entry[2] for entry in hist_bb]  # grad_norm approximates KKT
    f_bb = [entry[1] for entry in hist_bb]    # objective values
    times_bb = [entry[3] for entry in hist_bb]  # times
    
    # Compute actual KKT residual for final point
    grad_final = problem.grad(x_bb)
    projected_final = problem.project(x_bb - grad_final)
    kkt_final = np.linalg.norm(x_bb - projected_final)
    
    results['bb'] = {
        'kkt': kkt_bb,
        'objective': f_bb,
        'times': times_bb,
        'final_error': np.linalg.norm(x_bb - x_star),
        'final_kkt': kkt_final,
        'converged': kkt_final <= tol_kkt
    }
    
    # Multiplicative Updates
    start_time = time.perf_counter()
    x_mu, hist_mu = MU_pos(problem, max_iters=max_iters, x0=x0.copy(),
                          tol_grad=tol_kkt, verbose=False)
    time_mu = time.perf_counter() - start_time
    
    kkt_mu = [entry[2] for entry in hist_mu]  # grad_norm approximates KKT
    f_mu = [entry[1] for entry in hist_mu]    # objective values
    times_mu = [entry[3] for entry in hist_mu]  # times
    
    # Compute actual KKT residual for final point
    grad_final = problem.grad(x_mu)
    projected_final = problem.project(x_mu - grad_final)
    kkt_final = np.linalg.norm(x_mu - projected_final)
    
    results['mu'] = {
        'kkt': kkt_mu,
        'objective': f_mu,
        'times': times_mu,
        'final_error': np.linalg.norm(x_mu - x_star),
        'final_kkt': kkt_final,
        'converged': kkt_final <= tol_kkt
    }
    
    return results


def run_multi_seed_benchmark(n=120, m=120, density=0.15, eta=30.0, max_iters=400, 
                            n_trials=10, tol_kkt=1e-8):
    """Run benchmark with multiple seeds and compute statistics."""
    print(f"\n=== GRAVIDY–pos Paper Benchmark ===")
    print(f"Problem: n={n}, m={m}, density={density:.2f}, eta={eta}")
    print(f"Trials: {n_trials}, max_iters={max_iters}, KKT_tol={tol_kkt:.0e}")
    print(f"Objective: min_{{x >= 0}} 0.5 ||Ax - b||^2")
    print("-" * 70)
    
    # Generate problem (same for all trials)
    A, b, x_star, problem = create_pos_test_problem(n, m, density=density, seed=100)
    f_star = problem.f(x_star)
    
    print(f"Ground truth objective f* = {f_star:.6e}")
    print(f"Ground truth sparsity: {np.sum(x_star > 1e-12)}/{n} non-zero components")
    print(f"Matrix A: min={np.min(A):.3f}, max={np.max(A):.3f}")
    print(f"Vector b: min={np.min(b):.3f}, max={np.max(b):.3f}")
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
    methods = ['gravidy', 'apgd', 'bb', 'mu']
    method_names = ['GRAVIDY–pos', 'PGD+Nesterov', 'Proj-BB (Armijo)', 'MU (A≥0,b≥0)']
    
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
    methods = ['gravidy', 'apgd', 'bb', 'mu'] 
    colors = ['red', 'blue', 'green', 'magenta']
    linestyles = ['-', '--', ':', '-.']
    method_names = ['GRAVIDY–pos (implicit Newton)', 'PGD+Nesterov', 
                   'Proj-BB (Armijo)', 'MU (A≥0,b≥0)']
    
    # Determine common iteration grid
    max_iters = max(len(all_results[0][method]['kkt']) for method in methods)
    iter_grid = np.arange(max_iters)
    
    # Interpolate trajectories to common grids and compute statistics
    iter_stats = {}
    time_stats = {}
    
    for method in methods:
        # Iteration-based statistics
        kkt_matrix = np.full((len(all_results), max_iters), np.nan)
        obj_matrix = np.full((len(all_results), max_iters), np.nan)
        
        for trial, result in enumerate(all_results):
            kkt_traj = result[method]['kkt']
            obj_traj = result[method]['objective']
            kkt_matrix[trial, :len(kkt_traj)] = kkt_traj
            obj_matrix[trial, :len(obj_traj)] = obj_traj
        
        # Fill NaN with last value
        for trial in range(len(all_results)):
            mask = ~np.isnan(kkt_matrix[trial, :])
            if np.any(mask):
                last_kkt = kkt_matrix[trial, mask][-1]
                last_obj = obj_matrix[trial, mask][-1]
                kkt_matrix[trial, ~mask] = last_kkt
                obj_matrix[trial, ~mask] = last_obj
        
        iter_stats[method] = {
            'kkt_mean': np.nanmean(kkt_matrix, axis=0),
            'kkt_std': np.nanstd(kkt_matrix, axis=0),
            'obj_mean': np.nanmean(obj_matrix, axis=0),
            'obj_std': np.nanstd(obj_matrix, axis=0)
        }
        
        # Time-based statistics - use actual trajectory data
        time_stats[method] = {
            'times': [],
            'objectives': []
        }
        
        for trial, result in enumerate(all_results):
            times = result[method]['times']
            objectives = result[method]['objective']
            time_stats[method]['times'].extend(times)
            time_stats[method]['objectives'].extend(objectives)
    
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
    plt.title('Orthant (NNLS): KKT residual vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    # Save for paper
    import os
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/pos_err_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 2: Objective gap vs iterations
    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        mean_obj = iter_stats[method]['obj_mean']
        std_obj = iter_stats[method]['obj_std']
        
        plt.semilogy(iter_grid, np.abs(mean_obj - f_star + 1e-16), 
                    color=colors[i], linestyle=linestyles[i], 
                    linewidth=3, label=method_names[i])
        plt.fill_between(iter_grid, 
                        np.abs(mean_obj - std_obj - f_star + 1e-16),
                        np.abs(mean_obj + std_obj - f_star + 1e-16),
                        color=colors[i], alpha=0.2)
    
    plt.xlabel('Iterations', fontweight='bold')
    plt.ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    plt.title('Orthant (NNLS): objective gap vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("figs/pos_f_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 3: Objective gap vs time
    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        # Plot individual trajectories from all trials
        for trial, result in enumerate(all_results):
            times = result[method]['times']
            objectives = result[method]['objective']
            
            if trial == 0:  # Only label the first trial
                plt.loglog(times, np.abs(np.array(objectives) - f_star + 1e-16), 
                          color=colors[i], linestyle=linestyles[i], 
                          linewidth=2, alpha=0.7, label=method_names[i])
            else:
                plt.loglog(times, np.abs(np.array(objectives) - f_star + 1e-16), 
                          color=colors[i], linestyle=linestyles[i], 
                          linewidth=1, alpha=0.3)
    
    plt.xlabel('Time [seconds]', fontweight='bold')
    plt.ylabel(r'$|f(x_k) - f^*|$', fontweight='bold')
    plt.title('Orthant (NNLS): objective gap vs time', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("figs/pos_f_vs_time.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Paper-grade benchmark
    all_results, stats, problem, x_star, f_star = run_multi_seed_benchmark(
        n=120, m=120, density=0.15, eta=400.0, max_iters=400, 
        n_trials=10, tol_kkt=1e-6
    )
    
    # Create paper plots
    plot_paper_results(all_results, stats, problem, x_star, f_star)
