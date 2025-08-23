#!/usr/bin/env python3
"""
GRAVIDY–Δ Simplex Benchmark for Paper
Paper-grade benchmarking with multi-seed averaging and proper metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from utils.simplex_objective import create_test_problem
from solver.gravidy_delta_klprox import GRAVIDY_Delta_KLprox
from solver.gravidy_delta_mgn import GRAVIDY_Delta_MGN
from solver.pgd_simplex import PGD_simplex
from solver.apgd_simplex import APGD_simplex
from solver.emd_simplex import EMD_simplex


def compute_kkt_residual(x, problem):
    """Compute KKT residual: ||x - Π_C(x - ∇f(x))||_2"""
    from utils.simplex_utils import project_simplex
    grad = problem.grad(x)
    projected = project_simplex(x - grad)
    return np.linalg.norm(x - projected)


def run_single_trial(problem, x_star, eta=50.0, max_iters=400, seed=0, tol_kkt=1e-6):
    """Run single trial for all methods."""
    np.random.seed(seed)
    
    results = {}
    
    # GRAVIDY–Δ (KL-prox)
    start_time = time.perf_counter()
    x_kl, hist_kl = GRAVIDY_Delta_KLprox(problem, eta=eta, max_outer=max_iters, 
                                        tol_grad=tol_kkt, verbose=False)
    time_kl = time.perf_counter() - start_time
    
    kkt_kl = [entry[2] for entry in hist_kl]  # grad_norm approximates KKT
    f_kl = [entry[1] for entry in hist_kl]    # objective values
    times_kl = [entry[3] for entry in hist_kl]  # times
    
    # Compute actual KKT residual for final point
    grad_final = problem.grad(x_kl)
    from utils.simplex_utils import project_simplex
    projected_final = project_simplex(x_kl - grad_final)
    kkt_final = np.linalg.norm(x_kl - projected_final)
    
    results['kl_prox'] = {
        'kkt': kkt_kl,
        'objective': f_kl,
        'times': times_kl,
        'final_error': np.linalg.norm(x_kl - x_star),
        'final_kkt': kkt_final,
        'converged': kkt_final <= tol_kkt
    }
    
    # GRAVIDY–Δ (MGN)
    start_time = time.perf_counter()
    x_mgn, hist_mgn = GRAVIDY_Delta_MGN(problem, eta=eta, max_outer=max_iters, 
                                       tol_grad=tol_kkt, verbose=False)
    time_mgn = time.perf_counter() - start_time
    
    kkt_mgn = [entry[2] for entry in hist_mgn]
    f_mgn = [entry[1] for entry in hist_mgn]
    times_mgn = [entry[3] for entry in hist_mgn]
    
    grad_final = problem.grad(x_mgn)
    projected_final = project_simplex(x_mgn - grad_final)
    kkt_final = np.linalg.norm(x_mgn - projected_final)
    
    results['mgn'] = {
        'kkt': kkt_mgn,
        'objective': f_mgn,
        'times': times_mgn,
        'final_error': np.linalg.norm(x_mgn - x_star),
        'final_kkt': kkt_final,
        'converged': kkt_final <= tol_kkt
    }
    
    # PGD (baseline)
    start_time = time.perf_counter()
    x_pgd, hist_pgd = PGD_simplex(problem, max_iters=max_iters, 
                                 tol_grad=tol_kkt, verbose=False)
    time_pgd = time.perf_counter() - start_time
    
    kkt_pgd = [entry[2] for entry in hist_pgd]
    f_pgd = [entry[1] for entry in hist_pgd]
    times_pgd = [entry[3] for entry in hist_pgd]
    
    grad_final = problem.grad(x_pgd)
    projected_final = project_simplex(x_pgd - grad_final)
    kkt_final = np.linalg.norm(x_pgd - projected_final)
    
    results['pgd'] = {
        'kkt': kkt_pgd,
        'objective': f_pgd,
        'times': times_pgd,
        'final_error': np.linalg.norm(x_pgd - x_star),
        'final_kkt': kkt_final,
        'converged': kkt_final <= tol_kkt
    }
    
    # APGD (Nesterov)
    start_time = time.perf_counter()
    x_apgd, hist_apgd = APGD_simplex(problem, max_iters=max_iters, 
                                    tol_grad=tol_kkt, verbose=False)
    time_apgd = time.perf_counter() - start_time
    
    kkt_apgd = [entry[2] for entry in hist_apgd]
    f_apgd = [entry[1] for entry in hist_apgd]
    times_apgd = [entry[3] for entry in hist_apgd]
    
    grad_final = problem.grad(x_apgd)
    projected_final = project_simplex(x_apgd - grad_final)
    kkt_final = np.linalg.norm(x_apgd - projected_final)
    
    results['apgd'] = {
        'kkt': kkt_apgd,
        'objective': f_apgd,
        'times': times_apgd,
        'final_error': np.linalg.norm(x_apgd - x_star),
        'final_kkt': kkt_final,
        'converged': kkt_final <= tol_kkt
    }
    
    # EMD (baseline)
    start_time = time.perf_counter()
    x_emd, hist_emd = EMD_simplex(problem, max_iters=max_iters, 
                                 tol_grad=tol_kkt, verbose=False)
    time_emd = time.perf_counter() - start_time
    
    kkt_emd = [entry[2] for entry in hist_emd]
    f_emd = [entry[1] for entry in hist_emd]
    times_emd = [entry[3] for entry in hist_emd]
    
    grad_final = problem.grad(x_emd)
    projected_final = project_simplex(x_emd - grad_final)
    kkt_final = np.linalg.norm(x_emd - projected_final)
    
    results['emd'] = {
        'kkt': kkt_emd,
        'objective': f_emd,
        'times': times_emd,
        'final_error': np.linalg.norm(x_emd - x_star),
        'final_kkt': kkt_final,
        'converged': kkt_final <= tol_kkt
    }
    
    return results


def run_multi_seed_benchmark(n=40, m=40, cond=5.0, eta=50.0, max_iters=400, 
                            n_trials=10, tol_kkt=1e-6):
    """Run benchmark with multiple seeds and compute statistics."""
    print(f"\n=== GRAVIDY–Δ Simplex Paper Benchmark ===")
    print(f"Problem: n={n}, m={m}, condition={cond:.1f}, eta={eta}")
    print(f"Trials: {n_trials}, max_iters={max_iters}, KKT_tol={tol_kkt:.0e}")
    print(f"Objective: min_{{x in simplex}} 0.5 ||Ax - b||^2")
    print("-" * 70)
    
    # Generate problem (same for all trials)
    A, b, x_star, problem = create_test_problem(n, m, cond=cond, seed=42)
    f_star = problem.f(x_star)
    
    print(f"Ground truth objective f* = {f_star:.6e}")
    print(f"||x*||_1 = {np.sum(x_star):.6f} (should be 1.0)")
    print(f"Matrix condition number: {cond:.1f}")
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
    methods = ['kl_prox', 'mgn', 'pgd', 'apgd', 'emd']
    method_names = ['GRAVIDY–Δ (KL-prox)', 'GRAVIDY–Δ (MGN)', 'PGD (baseline)', 
                   'APGD (Nesterov)', 'EMD (baseline)']
    
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
    methods = ['kl_prox', 'mgn', 'pgd', 'apgd', 'emd'] 
    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    linestyles = ['-', '--', ':', '-.', '-']
    method_names = ['GRAVIDY–Δ (KL-prox)', 'GRAVIDY–Δ (MGN)', 'PGD (baseline)', 
                   'APGD (Nesterov)', 'EMD (baseline)']
    
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
    plt.title('Simplex: KKT residual vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    # Save for paper
    import os
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/simplex_err_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 2: Objective gap vs time
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
    plt.title('Simplex: objective gap vs time', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("figs/simplex_f_vs_time.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 3: Objective gap vs iterations
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
    plt.title('Simplex: objective gap vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("figs/simplex_f_vs_it.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Paper-grade benchmark
    all_results, stats, problem, x_star, f_star = run_multi_seed_benchmark(
        n=40, m=40, cond=5.0, eta=50.0, max_iters=400, 
        n_trials=10, tol_kkt=1e-6
    )
    
    # Create paper plots
    plot_paper_results(all_results, stats, problem, x_star, f_star)
