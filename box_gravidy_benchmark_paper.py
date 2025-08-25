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
    
    # GRAVIDY–box (Newton)
    start_time = time.perf_counter()
    x_grav_newton, hist_grav_newton = GRAVIDY_box(problem, eta=eta, max_outer=max_iters, 
                                                 tol_grad=tol_kkt, inner='newton', verbose=False)
    time_grav_newton = time.perf_counter() - start_time
    
    kkt_grav_newton = [entry[2] for entry in hist_grav_newton]  # grad_norm approximates KKT
    f_grav_newton = [entry[1] for entry in hist_grav_newton]    # objective values
    times_grav_newton = [entry[3] for entry in hist_grav_newton]  # times
    
    # Compute actual KKT residual for final point
    grad_final = problem.grad(x_grav_newton)
    projected_final = problem.project(x_grav_newton - grad_final)
    kkt_final = np.linalg.norm(x_grav_newton - projected_final)
    
    results['gravidy_newton'] = {
        'kkt': kkt_grav_newton,
        'objective': f_grav_newton,
        'times': times_grav_newton,
        'final_error': np.linalg.norm(x_grav_newton - x_star),
        'final_kkt': kkt_final,
        'converged': kkt_final <= tol_kkt
    }
    
    # GRAVIDY–box (MGN)
    start_time = time.perf_counter()
    x_grav_mgn, hist_grav_mgn = GRAVIDY_box(problem, eta=eta, max_outer=max_iters, 
                                           tol_grad=tol_kkt, inner='mgn', verbose=False)
    time_grav_mgn = time.perf_counter() - start_time
    
    kkt_grav_mgn = [entry[2] for entry in hist_grav_mgn]  # grad_norm approximates KKT
    f_grav_mgn = [entry[1] for entry in hist_grav_mgn]    # objective values
    times_grav_mgn = [entry[3] for entry in hist_grav_mgn]  # times
    
    # Compute actual KKT residual for final point
    grad_final = problem.grad(x_grav_mgn)
    projected_final = problem.project(x_grav_mgn - grad_final)
    kkt_final = np.linalg.norm(x_grav_mgn - projected_final)
    
    results['gravidy_mgn'] = {
        'kkt': kkt_grav_mgn,
        'objective': f_grav_mgn,
        'times': times_grav_mgn,
        'final_error': np.linalg.norm(x_grav_mgn - x_star),
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
    A, b, lo, hi, x_star, problem = create_box_test_problem(n, m, seed=100)
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
    methods = ['gravidy_newton', 'gravidy_mgn', 'apgd']
    method_names = ['GRAVIDY–box (Newton)', 'GRAVIDY–box (MGN)', 'APGD-box (Nesterov)']
    
    stats = {}
    for i, method in enumerate(methods):
        # Collect final metrics
        final_errors = [r[method]['final_error'] for r in all_results]
        final_kkt = [r[method]['final_kkt'] for r in all_results]
        final_objectives = [r[method]['objective'][-1] for r in all_results]  # Final objective value
        runtimes = [r[method]['times'][-1] for r in all_results]  # Final runtime
        
        stats[method] = {
            'name': method_names[i],
            'final_error_mean': np.mean(final_errors),
            'final_error_std': np.std(final_errors),
            'final_kkt_mean': np.mean(final_kkt),
            'final_kkt_std': np.std(final_kkt),
            'final_obj_mean': np.mean(final_objectives),
            'final_obj_std': np.std(final_objectives),
            'runtime_mean': np.mean(runtimes),
            'runtime_std': np.std(runtimes)
        }
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY (averaged over {} trials)".format(n_trials))
    print("="*80)
    print(f"{'Method':<25} {'||x-x*||_2':<15} {'KKT residual':<15} {'Final obj':<15} {'Runtime [s]':<15}")
    print("-" * 95)
    for method in methods:
        s = stats[method]
        print(f"{s['name']:<25} {s['final_error_mean']:<7.3e}±{s['final_error_std']:<6.2e} "
              f"{s['final_kkt_mean']:<7.3e}±{s['final_kkt_std']:<6.2e} {s['final_obj_mean']:<7.3e}±{s['final_obj_std']:<6.2e} "
              f"{s['runtime_mean']:<7.3f}±{s['runtime_std']:<6.2f}")
    print("="*95)
    
    return all_results, stats, problem, x_star, f_star


def plot_paper_results(all_results, stats, problem, x_star, f_star):
    """Create paper-quality plots with error bars."""
    methods = ['gravidy_newton', 'gravidy_mgn', 'apgd'] 
    colors = ['red', 'green', 'blue']
    linestyles = ['-', '-', '--']
    method_names = ['GRAVIDY–box (Newton)', 'GRAVIDY–box (MGN)', 'APGD-box (Nesterov)']
    
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
    plt.title('Box: KKT residual vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    # Save for paper
    import os
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/box_err_vs_it.pdf", bbox_inches="tight")
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
    plt.title('Box: objective gap vs time', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("figs/box_f_vs_time.pdf", bbox_inches="tight")
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
    plt.title('Box: objective gap vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("figs/box_f_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 4: Final KKT residuals (bar chart across all trials)
    plt.figure(figsize=(8, 6))
    
    # Compute final KKT residuals for each method across all trials
    final_kkt_values = []
    final_kkt_std = []
    
    for method in methods:
        kkt_finals = []
        for trial, result in enumerate(all_results):
            kkt_traj = result[method]['kkt']
            if len(kkt_traj) > 0:
                kkt_finals.append(kkt_traj[-1])  # Final KKT value
        
        final_kkt_values.append(np.mean(kkt_finals))
        final_kkt_std.append(np.std(kkt_finals))
    
    # Create bar chart
    bars = plt.bar(method_names, final_kkt_values, yerr=final_kkt_std, 
                   color=colors, alpha=0.8, capsize=5)
    plt.ylabel(r'Final KKT residual $\|x - \Pi_C(x - \nabla f(x))\|_2$', fontweight='bold')
    plt.title('Box: Final KKT Residuals (10 trials)', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value, std_val in zip(bars, final_kkt_values, final_kkt_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2e}\n±{std_val:.2e}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("figs/box_final_kkt.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Paper-grade benchmark
    all_results, stats, problem, x_star, f_star = run_multi_seed_benchmark(
        n=120, m=120, eta=150.0, max_iters=400, 
        n_trials=10, tol_kkt=1e-6
    )
    
    # Create paper plots
    plot_paper_results(all_results, stats, problem, x_star, f_star)
