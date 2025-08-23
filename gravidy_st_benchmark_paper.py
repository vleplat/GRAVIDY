#!/usr/bin/env python3
"""
GRAVIDY–St Stiefel Benchmark for Paper
Paper-grade benchmarking with multi-seed averaging, gradient norm tracking, and feasibility metrics.
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt

# Import utilities and solvers
from utils.stiefel_utils import rand_stiefel
from utils.objective import StiefelQuad
from solver.gravidy_st_fast import ICS_gravidy_fast
from solver.gravidy_st_nr_dense import ICS_gravidy_NR_dense
from solver.wy_cayley import WY_cayley
from solver.rgd_qr import RGD_QR


def make_problem(n=200, p=2, cond=50.0, seed=0):
    """Create a test problem with per-column SPD quadratic objectives."""
    rng = np.random.default_rng(seed)
    Q_list = []
    for _ in range(p):
        U, _ = np.linalg.qr(rng.standard_normal((n, n)))
        vals = np.logspace(0, math.log10(cond), n)
        Q = U @ np.diag(vals) @ U.T
        Q_list.append(Q)
    prob = StiefelQuad(Q_list)
    prob.Q_list = Q_list
    return prob


def run_single_trial(prob, n, p, seed=0, max_outer=100, max_iters=50000, tol_grad=1e-8):
    """Run single trial for all methods."""
    np.random.seed(seed)
    
    X0 = rand_stiefel(n, p, seed=seed+123)
    
    # Improved starting point scaling based on problem structure
    A0 = prob.A_skew(X0)
    g0 = prob.grad_riem(X0)
    A0_norm = np.linalg.norm(A0, 'fro')
    g0_norm = np.linalg.norm(g0, 'fro')
    
    # Better scaling: use both A0 and gradient norm, with problem size scaling
    scale_A = 1.0 / max(1e-12, A0_norm)
    scale_g = 1.0 / max(1e-12, g0_norm)
    
    # Newton methods: scale by problem size and A0 norm
    alpha0_fast = 0.5 * scale_A * p
    alpha0_dense = 0.5 * scale_A * p
    
    # Explicit methods: scale by gradient norm
    alpha0_wy = 0.5 * scale_g * p
    alpha0_rgd = 1.0 * scale_g * p
    
    results = {}
    
    # GRAVIDY–St (Fast)
    start_time = time.perf_counter()
    X_fast, H_fast = ICS_gravidy_fast(prob, X0, alpha0=alpha0_fast, max_outer=max_outer, 
                                     tol_grad=tol_grad, verbose=False)
    time_fast = time.perf_counter() - start_time
    
    # Extract trajectories with gradient norms and feasibility
    grad_norms_fast = []
    feasibility_fast = []
    objectives_fast = []
    times_fast = []
    
    for h in H_fast:
        it, obj, grad_norm, feas = h[0], h[1], h[2], h[3]
        grad_norms_fast.append(grad_norm)
        feasibility_fast.append(feas)
        objectives_fast.append(obj)
        if len(h) > 4:  # Fast solver has time in position 4
            times_fast.append(h[4])
        else:
            times_fast.append(it * time_fast / len(H_fast))  # Approximate timing
    
    results['gravidy_fast'] = {
        'grad_norm': grad_norms_fast,
        'feasibility': feasibility_fast,
        'objective': objectives_fast,
        'times': times_fast,
        'final_obj': objectives_fast[-1],
        'final_feas': feasibility_fast[-1],
        'final_grad': grad_norms_fast[-1],
        'converged': grad_norms_fast[-1] <= tol_grad
    }
    

    
    # Wen-Yin Cayley
    start_time = time.perf_counter()
    X_wy, H_wy = WY_cayley(prob, X0, alpha0=alpha0_wy, max_iters=max_iters, verbose=False)
    time_wy = time.perf_counter() - start_time
    
    # H_wy format: {"it": it, "f": f, "time": time, "feas": feas, "grad_norm": grad_norm}
    grad_norms_wy = H_wy["grad_norm"]
    
    results['wy_cayley'] = {
        'grad_norm': grad_norms_wy,
        'feasibility': H_wy["feas"],
        'objective': H_wy["f"],
        'times': H_wy["time"],
        'final_obj': H_wy["f"][-1],
        'final_feas': H_wy["feas"][-1],
        'final_grad': grad_norms_wy[-1],
        'converged': grad_norms_wy[-1] <= tol_grad
    }
    
    # RGD-QR
    start_time = time.perf_counter()
    X_rgd, H_rgd = RGD_QR(prob, X0, alpha0=alpha0_rgd, max_iters=max_iters, verbose=False)
    time_rgd = time.perf_counter() - start_time
    
    # H_rgd format: {"it": it, "f": f, "time": time, "feas": feas, "grad_norm": grad_norm}
    grad_norms_rgd = H_rgd["grad_norm"]
    
    results['rgd_qr'] = {
        'grad_norm': grad_norms_rgd,
        'feasibility': H_rgd["feas"],
        'objective': H_rgd["f"], 
        'times': H_rgd["time"],
        'final_obj': H_rgd["f"][-1],
        'final_feas': H_rgd["feas"][-1],
        'final_grad': grad_norms_rgd[-1],
        'converged': grad_norms_rgd[-1] <= tol_grad
    }
    
    return results


def run_multi_seed_benchmark(n=200, p=2, cond=50.0, max_outer=1000, max_iters=50000, 
                            n_trials=10, tol_grad=1e-6):
    """Run benchmark with multiple seeds and compute statistics."""
    print(f"\n=== GRAVIDY–St Stiefel Paper Benchmark ===")
    print(f"Problem: n={n}, p={p}, condition={cond}")
    print(f"Trials: {n_trials}, max_outer={max_outer}, max_iters={max_iters}, grad_tol={tol_grad:.0e}")
    print(f"Objective: Stiefel quadratic with per-column SPD matrices")
    print("-" * 70)
    
    # Generate problem (same for all trials)
    prob = make_problem(n=n, p=p, cond=cond, seed=42)
    
    print(f"Matrix condition number: {cond}")
    print(f"Stiefel manifold: St({n},{p})")
    print("-" * 70)
    
    # Run trials
    all_results = []
    for trial in range(n_trials):
        if trial % 2 == 0:
            print(f"Running trial {trial+1}/{n_trials}...")
        results = run_single_trial(prob, n, p, seed=trial, max_outer=max_outer, 
                                 max_iters=max_iters, tol_grad=tol_grad)
        all_results.append(results)
    
    # Compute statistics
    methods = ['gravidy_fast', 'wy_cayley', 'rgd_qr']
    method_names = ['GRAVIDY–St (Fast)', 'Wen–Yin Cayley', 'RGD–QR']
    
    stats = {}
    for i, method in enumerate(methods):
        # Collect final metrics
        final_obj = [r[method]['final_obj'] for r in all_results]
        final_feas = [r[method]['final_feas'] for r in all_results]
        final_grad = [r[method]['final_grad'] for r in all_results]
        runtimes = [r[method]['times'][-1] for r in all_results]  # Final runtime
        
        stats[method] = {
            'name': method_names[i],
            'final_obj_mean': np.mean(final_obj),
            'final_obj_std': np.std(final_obj),
            'final_feas_mean': np.mean(final_feas),
            'final_feas_std': np.std(final_feas),
            'final_grad_mean': np.mean(final_grad),
            'final_grad_std': np.std(final_grad),
            'runtime_mean': np.mean(runtimes),
            'runtime_std': np.std(runtimes)
        }
    
    # Print summary
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY (averaged over {} trials)".format(n_trials))
    print("="*90)
    print(f"{'Method':<25} {'Final Obj':<15} {'Final Feas':<15} {'Final Grad':<15} {'Runtime [s]':<15}")
    print("-" * 105)
    for method in methods:
        s = stats[method]
        print(f"{s['name']:<25} {s['final_obj_mean']:<7.3e}±{s['final_obj_std']:<6.2e} "
              f"{s['final_feas_mean']:<7.3e}±{s['final_feas_std']:<6.2e} "
              f"{s['final_grad_mean']:<7.3e}±{s['final_grad_std']:<6.2e} {s['runtime_mean']:<7.3f}±{s['runtime_std']:<6.2f}")
    print("="*105)
    
    return all_results, stats, prob


def plot_paper_results(all_results, stats, prob):
    """Create paper-quality plots with error bars."""
    methods = ['gravidy_fast', 'wy_cayley', 'rgd_qr'] 
    colors = ['red', 'blue', 'green']
    linestyles = ['-', '--', ':']
    method_names = ['GRAVIDY–St (Fast)', 'Wen–Yin Cayley', 'RGD–QR']
    
    # Determine common iteration grid
    max_iters = max(len(all_results[0][method]['grad_norm']) for method in methods)
    iter_grid = np.arange(max_iters)
    
    # Interpolate trajectories to common grids and compute statistics
    iter_stats = {}
    time_stats = {}
    
    for method in methods:
        # Iteration-based statistics
        grad_matrix = np.full((len(all_results), max_iters), np.nan)
        feas_matrix = np.full((len(all_results), max_iters), np.nan)
        
        for trial, result in enumerate(all_results):
            grad_traj = result[method]['grad_norm']
            feas_traj = result[method]['feasibility']
            grad_matrix[trial, :len(grad_traj)] = grad_traj
            feas_matrix[trial, :len(feas_traj)] = feas_traj
        
        # Fill NaN with last value
        for trial in range(len(all_results)):
            grad_mask = ~np.isnan(grad_matrix[trial, :])
            feas_mask = ~np.isnan(feas_matrix[trial, :])
            if np.any(grad_mask):
                last_grad = grad_matrix[trial, grad_mask][-1]
                last_feas = feas_matrix[trial, feas_mask][-1]
                grad_matrix[trial, ~grad_mask] = last_grad
                feas_matrix[trial, ~feas_mask] = last_feas
        
        iter_stats[method] = {
            'grad_mean': np.nanmean(grad_matrix, axis=0),
            'grad_std': np.nanstd(grad_matrix, axis=0),
            'feas_mean': np.nanmean(feas_matrix, axis=0),
            'feas_std': np.nanstd(feas_matrix, axis=0)
        }
        
        # Time-based statistics - use actual trajectory data
        time_stats[method] = {
            'times': [],
            'feasibility': []
        }
        
        for trial, result in enumerate(all_results):
            times = result[method]['times']
            feasibility = result[method]['feasibility']
            time_stats[method]['times'].extend(times)
            time_stats[method]['feasibility'].extend(feasibility)
    
    # Create plots
    plt.style.use('default')
    
    # Figure 1: Gradient norm vs iterations (required for paper)
    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        mean_grad = iter_stats[method]['grad_mean']
        std_grad = iter_stats[method]['grad_std']
        
        plt.semilogy(iter_grid, mean_grad, color=colors[i], linestyle=linestyles[i], 
                    linewidth=3, label=method_names[i])
        plt.fill_between(iter_grid, mean_grad - std_grad, mean_grad + std_grad, 
                        color=colors[i], alpha=0.2)
    
    plt.xlabel('Iterations', fontweight='bold')
    plt.ylabel(r'$\|\nabla\Phi(X_k)\|_F$', fontweight='bold')
    plt.title('Stiefel quadratic: gradient norm vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    # Save for paper
    import os
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/stiefel_grad_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 2: Feasibility vs time (required for paper)
    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        # Plot individual trajectories from all trials
        for trial, result in enumerate(all_results):
            times = result[method]['times']
            feasibility = result[method]['feasibility']
            
            if trial == 0:  # Only label the first trial
                plt.semilogy(times, feasibility, 
                          color=colors[i], linestyle=linestyles[i], 
                          linewidth=2, alpha=0.7, label=method_names[i])
            else:
                plt.semilogy(times, feasibility, 
                          color=colors[i], linestyle=linestyles[i], 
                          linewidth=1, alpha=0.3)
    
    plt.xlabel('Time [seconds]', fontweight='bold')
    plt.ylabel(r'$\|X_k^\top X_k - I\|_F$', fontweight='bold')
    plt.title('Stiefel quadratic: feasibility vs time', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("figs/stiefel_feas_vs_time.pdf", bbox_inches="tight")
    plt.show()
    
    # Additional figures for completeness
    
    # Figure 3: Feasibility vs iterations
    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        mean_feas = iter_stats[method]['feas_mean']
        std_feas = iter_stats[method]['feas_std']
        
        plt.semilogy(iter_grid, mean_feas, color=colors[i], linestyle=linestyles[i], 
                    linewidth=3, label=method_names[i])
        plt.fill_between(iter_grid, mean_feas - std_feas, mean_feas + std_feas, 
                        color=colors[i], alpha=0.2)
    
    plt.xlabel('Iterations', fontweight='bold')
    plt.ylabel(r'$\|X_k^\top X_k - I\|_F$', fontweight='bold')
    plt.title('Stiefel quadratic: feasibility vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("figs/stiefel_err_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 4: Final objective values (bar chart across all trials)
    plt.figure(figsize=(8, 6))
    
    # Compute final objective values for each method across all trials
    final_obj_values = []
    final_obj_std = []
    
    for method in methods:
        obj_finals = []
        for trial, result in enumerate(all_results):
            obj_traj = result[method]['objective']
            if len(obj_traj) > 0:
                obj_finals.append(obj_traj[-1])  # Final objective value
        
        final_obj_values.append(np.mean(obj_finals))
        final_obj_std.append(np.std(obj_finals))
    
    # Create bar chart
    bars = plt.bar(method_names, final_obj_values, yerr=final_obj_std, 
                   color=colors, alpha=0.8, capsize=5)
    plt.ylabel(r'Final Objective $\Phi(X_k)$', fontweight='bold')
    plt.title('Stiefel quadratic: Final Objective Values (10 trials)', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value, std_val in zip(bars, final_obj_values, final_obj_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}\n±{std_val:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("figs/stiefel_final_obj.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Paper-grade benchmark
    all_results, stats, prob = run_multi_seed_benchmark(
        n=100, p=2, cond=1000.0, max_outer=500, max_iters=50000, 
        n_trials=10, tol_grad=1e-5
    )
    
    # Create paper plots
    plot_paper_results(all_results, stats, prob)
