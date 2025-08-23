import time
import math
import numpy as np
import matplotlib.pyplot as plt

# Import utilities and solvers
from utils.stiefel_utils import rand_stiefel
from utils.objective import StiefelQuad
from solver.gravidy_st_nr_dense import ICS_NR_dense_fast
from solver.gravidy_st_nk import ICS_gravidy_NK
from solver.gravidy_st_fast import ICS_gravidy_fast
from solver.wy_cayley import WY_cayley
from solver.rgd_qr import RGD_QR

# ======= Benchmark harness =======

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
    # Store Q_list for smart initialization
    prob.Q_list = Q_list
    return prob

def run_benchmark(n=300, p=5, cond=50.0, seed=0,
                  alpha0_nr=0.5, alpha0_nk=0.5, alpha0_fast=0.5, alpha0_wy=0.5, alpha0_rgd=1.0,
                  tmax=5.0, verbose=False):
    """Run benchmark comparing all five solvers."""
    prob = make_problem(n=n, p=p, cond=cond, seed=seed)
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
    alpha0_nr = alpha0_nr * scale_A * p  # Use p instead of n for better scaling
    alpha0_nk = alpha0_nk * scale_A * p
    alpha0_fast = alpha0_fast * scale_A * p  # Fast ICS can handle larger steps
    
    # Explicit methods: scale by gradient norm
    alpha0_wy = alpha0_wy * scale_g * p
    alpha0_rgd = alpha0_rgd * scale_g * p

    # Run solvers
    print("Running Fast ICS...")
    X_fast, H_fast = ICS_gravidy_fast(prob, X0, alpha0=alpha0_fast, max_outer=100, tol_grad=1e-6, verbose=verbose)
    print("Running NR-Dense (Optimized)...")
    X_dense, H_dense = ICS_NR_dense_fast(prob, X0, alpha0=alpha0_fast, max_outer=100, tol_grad=1e-6, verbose=verbose)
    print("Running Wen-Yin Cayley...")
    X_wy, H_wy = WY_cayley(prob, X0, alpha0=alpha0_wy, max_iters=50000, verbose=verbose)
    print("Running RGD-QR...")
    X_rgd, H_rgd = RGD_QR(prob, X0, alpha0=alpha0_rgd, max_iters=50000, verbose=verbose)

    # Convert history format for Newton methods
    def convert_history(hist):
        it = [h[0] for h in hist]
        f = [h[1] for h in hist]
        if len(hist[0]) > 4:  # Fast solver has time in position 4
            time = [h[4] for h in hist]
        else:  # NR-Dense and NK-GMRES don't have time, use iteration count
            time = list(range(len(hist)))
        feas = [h[3] for h in hist]
        return {"it": it, "f": f, "time": time, "feas": feas}

    H_fast = convert_history(H_fast)
    H_dense = convert_history(H_dense)

    # Set up improved plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'lines.linewidth': 3,
        'lines.markersize': 8,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'figure.dpi': 100
    })
    
    # Define distinct colors and markers for better visibility
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    # Plot 1: Objective vs iterations (large, clear)
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.loglog(H_fast["it"], H_fast["f"], 
               color=colors[0], marker=markers[0], markevery=max(1, len(H_fast["it"])//20),
               linestyle=line_styles[0], linewidth=3, markersize=6,
               label="GRAVIDY–St (Fast)")
    plt.loglog(H_dense["it"], H_dense["f"], 
               color=colors[1], marker=markers[1], markevery=max(1, len(H_dense["it"])//20),
               linestyle=line_styles[1], linewidth=3, markersize=6,
               label="GRAVIDY–St (NR-Dense Fast)")
    plt.loglog(H_wy["it"], H_wy["f"], 
               color=colors[2], marker=markers[2], markevery=max(1, len(H_wy["it"])//20),
               linestyle=line_styles[2], linewidth=3, markersize=6,
               label="Wen–Yin Cayley")
    plt.loglog(H_rgd["it"], H_rgd["f"], 
               color=colors[3], marker=markers[3], markevery=max(1, len(H_rgd["it"])//20),
               linestyle=line_styles[3], linewidth=3, markersize=6,
               label="RGD–QR")
    
    plt.xlabel("Iterations", fontsize=14, fontweight='bold')
    plt.ylabel("Objective Value", fontsize=14, fontweight='bold')
    plt.title("Objective Convergence vs Iterations (loglog)", fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
        # Plot 2: Objective vs time (loglog)
    plt.subplot(2, 2, 2)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.loglog(H_fast["time"], H_fast["f"], 
               color=colors[0], marker=markers[0], markevery=max(1, len(H_fast["time"])//20),
               linestyle=line_styles[0], linewidth=3, markersize=6,
               label="GRAVIDY–St (Fast)")
    plt.loglog(H_dense["time"], H_dense["f"], 
               color=colors[1], marker=markers[1], markevery=max(1, len(H_dense["time"])//20),
               linestyle=line_styles[1], linewidth=3, markersize=6,
               label="GRAVIDY–St (NR-Dense Fast)")
    plt.loglog(H_wy["time"], H_wy["f"], 
               color=colors[2], marker=markers[2], markevery=max(1, len(H_wy["time"])//20),
               linestyle=line_styles[2], linewidth=3, markersize=6,
               label="Wen–Yin Cayley")
    plt.loglog(H_rgd["time"], H_rgd["f"], 
               color=colors[3], marker=markers[3], markevery=max(1, len(H_rgd["time"])//20),
               linestyle=line_styles[3], linewidth=3, markersize=6,
               label="RGD–QR")
    
    plt.xlabel("Time [seconds]", fontsize=14, fontweight='bold')
    plt.ylabel("Objective Value", fontsize=14, fontweight='bold')
    plt.title("Objective Convergence vs Time (loglog)", fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 3: Feasibility vs iterations
    plt.subplot(2, 2, 3)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.semilogy(H_fast["it"], H_fast["feas"], 
               color=colors[0], marker=markers[0], markevery=max(1, len(H_fast["it"])//20),
               linestyle=line_styles[0], linewidth=3, markersize=6,
               label="GRAVIDY–St (Fast)")
    plt.semilogy(H_dense["it"], H_dense["feas"], 
               color=colors[1], marker=markers[1], markevery=max(1, len(H_dense["it"])//20),
               linestyle=line_styles[1], linewidth=3, markersize=6,
               label="GRAVIDY–St (NR-Dense Fast)")
    plt.semilogy(H_wy["it"], H_wy["feas"], 
               color=colors[2], marker=markers[2], markevery=max(1, len(H_wy["it"])//20),
               linestyle=line_styles[2], linewidth=3, markersize=6,
               label="Wen–Yin Cayley")
    plt.semilogy(H_rgd["it"], H_rgd["feas"], 
               color=colors[3], marker=markers[3], markevery=max(1, len(H_rgd["it"])//20),
               linestyle=line_styles[3], linewidth=3, markersize=6,
               label="RGD–QR")
    
    plt.xlabel("Iterations", fontsize=14, fontweight='bold')
    plt.ylabel(r"Feasibility: $\|X^\top X - I\|_F$", fontsize=14, fontweight='bold')
    plt.title("Feasibility vs Iterations", fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 4: Feasibility vs time
    plt.subplot(2, 2, 4)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.semilogy(H_fast["time"], H_fast["feas"], 
               color=colors[0], marker=markers[0], markevery=max(1, len(H_fast["time"])//20),
               linestyle=line_styles[0], linewidth=3, markersize=6,
               label="GRAVIDY–St (Fast)")
    plt.semilogy(H_dense["time"], H_dense["feas"], 
               color=colors[1], marker=markers[1], markevery=max(1, len(H_dense["time"])//20),
               linestyle=line_styles[1], linewidth=3, markersize=6,
               label="GRAVIDY–St (NR-Dense Fast)")
    plt.semilogy(H_wy["time"], H_wy["feas"], 
               color=colors[2], marker=markers[2], markevery=max(1, len(H_wy["time"])//20),
               linestyle=line_styles[2], linewidth=3, markersize=6,
               label="Wen–Yin Cayley")
    plt.semilogy(H_rgd["time"], H_rgd["feas"], 
               color=colors[3], marker=markers[3], markevery=max(1, len(H_rgd["time"])//20),
               linestyle=line_styles[3], linewidth=3, markersize=6,
               label="RGD–QR")
    
    plt.xlabel("Time [seconds]", fontsize=14, fontweight='bold')
    plt.ylabel(r"Feasibility: $\|X^\top X - I\|_F$", fontsize=14, fontweight='bold')
    plt.title("Feasibility vs Time", fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout(pad=3.0)
    
    # Save figures for LaTeX
    import os
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/stiefel_benchmark.pdf", bbox_inches="tight")
    plt.show()
    
    # Create individual plots for systematic reporting
    
    # Figure 1: Error vs iterations (based on feasibility)
    plt.figure(figsize=(7, 5))
    plt.semilogy(H_fast["it"], H_fast["feas"], color=colors[0], linestyle=line_styles[0], 
                linewidth=3, label='GRAVIDY–St (Fast)')
    plt.semilogy(H_dense["it"], H_dense["feas"], color=colors[1], linestyle=line_styles[1], 
                linewidth=3, label='GRAVIDY–St (NR-Dense Fast)')
    plt.semilogy(H_wy["it"], H_wy["feas"], color=colors[2], linestyle=line_styles[2], 
                linewidth=3, label='Wen–Yin Cayley')
    plt.semilogy(H_rgd["it"], H_rgd["feas"], color=colors[3], linestyle=line_styles[3], 
                linewidth=3, label='RGD–QR')
    plt.xlabel('Iterations', fontweight='bold')
    plt.ylabel(r'$\|X^\top X - I\|_F$', fontweight='bold')
    plt.title('Stiefel: feasibility vs iterations', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.savefig("figs/stiefel_err_vs_it.pdf", bbox_inches="tight")
    plt.show()
    
    # Figure 2: Objective vs time
    plt.figure(figsize=(7, 5))
    plt.semilogy(H_fast["time"], H_fast["f"], color=colors[0], linestyle=line_styles[0], 
                linewidth=3, label='GRAVIDY–St (Fast)')
    plt.semilogy(H_dense["time"], H_dense["f"], color=colors[1], linestyle=line_styles[1], 
                linewidth=3, label='GRAVIDY–St (NR-Dense Fast)')
    plt.semilogy(H_wy["time"], H_wy["f"], color=colors[2], linestyle=line_styles[2], 
                linewidth=3, label='Wen–Yin Cayley')
    plt.semilogy(H_rgd["time"], H_rgd["f"], color=colors[3], linestyle=line_styles[3], 
                linewidth=3, label='RGD–QR')
    plt.xlabel('Time [seconds]', fontweight='bold')
    plt.ylabel(r'$f(X_k)$', fontweight='bold')
    plt.title('Stiefel: objective vs time', fontweight='bold')
    plt.grid(True, alpha=0.3, which='both', ls=':')
    plt.legend()
    plt.savefig("figs/stiefel_f_vs_time.pdf", bbox_inches="tight")
    plt.show()
    
    # Summary statistics
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY (n={}, p={})".format(n, p))
    print("="*80)
    
    # Final values
    final_obj_fast = H_fast["f"][-1]
    final_obj_dense = H_dense["f"][-1]
    final_obj_wy = H_wy["f"][-1]
    final_obj_rgd = H_rgd["f"][-1]
    
    final_feas_fast = H_fast["feas"][-1]
    final_feas_dense = H_dense["feas"][-1]
    final_feas_wy = H_wy["feas"][-1]
    final_feas_rgd = H_rgd["feas"][-1]
    
    # Compute final gradient norms
    final_grad_fast = H_fast["grad_norm"][-1] if "grad_norm" in H_fast else np.linalg.norm(prob.grad_riem(X_fast), 'fro')
    final_grad_dense = H_dense["grad_norm"][-1] if "grad_norm" in H_dense else np.linalg.norm(prob.grad_riem(X_dense), 'fro')
    final_grad_wy = H_wy["grad_norm"][-1] if "grad_norm" in H_wy else np.linalg.norm(prob.grad_riem(X_wy), 'fro')
    final_grad_rgd = H_rgd["grad_norm"][-1] if "grad_norm" in H_rgd else np.linalg.norm(prob.grad_riem(X_rgd), 'fro')
    
    print(f"{'Solver':<25} {'Final Obj':<15} {'Final Feas':<15} {'Final Grad':<15} {'Iterations':<12} {'Time [s]':<10}")
    print("-" * 100)
    print(f"{'GRAVIDY–St (Fast)':<25} {final_obj_fast:<15.6e} {final_feas_fast:<15.2e} {final_grad_fast:<15.2e} {len(H_fast['it']):<12} {H_fast['time'][-1]:<10.2f}")
    print(f"{'GRAVIDY–St (NR-Dense Fast)':<25} {final_obj_dense:<15.6e} {final_feas_dense:<15.2e} {final_grad_dense:<15.2e} {len(H_dense['it']):<12} {H_dense['time'][-1]:<10.2f}")
    print(f"{'Wen–Yin Cayley':<25} {final_obj_wy:<15.6e} {final_feas_wy:<15.2e} {final_grad_wy:<15.2e} {len(H_wy['it']):<12} {H_wy['time'][-1]:<10.2f}")
    print(f"{'RGD–QR':<25} {final_obj_rgd:<15.6e} {final_feas_rgd:<15.2e} {final_grad_rgd:<15.2e} {len(H_rgd['it']):<12} {H_rgd['time'][-1]:<10.2f}")
    print("="*100)

    return (X_fast, H_fast), (X_dense, H_dense), (X_wy, H_wy), (X_rgd, H_rgd)

if __name__ == "__main__":
    # Test on ill-conditioned problem
    run_benchmark(n=200, p=2, cond=1000.0, seed=42, verbose=False)
