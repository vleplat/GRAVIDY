#!/usr/bin/env python3
"""
Test script for GRAVIDY–St (NR-Dense Fast) on larger problem
"""

import time
import math
import numpy as np
from solver.gravidy_st_nr_dense import ICS_NR_dense_fast, StiefelQuad
from utils.stiefel_utils import rand_stiefel

def make_problem(n=400, p=4, cond=1000.0, seed=0):
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

if __name__ == "__main__":
    print("=== Testing GRAVIDY–St (NR-Dense Fast) ===")
    print("Problem: n=400, p=4, condition=1000.0")
    print("-" * 50)
    
    # Create problem
    prob = make_problem(n=400, p=4, cond=1000.0, seed=42)
    X0 = rand_stiefel(400, 4, seed=123)
    
    # Improved starting point scaling
    A0 = prob.A_skew(X0)
    g0 = prob.grad_riem(X0)
    A0_norm = np.linalg.norm(A0, 'fro')
    scale_A = 1.0 / max(1e-12, A0_norm)
    alpha0 = 0.5 * scale_A * 4  # Use p=4 for scaling
    
    print(f"Initial gradient norm: {np.linalg.norm(g0, 'fro'):.2e}")
    print(f"Initial objective: {prob.f(X0):.6e}")
    print(f"Initial feasibility: {np.linalg.norm(X0.T @ X0 - np.eye(4), 'fro'):.2e}")
    print(f"Alpha0: {alpha0:.2e}")
    print("-" * 50)
    
    # Run optimized dense solver
    start_time = time.perf_counter()
    X_final, history = ICS_NR_dense_fast(
        prob, X0, 
        alpha0=alpha0, 
        max_outer=500, 
        tol_grad=1e-6,
        verbose=True
    )
    total_time = time.perf_counter() - start_time
    
    # Final results
    final_grad = np.linalg.norm(prob.grad_riem(X_final), 'fro')
    final_obj = prob.f(X_final)
    final_feas = np.linalg.norm(X_final.T @ X_final - np.eye(4), 'fro')
    
    print("-" * 50)
    print("FINAL RESULTS:")
    print(f"Final gradient norm: {final_grad:.2e}")
    print(f"Final objective: {final_obj:.6e}")
    print(f"Final feasibility: {final_feas:.2e}")
    print(f"Total iterations: {len(history)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Time per iteration: {total_time/len(history):.3f}s")
    
    # Convergence analysis
    if len(history) > 1:
        grad_norms = [h[2] for h in history]
        print(f"Gradient norm reduction: {grad_norms[0]:.2e} → {grad_norms[-1]:.2e}")
        print(f"Convergence rate: {grad_norms[-1]/grad_norms[0]:.2e}")
