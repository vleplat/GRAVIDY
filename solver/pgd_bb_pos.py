"""
Projected Barzilai-Borwein (BB) solver for positive orthant optimization.
External baseline method for comparison with GRAVIDY–pos.
"""

import numpy as np
import time


def power_iteration_lmax(Mv, n, iters=50):
    """Estimate largest eigenvalue of a PSD matrix via matrix-vector product Mv(v)"""
    v = np.random.randn(n)
    v /= np.linalg.norm(v) + 1e-16
    lam = None
    for _ in range(iters):
        w = Mv(v)
        lam_new = float(v @ w)
        wn = np.linalg.norm(w)
        if wn == 0:
            break
        v = w / wn
        lam = lam_new
    if lam is None or not np.isfinite(lam) or lam <= 0:
        lam = 1.0
    return lam


def PGD_BB_pos(problem, max_iters=400, x0=None, window_monotone=True, tol_grad=1e-8, verbose=False):
    """
    Projected Barzilai-Borwein (BB) solver for positive orthant optimization.
    
    Args:
        problem: PositiveLeastSquares problem instance
        max_iters: Maximum iterations
        x0: Initial point (if None, uses random nonnegative start)
        window_monotone: Whether to use monotone or nonmonotone line search
        tol_grad: Convergence tolerance
        verbose: Print iteration progress
        
    Returns:
        x: Final solution
        history: List of (iter, obj_val, grad_norm, time) tuples
    """
    n = problem.n
    A = problem.A
    
    # Initialize with random nonnegative start
    if x0 is None:
        x = np.maximum(np.random.randn(n), 0.0)
    else:
        x = np.maximum(x0, 0.0)

    g = problem.grad(x)
    f = problem.f(x)
    
    # Initial step size estimate
    H = lambda v: A.T @ (A @ v)
    tau = 1.0 / power_iteration_lmax(H, n, iters=60)

    # Nonmonotone window (optional)
    M = 5
    Fwin = [f]

    history = []
    t0 = time.time()
    
    for k in range(max_iters):
        # Record current state
        current_time = time.time() - t0
        grad_norm = np.linalg.norm(g)
        history.append((k, f, grad_norm, current_time))
        
        if verbose and k % 50 == 0:
            print(f"[PGD-BB-pos] iter={k:4d} f={f:.6e} ||grad||={grad_norm:.3e} time={current_time:.2f}s")
        
        # Check convergence
        if grad_norm <= tol_grad:
            break
        
        # Trial BB step
        v = x - tau * g
        x_trial = np.maximum(v, 0.0)

        f_trial = problem.f(x_trial)
        
        # Armijo-like acceptance (monotone or nonmonotone)
        f_ref = max(Fwin) if (not window_monotone and len(Fwin) > 0) else f
        c = 1e-4
        
        # Sufficient decrease vs. linearized bound
        while f_trial > f_ref - c * np.linalg.norm(x_trial - x)**2 / max(tau, 1e-16):
            tau *= 0.5
            v = x - tau * g
            x_trial = np.maximum(v, 0.0)
            f_trial = problem.f(x_trial)

        s = x_trial - x
        x_prev, g_prev, f_prev = x, g, f
        x = x_trial
        g = problem.grad(x)
        f = f_trial

        # Update BB step size
        yk = g - g_prev
        sy = float(s @ yk)
        if sy > 1e-16:
            tau = float(s @ s) / sy
            # Cap tau for stability
            tau = np.clip(tau, 1e-12, 1e12)
        else:
            # Fallback to spectral step
            tau = 1.0 / power_iteration_lmax(H, n, iters=30)

        Fwin.append(f)
        if len(Fwin) > M:
            Fwin.pop(0)

    return x, history
