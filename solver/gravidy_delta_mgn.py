"""
GRAVIDY–Δ (MGN variant): Implicit Euler with reduced-logit MGN inner solver.

This is a variant of GRAVIDY–Δ that uses Modified Gauss-Newton on reduced logits
instead of the KL-prox Newton-KKT approach. Both solve the same implicit Euler step
but with different inner solvers.
"""

import numpy as np
import time
from utils.simplex_utils import softmax, jac_softmax, center_logits


def estimate_initial_M(J, iters=10):
    """Estimate spectral radius of J^T J using power iteration."""
    m = J.shape[1]
    if m == 0:
        return 1.0
    
    v0 = np.random.randn(m)
    v0 /= (np.linalg.norm(v0) + 1e-16)
    z = v0
    
    for _ in range(iters):
        w = J.T @ (J @ z)
        wn = np.linalg.norm(w)
        if wn == 0:
            break
        z = w / wn
    
    lam = float(z.T @ (J.T @ (J @ z)))
    return lam if np.isfinite(lam) and lam > 0 else 1.0


def modified_gauss_newton_step(F, J, M):
    """Modified Gauss-Newton step: solve (J^T J + M I) h = -J^T F."""
    A = J.T @ J
    rhs = -J.T @ F
    I = np.eye(A.shape[0])
    
    try:
        L = np.linalg.cholesky(A + M * I)
        y = np.linalg.solve(L, rhs)
        h = np.linalg.solve(L.T, y)
        return h
    except np.linalg.LinAlgError:
        # Fallback to direct solve
        return np.linalg.solve(A + M * I, rhs)


def adjust_M(M, success, factor=2.0, M_min=1e-10, M_max=1e10):
    """Adjust regularization parameter M based on step success."""
    if success:
        return max(M_min, M / factor)
    else:
        return min(M_max, M * factor)


def implicit_euler_simplex_mgn(u_k_red, eta, problem, M=None, tol=1e-8, max_iter=200):
    """
    Implicit Euler in reduced logits v in R^{n-1}.
    
    Uses Modified Gauss-Newton to solve the nonlinear system arising from
    the implicit Euler step on reduced logits.
    
    Args:
        u_k_red: Current reduced logits (n-1,)
        eta: Step size parameter
        problem: Objective with f(), grad(), hess() methods
        M: Regularization parameter (estimated if None)
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        v_next: Updated reduced logits (n-1,)
        x_next: Updated simplex point (n,)
        M_next: Updated regularization parameter
    """
    H = problem.hess()
    v = u_k_red.copy()
    n = H.shape[0]

    # Initialize M if not provided
    if M is None:
        u_full0 = np.append(v, 0.0)
        Jg0 = jac_softmax(u_full0)  # (n x n)
        J_red0 = np.eye(n-1) + eta * (H @ Jg0)[:n-1, :n-1]
        M = estimate_initial_M(J_red0)

    for _ in range(max_iter):
        # Full logits with gauge-fixed last component = 0
        u_full = np.append(v, 0.0)
        x = softmax(u_full)
        grad = problem.grad(x)

        # Residual in reduced coordinates
        F_red = v - u_k_red + eta * grad[:n-1]
        res_old = np.linalg.norm(F_red)
        
        if res_old < tol:
            break

        # Jacobian in reduced coordinates
        Jg = jac_softmax(u_full)
        J_red = np.eye(n-1) + eta * (H @ Jg)[:n-1, :n-1]

        # Modified Gauss-Newton step
        dv = modified_gauss_newton_step(F_red, J_red, M)
        v_trial = v + dv

        # Accept/reject based on residual decrease
        u_full_trial = np.append(v_trial, 0.0)
        x_trial = softmax(u_full_trial)
        grad_trial = problem.grad(x_trial)
        F_trial = v_trial - u_k_red + eta * grad_trial[:n-1]
        res_new = np.linalg.norm(F_trial)

        if res_new < res_old:
            v = v_trial
            M = adjust_M(M, True)
        else:
            M = adjust_M(M, False)

    return v, softmax(np.append(v, 0.0)), M


def GRAVIDY_Delta_MGN(problem, eta=30.0, max_outer=400, tol_grad=1e-8, 
                      x0=None, verbose=False):
    """
    GRAVIDY–Δ (MGN variant) solver for simplex-constrained optimization.
    
    This is a variant that uses Modified Gauss-Newton on reduced logits
    instead of KL-prox Newton-KKT.
    
    Args:
        problem: Objective with f(), grad(), hess() methods
        eta: Implicit step size parameter
        max_outer: Maximum outer iterations
        tol_grad: Convergence tolerance on gradient
        x0: Initial point on simplex (if None, uses uniform)
        verbose: Print iteration progress
        
    Returns:
        x: Final solution on simplex
        history: List of (iter, obj_val, grad_norm, time) tuples
    """
    n = problem.n
    
    # Initialize on simplex
    if x0 is None:
        v = np.zeros(n-1)  # Reduced logits
        x = softmax(np.append(v, 0.0))
    else:
        x = x0.copy()
        # Convert to reduced logits (gauge-fix last component = 0)
        u_full = center_logits(np.log(np.clip(x, 1e-16, 1.0)))
        v = u_full[:n-1]
    
    history = []
    t0 = time.time()
    M = None  # Will be estimated automatically
    
    for k in range(max_outer):
        # Compute current objective and gradient
        obj_val = problem.f(x)
        grad = problem.grad(x)
        grad_norm = np.linalg.norm(grad)
        current_time = time.time() - t0
        
        history.append((k, obj_val, grad_norm, current_time))
        
        if verbose:
            print(f"[GRAVIDY-Δ MGN] iter={k:3d} f={obj_val:.6e} ||grad||={grad_norm:.3e} time={current_time:.2f}s")
        
        # Check convergence
        if grad_norm <= tol_grad:
            break
        
        # Implicit Euler step via MGN on reduced logits
        v, x, M = implicit_euler_simplex_mgn(v, eta, problem, M=M, tol=1e-8)
    
    return x, history
