"""
GRAVIDY–Δ (KL-prox): Implicit Euler with KL-prox and Newton-KKT inner solver.

Solves the implicit step:
    x_{k+1} = argmin_{x in simplex} KL(x || softmax(u_k)) + eta * f(x)

using Newton-KKT with Schur complement for the sum constraint.
"""

import numpy as np
import time
from utils.simplex_utils import softmax, safe_log, center_logits


def kl_prox_simplex_step(u_k, eta, problem, tol=1e-10, max_newton=100, backtrack_beta=0.5):
    """
    One implicit Euler step in logits via KL-prox on the simplex.

    Solves: x_{k+1} = argmin_{x in simplex} KL(x || softmax(u_k)) + eta * f(x)
    
    Uses Newton-KKT solve on the simplex with Schur complement for the sum constraint.
    
    Args:
        u_k: Current logits (n,)
        eta: Step size parameter
        problem: Objective function with f(), grad(), hess() methods
        tol: Newton convergence tolerance
        max_newton: Maximum Newton iterations
        backtrack_beta: Backtracking line search parameter
        
    Returns:
        u_next: Updated logits (n,)
        x_next: Updated simplex point (n,)
    """
    n = len(u_k)
    H = problem.hess()  # Hessian of f (constant for least squares)
    p = softmax(u_k)    # Reference distribution
    x = p.copy()        # Warm start from reference
    ones = np.ones(n)

    def R_value(x):
        """Combined objective: KL(x||p) + eta * f(x)"""
        x_safe = np.clip(x, 1e-16, 1.0)
        kl_term = float(np.sum(x_safe * (safe_log(x_safe) - safe_log(p))))
        return kl_term + eta * problem.f(x)

    for it in range(max_newton):
        x_safe = np.clip(x, 1e-16, 1.0)
        
        # Gradient and Hessian of combined objective
        g = safe_log(x_safe) - safe_log(p) + eta * problem.grad(x)  # gradient (R^n)
        K = np.diag(1.0 / x_safe) + eta * H  # SPD on tangent space
        
        # Solve KKT system:
        #   K y = g,   K z = 1,   lambda from (1^T y + lambda 1^T z = 0)
        try:
            L = np.linalg.cholesky(K)
            y = np.linalg.solve(L.T, np.linalg.solve(L, g))
            z = np.linalg.solve(L.T, np.linalg.solve(L, ones))
        except np.linalg.LinAlgError:
            # Add tiny tangential regularization to maintain SPD numerics
            K_reg = K + 1e-12 * (np.eye(n) - np.outer(ones, ones) / n)
            try:
                L = np.linalg.cholesky(K_reg)
                y = np.linalg.solve(L.T, np.linalg.solve(L, g))
                z = np.linalg.solve(L.T, np.linalg.solve(L, ones))
            except np.linalg.LinAlgError:
                # Final fallback: use least squares
                y = np.linalg.lstsq(K, g, rcond=1e-12)[0]
                z = np.linalg.lstsq(K, ones, rcond=1e-12)[0]

        # Lagrange multiplier and Newton direction
        denom = float(ones @ z)
        if abs(denom) < 1e-14:
            denom = 1e-14 if denom >= 0 else -1e-14
        dlambda = -float(ones @ y) / denom
        dx = -y - z * dlambda  # Ensures sum(dx) = 0

        # Fraction-to-the-boundary step to preserve positivity
        neg = dx < 0
        if np.any(neg):
            alpha_max = 0.99 * np.min(-x[neg] / dx[neg])
            alpha = min(1.0, float(alpha_max))
        else:
            alpha = 1.0

        # Armijo backtracking on combined objective R(x)
        R0 = R_value(x)
        while True:
            x_trial = x + alpha * dx
            if (x_trial > 0).all():
                R1 = R_value(x_trial)
                if R1 <= R0 or alpha < 1e-10:
                    break
            alpha *= backtrack_beta

        x = x_trial

        # Check convergence
        if np.linalg.norm(dx, 1) < tol:
            break

    # Return centered logits for the next outer step (remove gauge freedom)
    u_next = safe_log(x)
    u_next = center_logits(u_next)
    
    return u_next, x


def GRAVIDY_Delta_KLprox(problem, eta=30.0, max_outer=400, tol_grad=1e-8, 
                         x0=None, verbose=False):
    """
    GRAVIDY–Δ (KL-prox) solver for simplex-constrained optimization.
    
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
        u = np.zeros(n)  # Centered logits => uniform distribution
        x = softmax(u)
    else:
        x = x0.copy()
        u = center_logits(safe_log(x))
    
    history = []
    t0 = time.time()
    
    for k in range(max_outer):
        # Compute current objective and gradient
        obj_val = problem.f(x)
        grad = problem.grad(x)
        grad_norm = np.linalg.norm(grad)
        current_time = time.time() - t0
        
        history.append((k, obj_val, grad_norm, current_time))
        
        if verbose:
            print(f"[GRAVIDY-Δ KL] iter={k:3d} f={obj_val:.6e} ||grad||={grad_norm:.3e} time={current_time:.2f}s")
        
        # Check convergence
        if grad_norm <= tol_grad:
            break
        
        # Implicit Euler step via KL-prox
        u, x = kl_prox_simplex_step(u, eta, problem, tol=1e-10)
    
    return x, history
