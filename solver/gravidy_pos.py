"""
GRAVIDY–pos solver for positive orthant optimization (NNLS).
Implements implicit Euler in log-coordinates with damped Newton.
"""

import numpy as np
import time


def safe_exp(u):
    """Clamp to avoid overflow/underflow; keeps positivity mapping numerically stable"""
    return np.exp(np.clip(u, -40.0, 40.0))


def gravidy_pos_step(u_k, eta, A, b, tol=1e-10, max_newton=50, backtrack_beta=0.5, damping=1e-10):
    """
    Single GRAVIDY–pos step with implicit Euler in log-coordinates.
    
    Reparameterization: x = g(u) = exp(u), d = g'(u) = x, d'' = g''(u) = x
    Implicit step: F(u) = u - u_k + eta * D(u) ∇f(x(u)) = 0
    Newton Jacobian: K = I + eta * [ diag(d'' ∘ ∇f(x)) + D H D ], H = A^T A
    
    Args:
        u_k: Current log-coordinates
        eta: Step size parameter
        A, b: Least squares problem data
        tol: Newton tolerance
        max_newton: Maximum Newton iterations
        backtrack_beta: Backtracking factor
        damping: Tikhonov damping for numerical stability
        
    Returns:
        u_next: Next log-coordinates
        x_next: Next solution in original coordinates
    """
    n = u_k.size
    H = A.T @ A

    def x_of(u):  # positivity guaranteed
        return safe_exp(u)

    def F_val(u):
        x = x_of(u)
        return u - u_k + eta * (x * (A.T @ (A @ x - b)))

    def merit(u):
        F = F_val(u)
        return 0.5 * float(F @ F)

    u = u_k.copy()
    for _ in range(max_newton):
        x = x_of(u)
        g = A.T @ (A @ x - b)
        F = u - u_k + eta * (x * g)
        nF = np.linalg.norm(F)
        if nF < tol:
            break

        D = np.diag(x)  # diag(d)
        # diag(d'' ∘ g) = diag(x ∘ g) because d''=x for exp
        K = np.eye(n) + eta * (np.diag(x * g) + D @ H @ D)

        # solve for Newton step
        try:
            # small Tikhonov helps if K is near singular
            h = -np.linalg.solve(K + damping * np.eye(n), F)
        except np.linalg.LinAlgError:
            h = -np.linalg.pinv(K) @ F

        # Armijo backtracking on m(u) = 0.5||F(u)||^2
        m0 = merit(u)
        alpha = 1.0
        while True:
            u_trial = u + alpha * h
            if merit(u_trial) <= m0 or alpha < 1e-12:
                break
            alpha *= backtrack_beta
        u = u_trial

    x_next = x_of(u)
    return u, x_next


def GRAVIDY_pos(problem, eta=30.0, max_outer=400, tol_grad=1e-10, inner='newton', verbose=False):
    """
    GRAVIDY–pos solver for positive orthant optimization.
    
    Args:
        problem: PositiveLeastSquares problem instance
        eta: Step size parameter
        max_outer: Maximum outer iterations
        tol_grad: Convergence tolerance
        inner: Inner solver ('newton' or 'mgn')
        verbose: Print progress
        
    Returns:
        x: Final solution
        history: List of (iter, obj_val, grad_norm, time) tuples
    """
    A = problem.A
    b = problem.b
    n = problem.n
    
    # Initialize from midpoint in log-space: u=0 => x=1
    u = np.zeros(n)
    x = safe_exp(u)
    
    history = []
    t0 = time.time()
    
    for k in range(max_outer):
        # Compute objective and gradient
        obj_val = problem.f(x)
        grad = problem.grad(x)
        grad_norm = np.linalg.norm(grad)
        current_time = time.time() - t0
        
        history.append((k, obj_val, grad_norm, current_time))
        
        if verbose and k % 50 == 0:
            print(f"[GRAVIDY-pos/{inner}] iter={k:4d} f={obj_val:.6e} ||grad||={grad_norm:.3e} time={current_time:.2f}s")
        
        # Check convergence
        if grad_norm <= tol_grad:
            break
        
        # GRAVIDY–pos step with chosen inner solver
        if inner == 'newton':
            u, x = gravidy_pos_step(u, eta, A, b, tol=1e-10)
        elif inner == 'mgn':
            # Import and use MGN solver
            from .gravidy_pos_mgn import gravidy_pos_step_mgn, PositiveLeastSquares
            prob_wrapper = PositiveLeastSquares(A, b)
            u, x = gravidy_pos_step_mgn(u, eta, prob_wrapper, tol=1e-10, max_iter=50)
        else:
            raise ValueError("inner must be 'newton' or 'mgn'")
    
    return x, history
