"""
GRAVIDY–box solver for box-constrained optimization.
Implements implicit reparameterization with damped Newton.
"""

import numpy as np
import time


def sigmoid(z):
    """Sigmoid function: 1/(1 + exp(-z))"""
    return 1.0 / (1.0 + np.exp(-z))


def gravidy_box_step(z_k, eta, A, b, lo, hi, tol=1e-10, max_newton=50, backtrack_beta=0.5):
    """
    Single GRAVIDY–box step with implicit reparameterization.
    
    Reparameterization x(z) = lo + (hi-lo) ∘ sigmoid(z).
    Backward Euler in z: F(z) = z - z_k + eta * D(z) ∇f(x(z)) = 0,
    where D(z) = diag(g'(z)) and g'(z) = (hi-lo)*sigmoid(z)*(1-sigmoid(z)).
    
    Args:
        z_k: Current parameterization
        eta: Step size parameter
        A, b: Least squares problem data
        lo, hi: Box bounds
        tol: Newton tolerance
        max_newton: Maximum Newton iterations
        backtrack_beta: Backtracking factor
        
    Returns:
        z_next: Next parameterization
        x_next: Next solution in original coordinates
    """
    n = z_k.size
    H = A.T @ A  # Hessian for least squares
    z = z_k.copy()

    def g_x(z):
        """Reparameterization: x(z) = lo + (hi-lo) * sigmoid(z)"""
        s = sigmoid(z)
        return lo + (hi - lo) * s

    def gprime(z):
        """Derivative of reparameterization: g'(z) = (hi-lo) * sigmoid(z) * (1-sigmoid(z))"""
        s = sigmoid(z)
        return (hi - lo) * s * (1.0 - s)

    def gsecond(z):
        """Second derivative: g''(z) = (hi-lo) * sigmoid(z) * (1-sigmoid(z)) * (1-2*sigmoid(z))"""
        s = sigmoid(z)
        return (hi - lo) * s * (1.0 - s) * (1.0 - 2.0 * s)

    def F_val(z):
        """Residual function: F(z) = z - z_k + eta * D(z) * ∇f(x(z))"""
        x = g_x(z)
        return z - z_k + eta * (gprime(z) * (A.T @ (A @ x - b)))

    def merit(z):
        """Merit function: 0.5 * ||F(z)||^2"""
        F = F_val(z)
        return 0.5 * float(F @ F)

    for _ in range(max_newton):
        x = g_x(z)
        gp = gprime(z)
        gpp = gsecond(z)
        gradf = A.T @ (A @ x - b)

        F = z - z_k + eta * (gp * gradf)
        if np.linalg.norm(F) < tol:
            break

        # Jacobian K = I + eta * [ diag(g'' ∘ gradf) + D H D ]
        D = np.diag(gp)
        K = np.eye(n) + eta * (np.diag(gpp * gradf) + D @ H @ D)

        # Solve K h = -F
        try:
            h = -np.linalg.solve(K, F)
        except np.linalg.LinAlgError:
            # Add small damping for numerical stability
            Kd = K + 1e-10 * np.eye(n)
            h = -np.linalg.solve(Kd, F)

        # Armijo backtracking on merit function
        m0 = merit(z)
        alpha = 1.0
        while True:
            z_trial = z + alpha * h
            if merit(z_trial) <= m0 or alpha < 1e-12:
                break
            alpha *= backtrack_beta
        z = z_trial

    return z, g_x(z)


def GRAVIDY_box(problem, eta=10.0, max_outer=200, tol_grad=1e-10, inner='newton', verbose=False):
    """
    GRAVIDY–box solver for box-constrained optimization.
    
    Args:
        problem: BoxLeastSquares problem instance
        eta: Step size parameter
        max_outer: Maximum outer iterations
        tol_grad: Convergence tolerance
        verbose: Print progress
        
    Returns:
        x: Final solution
        history: List of (iter, obj_val, grad_norm, time) tuples
    """
    A = problem.A
    b = problem.b
    lo = problem.lo
    hi = problem.hi
    n = problem.n
    
    # Initialize parameterization to map to midpoint
    z = np.zeros(n)
    x = lo + (hi - lo) * sigmoid(z)
    
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
            print(f"[GRAVIDY-box/{inner}] iter={k:4d} f={obj_val:.6e} ||grad||={grad_norm:.3e} time={current_time:.2f}s")
        
        # Check convergence
        if grad_norm <= tol_grad:
            break
        
        # GRAVIDY–box step
        if inner == 'newton':
            z, x = gravidy_box_step(z, eta, A, b, lo, hi, tol=1e-10)
        elif inner == 'mgn':
            # Import MGN step from the new module
            from .gravidy_box_mgn import gravidy_box_step_mgn, BoxLeastSquares
            prob_mgn = BoxLeastSquares(A, b, lo, hi)
            z, x = gravidy_box_step_mgn(z, eta, prob_mgn, tol=1e-10, max_iter=50)
        else:
            raise ValueError("inner must be 'newton' or 'mgn'")
    
    return x, history
