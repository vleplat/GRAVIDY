"""
GRAVIDY–St (NR-Dense) - Optimized dense Newton-Raphson solver for Stiefel manifolds
Implicit Cayley-Stiefel with dense Newton-Raphson inner solver (optimized version)
"""

import numpy as np
import time
from scipy.linalg import lu_factor, lu_solve

def sym(M): 
    return 0.5*(M+M.T)

class StiefelQuad:
    """
    Phi(X) = 0.5 * sum_j x_j^T Q_j x_j
    grad_R(X) = G - X sym(X^T G),  G = [Q_1 x_1, ..., Q_p x_p]
    A(Y) = G(Y) Y^T - Y G(Y)^T
    """
    def __init__(self, Q_list):
        self.Q_list = [np.ascontiguousarray(Q, dtype=np.float64) for Q in Q_list]
        self.n = Q_list[0].shape[0]
        self.p = len(Q_list)

    def f(self, X):
        X = np.ascontiguousarray(X)
        s = 0.0
        for j in range(self.p):
            xj = X[:, j]
            s += 0.5 * float(xj.T @ (self.Q_list[j] @ xj))
        return s

    def apply_Q(self, X):
        X = np.ascontiguousarray(X)
        G = np.empty_like(X)
        for j in range(self.p):
            G[:, j] = self.Q_list[j] @ X[:, j]
        return G

    def grad_riem(self, X):
        G = self.apply_Q(X)
        return G - X @ sym(X.T @ G)

    def A_skew(self, X):
        G = self.apply_Q(X)
        return G @ X.T - X @ G.T

# ---- vec/unvec in Fortran order (column stacking) ----
def vecF(M): 
    return np.reshape(M, (-1,), order='F')

def unvecF(v, n, p): 
    return np.reshape(v, (n, p), order='F')

# ---- ICS residual and exact Fréchet derivative action ----
def F_of_Y(problem, Y, Xk, a, work=None):
    n = problem.n
    A = problem.A_skew(Y)
    # (I + 0.5 a A)Y - (I - 0.5 a A)Xk
    return (Y + 0.5*a*(A @ Y)) - (Xk - 0.5*a*(A @ Xk))

def JF_action(problem, Y, H, Xk, a, G_Xk=None, work=None):
    """
    JF[Y][H] = (I + a/2 A(Y)) H + (a/2) * W * (Y + Xk),
      W = (G(Xk)H^T - H G(Xk)^T) + (G(H) Xk^T - Xk G(H)^T)
    """
    if G_Xk is None:
        G_Xk = problem.apply_Q(Xk)
    A_Y  = problem.A_skew(Y)
    G_H  = problem.apply_Q(H)
    W = (G_Xk @ H.T - H @ G_Xk.T) + (G_H @ Xk.T - Xk @ G_H.T)
    return (H + 0.5*a*(A_Y @ H)) + 0.5*a * (W @ (Y + Xk))

def build_dense_J(problem, Y, Xk, a, G_Xk=None, out=None):
    """Dense (np x np) Jacobian assembled by columns (F-order)."""
    n, p = problem.n, problem.p
    N = n*p
    if out is None:
        J = np.empty((N, N), dtype=np.float64, order='F')
    else:
        J = out
    if G_Xk is None:
        G_Xk = problem.apply_Q(Xk)

    # Use Fortran-ordered basis to keep memory friendly
    for col in range(N):
        # unit vector in vecF basis
        # create H with a single 1 at position 'col' in F-order
        e = np.zeros(N, dtype=np.float64)
        e[col] = 1.0
        E = unvecF(e, n, p)
        JE = JF_action(problem, Y, E, Xk, a, G_Xk=G_Xk)
        J[:, col] = vecF(JE)
    return J

def ICS_NR_dense_fast(problem,
                      X0,
                      alpha0=1.0,
                      tol_grad=1e-8,
                      newton_tol=1e-10,
                      max_outer=200,
                      max_inner=10,
                      ls_beta=0.5,
                      alpha_grow=1.5,
                      alpha_shrink=0.5,
                      alpha_min=1e-8,
                      alpha_max=1e8,
                      verbose=True):
    """
    Same Newton–Raphson algorithm, faster calls:
      - LU factorization (lu_factor / lu_solve) for the dense linear solve
      - Fortran layout everywhere (cheaper vec/unvec)
      - Reuse temporaries to reduce allocations
    """
    n, p = X0.shape
    X = np.array(X0, dtype=np.float64, order='F')
    alpha = float(alpha0)

    # preallocate scratch
    J = np.empty((n*p, n*p), dtype=np.float64, order='F')
    rhs = np.empty(n*p, dtype=np.float64, order='F')

    f = problem.f
    history = []
    t0 = time.time()

    for k in range(max_outer):
        g = problem.grad_riem(X)
        gnorm = np.linalg.norm(g, 'fro')
        feas  = np.linalg.norm(X.T @ X - np.eye(p), 'fro')
        fval  = f(X)
        history.append((k, fval, gnorm, feas, time.time() - t0))
        if verbose:
            print(f"[ICS-NR-fast] it={k:3d} f={fval:.6e} ||grad||={gnorm:.2e} "
                  f"feas={feas:.2e} alpha={alpha:.2e}")

        if gnorm <= tol_grad:
            break

        # cache G(Xk) for this outer iteration
        G_Xk = problem.apply_Q(X)

        # ----- Inner Newton: solve F(Y)=0 to tolerance -----
        Y = np.array(X, copy=True, order='F')
        converged = False
        for it in range(max_inner):
            F = F_of_Y(problem, Y, X, alpha)
            rF = np.linalg.norm(F, 'fro')
            if rF <= newton_tol * (1.0 + np.linalg.norm(X, 'fro')):
                converged = True
                break

            # J vec(H) = -vec(F)
            build_dense_J(problem, Y, X, alpha, G_Xk=G_Xk, out=J)
            rhs[:] = -vecF(F)

            # fast LU solve
            try:
                lu, piv = lu_factor(J, overwrite_a=True, check_finite=False)
                h = lu_solve((lu, piv), rhs, overwrite_b=True, check_finite=False)
            except Exception:
                # fallback
                h = np.linalg.lstsq(J, rhs, rcond=None)[0]

            H = unvecF(h, n, p)

            # merit backtracking on m(Y)=0.5||F||^2
            m0 = 0.5 * (rF*rF)
            t = 1.0
            while True:
                Y_trial = Y + t * H
                Ft = F_of_Y(problem, Y_trial, X, alpha)
                mt = 0.5 * float(np.linalg.norm(Ft, 'fro')**2)
                if (mt <= m0) or (t < 1e-12):
                    Y = Y_trial
                    break
                t *= ls_beta

        # not converged ⇒ shrink α and retry
        if not converged:
            alpha = max(alpha * alpha_shrink, alpha_min)
            continue

        # Accept update (we solved F≈0); adapt α by objective decrease
        f_new = f(Y)
        if f_new <= fval - 1e-4 * alpha * (gnorm ** 2):
            X = Y
            alpha = min(alpha * alpha_grow, alpha_max)
        else:
            X = Y
            alpha = max(alpha * alpha_shrink, alpha_min)

    return X, history

# Alias for backward compatibility
ICS_gravidy_NR_dense = ICS_NR_dense_fast
