"""
GRAVIDY–St (Fast) - Optimized Newton-Krylov solver for Stiefel manifolds
Implicit Cayley-Stiefel with Newton-Krylov (GMRES) inner solver (optimized version)
"""

import numpy as np
import time
import scipy.linalg as la
from scipy.sparse.linalg import gmres, LinearOperator

# ---------- helpers ----------
def sym(M):  
    return 0.5*(M + M.T)

def skew(M): 
    return 0.5*(M - M.T)

def retract(Y, method="polar"):
    """
    Retraction to St(n,p).
    method="polar" (closest in Fro norm) or "qr" (faster).
    """
    if method == "qr":
        # economy QR: Y = QR, return Q
        Q, _ = np.linalg.qr(Y, mode='reduced')
        return Q
    # polar via thin SVD
    U, _, Vt = np.linalg.svd(Y, full_matrices=False)
    return U @ Vt

def tangent_project(X, V):
    return V - X @ sym(X.T @ V)

def spectral_init_avgQ(Q_list, p):
    n = Q_list[0].shape[0]
    Qbar = sum(Q_list) / len(Q_list)
    w, V = la.eigh(Qbar)
    idx = np.argsort(w)[:p]
    X0 = V[:, idx]
    U, _, Vt = np.linalg.svd(X0, full_matrices=False)
    return U @ Vt

# ---------- problem ----------
class StiefelQuad:
    """
    Phi(X) = 0.5 * sum_j x_j^T Q_j x_j  on St(n,p)
    Riemannian grad = G - X sym(X^T G), where G = apply_Q(X)
    """
    def __init__(self, Q_list):
        # keep contiguous float64 for BLAS
        self.Q_list = [np.ascontiguousarray(Q, dtype=np.float64) for Q in Q_list]
        self.n = Q_list[0].shape[0]
        self.p = len(Q_list)

    def f(self, X):
        X = np.ascontiguousarray(X)
        return 0.5 * sum((X[:, j].T @ (self.Q_list[j] @ X[:, j]))
                         for j in range(self.p))

    def apply_Q(self, X):
        X = np.ascontiguousarray(X)
        G = np.empty_like(X)
        for j, Q in enumerate(self.Q_list):
            G[:, j] = Q @ X[:, j]
        return G

    def grad_riem(self, X):
        G = self.apply_Q(X)
        return G - X @ sym(X.T @ G)

    def A_skew(self, X):
        G = self.apply_Q(X)
        return G @ X.T - X @ G.T   # skew-symmetric (rank ≤ 2p)


# ---------- Woodbury-based (I + c A(Y))^{-1} with LU cache ----------
def build_cayley_cache(problem, Y, alpha):
    """
    Cache small (2p x 2p) LU factorization for repeated applies of (I + c A(Y))^{-1}.
    A(Y) = U V^T - V U^T, with U = Q(Y), V = Y, c = alpha/2.
    """
    c = 0.5 * alpha
    U = problem.apply_Q(Y)                  # n x p
    V = Y                                   # n x p
    W = np.concatenate([U, V], axis=1, dtype=np.float64)  # n x 2p
    Z = np.concatenate([V, U], axis=1, dtype=np.float64)  # n x 2p
    p = V.shape[1]
    # S = diag(+c I_p, -c I_p)
    S = np.zeros((2*p, 2*p), dtype=np.float64)
    S[:p, :p] =  c * np.eye(p)
    S[p:, p:] = -c * np.eye(p)
    # M = I + S (Z^T W)  (2p x 2p)
    M = np.eye(2*p, dtype=np.float64) + S @ (Z.T @ W)
    # LU once; avoid explicit inverse
    lu, piv = la.lu_factor(M, overwrite_a=True, check_finite=False)
    # Precompute SZt = S @ Z^T for faster RHS formation
    SZt = S @ Z.T
    return (W, SZt, (lu, piv))   # compact cache

def apply_Linv(R, cache):
    """
    Apply (I + c A)^(-1) R via Woodbury: R - W * solve(M, S Z^T R)
    cache = (W, SZt, (lu, piv))
    """
    W, SZt, lupiv = cache
    rhs_small = SZt @ R                      # (2p x p)
    y = la.lu_solve(lupiv, rhs_small, check_finite=False, overwrite_b=True)
    return R - W @ y


# ---------- residual and Jacobian action (with cached pieces) ----------
def F_of_Y(problem, Y, Xk, alpha, A_Y=None):
    c = 0.5 * alpha
    if A_Y is None:
        A_Y = problem.A_skew(Y)
    # (I + c A)Y - (I - c A)Xk  = Y + c A Y  - Xk + c A Xk
    return (Y + c*(A_Y @ Y)) - (Xk - c*(A_Y @ Xk))

def JF_action_cached(problem, Y, H, Xk, alpha, A_Y, G_Xk, Y_plus_Xk):
    """
    Same as JF_action but A(Y), G(Xk), and (Y + Xk) are precomputed and passed in.
    JF[Y][H] = (I + c A_Y) H + c * W * (Y + Xk),
      W = (G(Xk)H^T - H G(Xk)^T) + (G(H) Xk^T - Xk G(H)^T)
    """
    c = 0.5 * alpha
    G_H = problem.apply_Q(H)
    W = (G_Xk @ H.T - H @ G_Xk.T) + (G_H @ Xk.T - Xk @ G_H.T)
    return (H + c*(A_Y @ H)) + c * (W @ Y_plus_Xk)


# ---------- predictor (feasible) ----------
def symmetric_cayley_predictor(problem, Xk, Xkm1, alpha_k, alpha_km1, retract_method="polar"):
    """
    BDF2-like extrapolation + one Cayley transform built at the extrapolated point.
    Always returns a feasible Y0.
    """
    # On-manifold extrapolation (optional)
    if Xkm1 is not None and alpha_km1 is not None and alpha_km1 > 0:
        lam = alpha_k / alpha_km1
        V = tangent_project(Xk, Xk - Xkm1)
        Xk_ex = retract(Xk + lam * V, method=retract_method)
    else:
        Xk_ex = Xk

    # Cayley predictor Y0 ≈ Q_{Xk_ex} Xk = (I - c A(Xk_ex))(I + c A(Xk_ex))^{-1} Xk
    cay_cache = build_cayley_cache(problem, Xk_ex, alpha_k)
    W = apply_Linv(Xk.copy(), cay_cache)          # W = (I + c A)^{-1} Xk
    U = problem.apply_Q(Xk_ex);  V = Xk_ex;  c = 0.5 * alpha_k
    Y = W - c * (U @ (V.T @ W) - V @ (U.T @ W))   # (I - c A) W
    return retract(Y, method=retract_method)


# ---------- main ICS (Newton–Krylov with Cayley preconditioning) ----------
def ICS_gravidy_fast(problem, X0, alpha0=1.0, tol_grad=1e-8,
                     max_outer=200, max_inner=5, gmres_maxit=200,
                     use_spectral_init=True, retract_method="polar",
                     verbose=True):
    """
    Implicit Cayley–Stiefel (ICS): trapezoidal step + NK (GMRES)
      - Predictor: symmetric Cayley (feasible)
      - Corrector: inexact Newton with left-preconditioned GMRES
        preconditioner L^{-1} = (I + (α/2)A(Y))^{-1} via Woodbury+LU
      - Drop-in compatible with your earlier ICS; faster low-level calls.
    """
    n, p = X0.shape
    # Initialize feasibly (spectral is robust and cheap)
    X = spectral_init_avgQ(problem.Q_list, p) if use_spectral_init else retract(X0, method=retract_method)
    alpha = float(alpha0)

    hist = []
    t0 = time.time()
    X_prev, alpha_prev = None, None

    for k in range(max_outer):
        g = problem.grad_riem(X)
        gnorm = np.linalg.norm(g, 'fro')
        feas = np.linalg.norm(X.T @ X - np.eye(p), 'fro')
        fval = problem.f(X)
        elapsed = time.time() - t0
        hist.append((k, fval, gnorm, feas, elapsed))
        if verbose:
            print(f"[ICS-fast] it={k:3d} f={fval:.6e} ||grad||={gnorm:.2e} "
                  f"feas={feas:.2e} α={alpha:.2e} t={elapsed:.2f}s")
        if gnorm <= tol_grad:
            break

        # ---- predictor (always feasible) ----
        Y = symmetric_cayley_predictor(problem, X, X_prev, alpha, alpha_prev, retract_method=retract_method)
        X_prev, alpha_prev = X.copy(), alpha

        # ---- corrector: inexact Newton with left-preconditioned GMRES ----
        for it in range(max_inner):
            # cache quantities that stay fixed within a GMRES linear solve
            A_Y = problem.A_skew(Y)
            R = -F_of_Y(problem, Y, X, alpha, A_Y=A_Y)
            Rn = np.linalg.norm(R, 'fro')
            if Rn <= 1e-12 * (1.0 + np.linalg.norm(X, 'fro')):
                break

            # Precompute once per inner iteration
            G_Xk = problem.apply_Q(X)
            Y_plus_Xk = Y + X
            cay_cache = build_cayley_cache(problem, Y, alpha)

            def matvec(hvec):
                H = hvec.reshape(n, p, order='F')
                JH = JF_action_cached(problem, Y, H, X, alpha, A_Y, G_Xk, Y_plus_Xk)
                Z  = apply_Linv(JH, cay_cache)
                return Z.reshape(-1, order='F')

            b = apply_Linv(R, cay_cache).reshape(-1, order='F')

            Aop = LinearOperator(shape=(n*p, n*p), matvec=matvec, dtype=np.float64)
            # Eisenstat–Walker forcing term (loose early, tighter later)
            rtol = min(1e-1, max(1e-6, 0.1*Rn))
            hvec, info = gmres(Aop, b, rtol=rtol, atol=0.0, restart=None, maxiter=gmres_maxit)
            H = hvec.reshape(n, p, order='F')

            # feasibility-preserving update
            Y = retract(Y + H, method=retract_method)

        # ---- accept + stepsize adapt ----
        f_new = problem.f(Y)
        if f_new <= fval - 1e-4 * alpha * (gnorm**2):
            X = Y
            alpha = min(alpha * 1.5, 1e8)
        else:
            alpha = max(alpha * 0.5, 1e-8)

    return X, hist


# ---------- quick sanity test ----------
if __name__ == "__main__":
    np.random.seed(0)
    n, p = 200, 10
    # simple diagonal SPD Q's
    Q_list = [np.diag(np.linspace(0.1, 10.0, n)) for _ in range(p)]
    prob = StiefelQuad(Q_list)
    X0 = np.random.randn(n, p)
    X0 = retract(X0, method="qr")  # any feasible start
    X, hist = ICS_gravidy_fast(prob, X0, alpha0=1.0, tol_grad=1e-6, retract_method="qr", verbose=True)
    print("final f:", prob.f(X))
