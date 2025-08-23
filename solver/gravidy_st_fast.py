import numpy as np
import time
import scipy.linalg as la
from scipy.sparse.linalg import gmres, LinearOperator

# ---------- helpers ----------
def sym(M):  return 0.5*(M + M.T)
def skew(M): return 0.5*(M - M.T)

def polar_retract(Y):
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
        self.Q_list = Q_list
        self.n = Q_list[0].shape[0]
        self.p = len(Q_list)

    def f(self, X):
        return 0.5 * sum((X[:, j].T @ (self.Q_list[j] @ X[:, j]))
                         for j in range(self.p))

    def apply_Q(self, X):
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


# ---------- Woodbury-based inverse (I + c A(Y))^{-1} ----------
def build_cayley_cache(problem, Y, alpha):
    """
    Cache small (2p x 2p) factorization for (I + c A(Y))^{-1} applies during GMRES.
    A(Y) = U V^T - V U^T, with U = Q(Y), V = Y, c = alpha/2.
    """
    c = 0.5 * alpha
    U = problem.apply_Q(Y)            # n x p
    V = Y                             # n x p
    W = np.concatenate([U, V], axis=1)  # n x 2p
    Z = np.concatenate([V, U], axis=1)  # n x 2p
    p = V.shape[1]
    S = la.block_diag(c * np.eye(p), -c * np.eye(p))  # 2p x 2p
    # M = I + S Z^T W  (2p x 2p)
    M = np.eye(2*p) + S @ (Z.T @ W)
    # factorize M once
    try:
        Minv = la.inv(M)
        fact = ('inv', Minv)
    except la.LinAlgError:
        # Robust fallback: use pinv (2p is small)
        Minv = la.pinv(M)
        fact = ('pinv', Minv)
    return (W, Z, S, fact)

def apply_Linv(R, cache):
    """
    Apply (I + c A)^(-1) R via Woodbury: R - W (I + S Z^T W)^{-1} S (Z^T R).
    cache = (W, Z, S, fact)
    """
    W, Z, S, fact = cache
    rhs_small = S @ (Z.T @ R)                 # 2p x p
    kind, F = fact
    # Since F is Minv, multiply Minv @ rhs_small
    y = F @ rhs_small
    return R - W @ y

# ---------- residual and Jacobian action ----------
def F_of_Y(problem, Y, Xk, alpha):
    c = 0.5 * alpha
    A = problem.A_skew(Y)
    I = np.eye(problem.n)
    return (I + c * A) @ Y - (I - c * A) @ Xk

def JF_action(problem, Y, H, Xk, alpha):
    """
    Directional derivative of F at Y in direction H.
    """
    n = problem.n
    c = 0.5 * alpha
    A_Y  = problem.A_skew(Y)
    G_Xk = problem.apply_Q(Xk)
    G_H  = problem.apply_Q(H)
    # W is the directional derivative of A(.) applied to (Y + Xk)
    W = (G_Xk @ H.T - H @ G_Xk.T) + (G_H @ Xk.T - Xk @ G_H.T)
    return (np.eye(n) + c * A_Y) @ H + c * (W @ (Y + Xk))

# ---------- predictor ----------
def symmetric_cayley_predictor(problem, Xk, Xkm1, alpha_k, alpha_km1):
    """
    BDF2-like extrapolation + one Cayley transform built at the extrapolated point.
    Always returns a feasible Y0.
    """
    # on-manifold extrapolation
    if Xkm1 is not None and alpha_km1 is not None and alpha_km1 > 0:
        lam = alpha_k / alpha_km1
        V = tangent_project(Xk, Xk - Xkm1)
        Xk_ex = polar_retract(Xk + lam * V)
    else:
        Xk_ex = Xk

    # Cayley predictor Y0 ≈ Q_{Xk_ex} Xk = (I - c A(Xk_ex))(I + c A(Xk_ex))^{-1} Xk
    cache = build_cayley_cache(problem, Xk_ex, alpha_k)
    R = Xk.copy()
    W = apply_Linv(R, cache)                    # W = (I + c A)^{-1} Xk
    U = problem.apply_Q(Xk_ex)
    V = Xk_ex
    c = 0.5 * alpha_k
    Y = W - c * (U @ (V.T @ W) - V @ (U.T @ W))  # (I - c A) W
    return polar_retract(Y)

# ---------- main ICS (Newton–Krylov with Cayley preconditioning) ----------
def ICS_gravidy_fast(problem, X0, alpha0=1.0, tol_grad=1e-8,
                     max_outer=200, max_inner=5, gmres_maxit=200,
                     use_spectral_init=True, verbose=True):
    """
    Implicit Cayley–Stiefel (ICS): trapezoidal step + NK (GMRES)
    - Predictor: symmetric Cayley (feasible)
    - Corrector: inexact Newton with left-preconditioned GMRES
      preconditioner L^{-1} = (I + (α/2)A(Y))^{-1} via Woodbury
    """
    n, p = X0.shape
    # Initialize feasibly
    X = spectral_init_avgQ(problem.Q_list, p) if use_spectral_init else polar_retract(X0)
    alpha = alpha0

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
            print(f"[ICS] it={k:3d} f={fval:.6e} ||grad||={gnorm:.2e} feas={feas:.2e} α={alpha:.2e} t={elapsed:.2f}s")
        if gnorm <= tol_grad:
            break

        # predictor (feasible)
        Y = symmetric_cayley_predictor(problem, X, X_prev, alpha, alpha_prev)
        X_prev, alpha_prev = X.copy(), alpha

        # corrector: inexact Newton with left-preconditioned GMRES
        for it in range(max_inner):
            R = -F_of_Y(problem, Y, X, alpha)
            Rn = np.linalg.norm(R, 'fro')
            if Rn <= 1e-12 * (1.0 + np.linalg.norm(X, 'fro')):
                break

            # build preconditioner at current Y once
            cay_cache = build_cayley_cache(problem, Y, alpha)

            # define matrix-free linear operator: L^{-1} JF
            def mv(hvec):
                H = hvec.reshape(n, p, order='F')
                JH = JF_action(problem, Y, H, X, alpha)
                Z = apply_Linv(JH, cay_cache)
                return Z.reshape(-1, order='F')

            # left-preconditioned RHS: L^{-1} R
            b = apply_Linv(R, cay_cache).reshape(-1, order='F')

            Aop = LinearOperator(shape=(n*p, n*p), matvec=mv, dtype=float)
            # Eisenstat–Walker forcing
            gmres_tol = min(1e-1, max(1e-6, 0.1*Rn))
            hvec, info = gmres(Aop, b, rtol=gmres_tol, restart=None, maxiter=gmres_maxit)
            H = hvec.reshape(n, p, order='F')

            # feasibility-preserving correction
            Y = polar_retract(Y + H)

        # accept + stepsize adapt
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
    # diagonal-ish Q's (SPD)
    Q_list = [la.diagsvd(np.linspace(0.1, 10.0, n), n, n) for _ in range(p)]
    prob = StiefelQuad(Q_list)
    X0 = np.random.randn(n, p)
    X0 = polar_retract(X0)
    X, hist = ICS_gravidy_fast(prob, X0, alpha0=1.0, tol_grad=1e-6, verbose=True)
    print("final f:", prob.f(X))
