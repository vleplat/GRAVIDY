import numpy as np
import time
import scipy.linalg

def sym(M): return 0.5*(M+M.T)
def skew(M): return 0.5*(M-M.T)

# ========== Smart starting point helpers ==========

def polar_retract(X, Z=None):
    """
    X in St(n,p), Z tangent at X (optional); returns Retr_X(Z) via polar(X+Z)
    If Z is None, just orthonormalize X itself.
    """
    if Z is None:
        Y = X
    else:
        Y = X + Z
    # thin polar: Y = U S V^T, return U V^T (via SVD on p×p normal eq)
    U, _, Vt = np.linalg.svd(Y, full_matrices=False)
    return U @ Vt

def tangent_project(X, V):
    """Project V to T_X St: V - X sym(X^T V)"""
    return V - X @ sym(X.T @ V)

def spectral_init_avgQ(Q_list, p):
    """Spectral initialization using average of Q matrices"""
    n = Q_list[0].shape[0]
    Qbar = sum(Q_list)/len(Q_list)
    # smallest p eigenvectors (dense fallback)
    w, V = np.linalg.eigh(Qbar)
    idx = np.argsort(w)[:p]
    X0 = V[:, idx]
    # ensure orthonormal (should already be)
    U, _, Vt = np.linalg.svd(X0, full_matrices=False)
    return U @ Vt

def cayley_left_solve(problem, A_left_at, RHS, alpha):
    """
    Solve (I + (α/2) A(Y)) Y = RHS using Woodbury (rank-2p).
    A_left_at: matrix to build A() on the left (Y-hat for symmetric predictor).
    """
    # Build Woodbury factors at A(Y_hat)
    Yh = A_left_at
    U = problem.apply_Q(Yh)              # n×p
    V = Yh                               # n×p
    W = np.concatenate([U, V], axis=1)   # n×2p
    Z = np.concatenate([V, U], axis=1)   # n×2p
    c = 0.5*alpha
    # S^{-1} = diag( (1/c)I, -(1/c)I )
    p_ = V.shape[1]
    Sinv = np.block([[ (1.0/c)*np.eye(p_),          np.zeros((p_,p_)) ],
                     [ np.zeros((p_,p_)),          -(1.0/c)*np.eye(p_) ]])
    M = Sinv + Z.T @ W
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        Minv = np.linalg.pinv(M)
    # Apply (I - W Minv Z^T) to each RHS column
    return RHS - W @ (Minv @ (Z.T @ RHS))

def cayley_apply_QX(problem, Yhat, X, alpha):
    """
    Apply Q_{Yhat} X where Q_{Yhat} = (I - c A(Yhat)) (I + c A(Yhat))^{-1}, c = alpha/2.
    Implementation: W = (I + c A)^{-1} X ;  return (I - c A) W.
    Uses rank-2p Woodbury (matrix-free).
    """
    c = 0.5 * alpha
    U = problem.apply_Q(Yhat)            # n×p
    V = Yhat                             # n×p
    Wf = np.concatenate([U, V], axis=1)  # n×2p
    Zf = np.concatenate([V, U], axis=1)  # n×2p

    # Solve (I + c A) W = X  via Woodbury: (I + Wf Sf Zf^T)
    # with Sf = diag(+c I_p, -c I_p) and A = U V^T - V U^T.
    p = V.shape[1]
    Sf = np.block([[ c*np.eye(p),            np.zeros((p,p)) ],
                   [ np.zeros((p,p)),       -c*np.eye(p)     ]])
    # (I + W S Z^T)^{-1} X = X - W (I + S Z^T W)^{-1} S (Z^T X)
    M = np.eye(2*p) + Sf @ (Zf.T @ Wf)
    rhs_small = Sf @ (Zf.T @ X)
    try:
        Minv_rhs = np.linalg.solve(M, rhs_small)
    except np.linalg.LinAlgError:
        Minv_rhs = np.linalg.lstsq(M, rhs_small, rcond=None)[0]
    W = X - Wf @ Minv_rhs

    # Now Y = (I - c A) W = W - c*(U V^T W - V U^T W)
    VW = V.T @ W
    UW = U.T @ W
    Y = W - c * ( U @ (VW) - V @ (UW) )
    # tiny orthonormalization (numerical hygiene)
    return polar_retract(Y)

def symmetric_cayley_predictor(problem, Xk, Xkm1, alpha_k, alpha_km1):
    """
    BDF2-like extrapolation (optional) + symmetric Cayley predictor.
    Always returns Y0 in St(n,p).
    """
    # Extrapolate on-manifold (optional)
    if Xkm1 is not None and alpha_km1 is not None and alpha_km1 > 0:
        lam = alpha_k / alpha_km1
        V = tangent_project(Xk, Xk - Xkm1)
        Xk_ex = polar_retract(Xk + lam * V)
    else:
        Xk_ex = Xk

    # Symmetric Cayley predictor (uses the same A(.) on both sides)
    Y0 = cayley_apply_QX(problem, Yhat=Xk_ex, X=Xk, alpha=alpha_k)
    return Y0

def feasible_newton_step(Yi, Hi):
    """Enforce feasibility after Newton correction"""
    return polar_retract(Yi + Hi)

class StiefelQuad:
    # Phi(X) = 1/2 sum_j x_j^T Q_j x_j ; Riemannian grad = G - X sym(X^T G)
    def __init__(self, Q_list):
        self.Q_list = Q_list
        self.n = Q_list[0].shape[0]
        self.p = len(Q_list)

    def f(self, X):
        return 0.5*sum((X[:,j].T @ (self.Q_list[j] @ X[:,j])) for j in range(self.p))

    def apply_Q(self, X):
        G = np.empty_like(X)
        for j,Q in enumerate(self.Q_list): G[:,j] = Q @ X[:,j]
        return G

    def grad_riem(self, X):
        G = self.apply_Q(X)
        return G - X @ sym(X.T @ G)

    def A_skew(self, X):
        G = self.apply_Q(X)
        return G @ X.T - X @ G.T   # skew

# ---------- improved GMRES with LU preconditioner ----------
def _gmres(Aop, b, tol=1e-8, maxiter=200):
    n = b.size
    beta = np.linalg.norm(b)
    if beta == 0: return np.zeros_like(b), 0
    V = np.zeros((n, maxiter+1)); H = np.zeros((maxiter+1, maxiter))
    V[:,0] = b / beta
    e1 = np.zeros(maxiter+1); e1[0] = beta
    
    for j in range(maxiter):
        w = Aop(V[:,j])
        # Modified Gram-Schmidt with reorthogonalization
        for i in range(j+1):
            H[i,j] = np.dot(V[:,i], w)
            w -= H[i,j] * V[:,i]
        # Reorthogonalization for better stability
        for i in range(j+1):
            h_ij = np.dot(V[:,i], w)
            H[i,j] += h_ij
            w -= h_ij * V[:,i]
        
        H[j+1,j] = np.linalg.norm(w)
        if H[j+1,j] < 1e-12:  # Early termination if breakdown
            break
        V[:,j+1] = w / H[j+1,j]
        
        # Solve least squares problem more efficiently
        Hj = H[:j+2,:j+1]
        try:
            y = np.linalg.solve(Hj.T @ Hj, Hj.T @ e1[:j+2])
        except np.linalg.LinAlgError:
            y, *_ = np.linalg.lstsq(Hj, e1[:j+2], rcond=None)
        
        rnorm = np.linalg.norm(e1[:j+2] - Hj @ y)
        if rnorm <= tol * beta:
            return V[:,:j+1] @ y, 0
    
    # Final solve if maxiter reached
    Hj = H[:maxiter+1,:maxiter]
    try:
        y = np.linalg.solve(Hj.T @ Hj, Hj.T @ e1[:maxiter+1])
    except np.linalg.LinAlgError:
        y, *_ = np.linalg.lstsq(Hj, e1[:maxiter+1], rcond=None)
    return V[:,:maxiter] @ y, maxiter

# ---------- Woodbury apply for L^{-1} = (I + (α/2) A(Y))^{-1} ----------
def _Linv_apply(R, problem, Y, alpha, cache=None):
    # Build once per Newton iterate
    if cache is None:
        n,p = problem.n, problem.p
        U = problem.apply_Q(Y)              # n×p
        V = Y                               # n×p
        W = np.concatenate([U, V], axis=1)  # n×2p
        Z = np.concatenate([V, U], axis=1)  # n×2p
        c = 0.5*alpha
        # S^{-1} = diag( (1/c)I, -(1/c)I )
        Sinv = np.block([[ (1.0/c)*np.eye(p),             np.zeros((p,p)) ],
                         [ np.zeros((p,p)),               -(1.0/c)*np.eye(p) ]])
        M = Sinv + Z.T @ W                  # 2p×2p
        try:
            # Use LU factorization for better stability and speed
            P, L_lu, U = scipy.linalg.lu_factor(M)
            cache = (W, Z, (P, L_lu, U), 'lu')
        except:
            # Fallback to direct inverse
            try:
                Minv = np.linalg.inv(M)
                cache = (W, Z, Minv, 'inv')
            except np.linalg.LinAlgError:
                Minv = np.linalg.pinv(M)
                cache = (W, Z, Minv, 'pinv')
    
    # Extract cache components
    W, Z, Minv_data, method = cache

    # Apply: (I - W Minv Z^T) R
    if method == 'lu':
        P, L_lu, U = Minv_data
        temp = scipy.linalg.lu_solve((P, L_lu, U), Z.T @ R)
    elif method == 'inv':
        temp = Minv_data @ (Z.T @ R)
    else:  # method == 'pinv'
        temp = Minv_data @ (Z.T @ R)
    
    return R - W @ temp, cache

# ---------- Residual F and Jacobian action ----------
def _F_of_Y(problem, Y, Xk, alpha):
    A = problem.A_skew(Y); I = np.eye(problem.n)
    return (I + 0.5*alpha*A) @ Y - (I - 0.5*alpha*A) @ Xk

def _JF_action(problem, Y, H, Xk, alpha):
    n = problem.n
    A_Y  = problem.A_skew(Y)
    G_Xk = problem.apply_Q(Xk)
    G_H  = problem.apply_Q(H)
    W = (G_Xk @ H.T - H @ G_Xk.T) + (G_H @ Xk.T - Xk @ G_H.T)
    return (np.eye(n) + 0.5*alpha*A_Y) @ H + 0.5*alpha * (W @ (Y + Xk))

# ---------- cheap Wen–Yin predictor via Cayley (also Woodbury-accelerated) ----------
def _cayley_predictor(problem, Xk, alpha):
    n = problem.n
    A = problem.A_skew(Xk)                  # n×n skew (rank ≤ 2p)
    # L = I + (α/2)A, R = (I - (α/2)A) Xk ⇒ solve L Y = R
    I = np.eye(n); c = 0.5*alpha
    R = (I - c*A) @ Xk
    # Woodbury apply to each RHS
    Y = R.copy()
    Y, _ = _Linv_apply(R, problem, Xk, alpha, cache=None)
    return Y

# ---------- FAST ICS: NK + Woodbury + predictor + inexact Newton ----------
def ICS_gravidy_fast(problem, X0,
                     alpha0=1.0,
                     tol_grad=1e-8,
                     max_outer=200,
                     max_inner=5,
                     gmres_maxit=100,
                     use_smart_init=True,
                     verbose=True):
    n,p = X0.shape
    
    # Layer A: Smart initialization
    if use_smart_init:
        # Use spectral initialization if Q_list is available
        if hasattr(problem, 'Q_list'):
            X = spectral_init_avgQ(problem.Q_list, p)
        else:
            X = X0.copy()
    else:
        X = X0.copy()
    
    alpha = alpha0
    hist = []
    t0 = time.time()
    
    # Track previous iteration for BDF2 predictor
    prev_X = None
    prev_alpha = None

    for k in range(max_outer):
        g = problem.grad_riem(X)
        gnorm = np.linalg.norm(g, 'fro')
        feas = np.linalg.norm(X.T @ X - np.eye(p), 'fro')
        fval = problem.f(X)
        current_time = time.time() - t0
        hist.append((k, fval, gnorm, feas, current_time))
        if verbose:
            print(f"[ICS-fast] it={k:3d} f={fval:.6e} ||grad||={gnorm:.2e} feas={feas:.2e} α={alpha:.2e} time={current_time:.2f}s")
        if gnorm <= tol_grad:
            break

        # Layer B: Symmetric Cayley predictor (always feasible)
        Y = symmetric_cayley_predictor(problem, X, prev_X, alpha, prev_alpha)
        
        # Store for next iteration
        prev_X = X.copy()
        prev_alpha = alpha

        # ---- corrector: inexact Newton with Woodbury-precond GMRES ----
        for it in range(max_inner):
            R = -_F_of_Y(problem, Y, X, alpha)
            Rn = np.linalg.norm(R, 'fro')
            # Eisenstat–Walker forcing (looser early, tighter later)
            eta = min(1e-1, max(1e-4, 0.1*Rn))
            # Build preconditioner L^{-1} at current Y (Woodbury)
            Linv_cache = None
            def Aop(hvec):
                H = hvec.reshape(n, p, order='F')
                JH = _JF_action(problem, Y, H, X, alpha)
                Z, _ = _Linv_apply(JH, problem, Y, alpha, cache=Linv_cache)
                return Z.reshape(-1, order='F')
            b, Linv_cache = _Linv_apply(R, problem, Y, alpha, cache=Linv_cache)
            b = b.reshape(-1, order='F')
            hvec, _ = _gmres(Aop, b, tol=eta, maxiter=gmres_maxit)
            H = hvec.reshape(n, p, order='F')
            Y = feasible_newton_step(Y, H)  # Retraction after Newton step
            if Rn <= 1e-10 * (1.0 + np.linalg.norm(X, 'fro')):
                break

        # ---- accept / adapt α (simple Armijo-like rule) ----
        f_new = problem.f(Y)
        if f_new <= fval - 1e-4 * alpha * (gnorm**2):
            X = Y
            alpha = min(alpha*1.5, 1e8)
        else:
            alpha = max(alpha*0.5, 1e-8)
            # retry next outer with smaller step

    return X, hist
