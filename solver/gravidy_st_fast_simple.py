import numpy as np
import time
import scipy.linalg

def sym(M): return 0.5*(M+M.T)
def skew(M): return 0.5*(M-M.T)

# ========== Simple but effective optimizations ==========

def polar_retract_fast(X, Z=None):
    """Fast polar retraction using optimized SVD"""
    if Z is None:
        Y = X
    else:
        Y = X + Z
    # Use overwrite_a=True for speed (we don't need Y anymore)
    U, _, Vt = scipy.linalg.svd(Y, full_matrices=False, overwrite_a=True, lapack_driver='gesdd')
    return U @ Vt

def tangent_project(X, V):
    """Project V to T_X St: V - X sym(X^T V)"""
    return V - X @ sym(X.T @ V)

def spectral_init_avgQ_fast(Q_list, p):
    """Fast spectral initialization - just use numpy's optimized eigh"""
    n = Q_list[0].shape[0]
    Qbar = sum(Q_list)/len(Q_list)
    
    # Use numpy's optimized eigenvalue solver
    w, V = np.linalg.eigh(Qbar)
    idx = np.argsort(w)[:p]
    X0 = V[:, idx]
    
    # Fast orthogonalization
    return polar_retract_fast(X0)

def cayley_apply_QX_fast(problem, Yhat, X, alpha):
    """
    Fast Cayley application using direct dense linear algebra
    """
    c = 0.5 * alpha
    U = problem.apply_Q(Yhat)            # n×p
    V = Yhat                             # n×p
    
    # Build 2p×2p system directly (small and dense - perfect for direct solve)
    p = V.shape[1]
    
    # Build Woodbury components more efficiently
    W = np.column_stack([U, V])  # n×2p  
    Z = np.column_stack([V, U])  # n×2p
    
    # Build small coefficient matrix
    S = np.block([[ c*np.eye(p),            np.zeros((p,p)) ],
                  [ np.zeros((p,p)),       -c*np.eye(p)     ]])
    
    # Use scipy's optimized operations
    ZtW = Z.T @ W  # 2p×2p
    M = np.eye(2*p) + S @ ZtW
    rhs = S @ (Z.T @ X)
    
    # Direct solve (much faster than iterative for small dense systems)
    try:
        temp = scipy.linalg.solve(M, rhs, assume_a='gen', overwrite_a=True, overwrite_b=True)
    except:
        temp = np.linalg.solve(M, rhs)
    
    # Apply Woodbury formula
    Y_temp = X - W @ temp
    
    # Apply the (I - cA) part directly
    VW = V.T @ Y_temp
    UW = U.T @ Y_temp
    Y = Y_temp - c * (U @ VW - V @ UW)
    
    # Fast orthogonalization
    return polar_retract_fast(Y)

def symmetric_cayley_predictor_fast(problem, Xk, Xkm1, alpha_k, alpha_km1):
    """Fast symmetric Cayley predictor"""
    # Simple extrapolation (avoid expensive operations when not needed)
    if Xkm1 is not None and alpha_km1 is not None and alpha_km1 > 0:
        lam = alpha_k / alpha_km1
        V = tangent_project(Xk, Xk - Xkm1)
        Xk_ex = polar_retract_fast(Xk, lam * V)
    else:
        Xk_ex = Xk

    # Fast symmetric Cayley predictor
    return cayley_apply_QX_fast(problem, Yhat=Xk_ex, X=Xk, alpha=alpha_k)

class StiefelQuad:
    """Same StiefelQuad class"""
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
        return G @ X.T - X @ G.T

# ========== Fast Woodbury operations ==========

def _Linv_apply_fast(R, problem, Y, alpha, cache=None):
    """Fast Woodbury preconditioner using direct dense solve"""
    if cache is None:
        n,p = problem.n, problem.p
        U = problem.apply_Q(Y)              # n×p
        V = Y                               # n×p
        W = np.column_stack([U, V])         # n×2p
        Z = np.column_stack([V, U])         # n×2p
        c = 0.5*alpha
        
        # Build small dense system (2p×2p)
        Sinv = np.block([[ (1.0/c)*np.eye(p),             np.zeros((p,p)) ],
                         [ np.zeros((p,p)),               -(1.0/c)*np.eye(p) ]])
        M = Sinv + Z.T @ W
        
        # Use scipy's fast LU factorization
        try:
            LU = scipy.linalg.lu_factor(M, overwrite_a=True)
            cache = (W, Z, LU, 'lu')
        except:
            # Fallback to direct inverse for small matrices
            try:
                Minv = scipy.linalg.inv(M)
                cache = (W, Z, Minv, 'inv')
            except:
                Minv = np.linalg.pinv(M)
                cache = (W, Z, Minv, 'pinv')
    
    # Extract cache
    W, Z, solve_data, method = cache

    # Fast application
    ZtR = Z.T @ R
    if method == 'lu':
        temp = scipy.linalg.lu_solve(solve_data, ZtR)
    else:  # 'inv' or 'pinv'
        temp = solve_data @ ZtR
    
    return R - W @ temp, cache

# ========== Core functions (unchanged) ==========

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

def _simple_gmres_solve(problem, Y, X, alpha, newton_tol):
    """
    Simple GMRES solve - but much simpler than the original
    """
    R = -_F_of_Y(problem, Y, X, alpha)
    Rn = np.linalg.norm(R, 'fro')
    if Rn <= newton_tol * (1.0 + np.linalg.norm(X, 'fro')):
        return Y, True  # Already converged
    
    n, p = Y.shape
    
    # Use the Woodbury preconditioner from the original Fast ICS
    cache = None
    
    def Aop(hvec):
        H = hvec.reshape(n, p, order='F')
        JH = _JF_action(problem, Y, H, X, alpha)
        Z, _ = _Linv_apply_fast(JH, problem, Y, alpha, cache)
        return Z.reshape(-1, order='F')
    
    b, cache = _Linv_apply_fast(R, problem, Y, alpha, cache)
    b = b.reshape(-1, order='F')
    
    # Simple GMRES (max 10 iterations, very loose tolerance)
    h, _ = _simple_gmres(Aop, b, tol=1e-6, maxiter=10)
    
    return polar_retract_fast(Y + h.reshape((n, p), order='F')), True

def _simple_gmres(Aop, b, tol=1e-6, maxiter=10):
    """
    Very simple GMRES - just 10 iterations max
    """
    n = b.size
    beta = np.linalg.norm(b)
    if beta == 0:
        return np.zeros_like(b), 0
    
    V = np.zeros((n, maxiter+1))
    H = np.zeros((maxiter+1, maxiter))
    V[:, 0] = b / beta
    e1 = np.zeros(maxiter+1)
    e1[0] = beta
    
    for j in range(maxiter):
        w = Aop(V[:, j])
        for i in range(j+1):
            H[i, j] = V[:, i].dot(w)
            w -= H[i, j] * V[:, i]
        H[j+1, j] = np.linalg.norm(w)
        if H[j+1, j] != 0:
            V[:, j+1] = w / H[j+1, j]
        
        # Solve least squares problem
        Hj = H[:j+2, :j+1]
        y, *_ = np.linalg.lstsq(Hj, e1[:j+2], rcond=None)
        r = e1[:j+2] - Hj @ y
        if np.linalg.norm(r) <= tol * beta:
            return V[:, :j+1] @ y, 0
    
    # Max iterations reached
    Hj = H[:maxiter+1, :maxiter]
    y, *_ = np.linalg.lstsq(Hj, e1[:maxiter+1], rcond=None)
    return V[:, :maxiter] @ y, maxiter

# ========== Main fast solver ==========

def ICS_gravidy_fast_simple(problem, X0,
                            alpha0=1.0,
                            tol_grad=1e-8,
                            max_outer=200,
                            newton_tol=1e-10,
                            use_smart_init=True,
                            verbose=True):
    """
    Simple and fast ICS solver - NO GMRES, just direct linear algebra
    """
    n,p = X0.shape
    
    # Fast initialization
    if use_smart_init and hasattr(problem, 'Q_list'):
        X = spectral_init_avgQ_fast(problem.Q_list, p)
    else:
        X = X0.copy()
    
    alpha = alpha0
    hist = []
    t0 = time.time()
    
    # Track previous iteration
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
            print(f"[ICS-simple] it={k:3d} f={fval:.6e} ||grad||={gnorm:.2e} feas={feas:.2e} α={alpha:.2e} time={current_time:.2f}s")
        if gnorm <= tol_grad:
            break

        # Fast predictor
        Y = symmetric_cayley_predictor_fast(problem, X, prev_X, alpha, prev_alpha)
        
        # Store for next iteration
        prev_X = X.copy()
        prev_alpha = alpha

        # Simple GMRES solve (limited iterations)
        Y, success = _simple_gmres_solve(problem, Y, X, alpha, newton_tol)
        
        # Accept/reject step with more conservative criterion
        f_new = problem.f(Y)
        if f_new <= fval - 1e-4 * alpha * (gnorm**2):
            X = Y
            alpha = min(alpha*1.2, 1e8)  # More conservative growth
        else:
            alpha = max(alpha*0.7, 1e-8)  # Less aggressive reduction

    return X, hist
