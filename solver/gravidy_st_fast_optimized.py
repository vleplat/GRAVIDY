import numpy as np
import time
import scipy.linalg
from scipy.sparse.linalg import eigsh, gmres, LinearOperator
from scipy.linalg import solve, svd

def sym(M): return 0.5*(M+M.T)
def skew(M): return 0.5*(M-M.T)

# ========== Optimized helpers ==========

def polar_retract_optimized(X, Z=None):
    """
    Optimized polar retraction using SciPy's faster SVD
    """
    if Z is None:
        Y = X
    else:
        Y = X + Z
    # Use SciPy's optimized SVD with gesdd driver (faster for large matrices)
    U, _, Vt = svd(Y, full_matrices=False, lapack_driver='gesdd', overwrite_a=True)
    return U @ Vt

def tangent_project(X, V):
    """Project V to T_X St: V - X sym(X^T V)"""
    return V - X @ sym(X.T @ V)

def spectral_init_avgQ_optimized(Q_list, p):
    """Optimized spectral initialization using sparse eigensolvers only for large problems"""
    n = Q_list[0].shape[0]
    Qbar = sum(Q_list)/len(Q_list)
    
    # Use sparse eigensolver only for very large matrices where it's actually faster
    if n > 500 and p < n//20:
        try:
            # Use sparse eigenvalue solver for smallest eigenvalues
            w, V = eigsh(Qbar, k=p, which='SM', maxiter=min(500, 5*n))
        except:
            # Fallback to dense solver if sparse fails
            w, V = np.linalg.eigh(Qbar)
            idx = np.argsort(w)[:p]
            V = V[:, idx]
    else:
        # Use dense solver for smaller matrices (faster for most cases)
        w, V = np.linalg.eigh(Qbar)
        idx = np.argsort(w)[:p]
        V = V[:, idx]
    
    # Use simple SVD orthogonalization (faster than polar_retract for small p)
    U, _, Vt = np.linalg.svd(V, full_matrices=False)
    return U @ Vt

def cayley_apply_QX_optimized(problem, Yhat, X, alpha):
    """
    Optimized Cayley application with better numerical methods
    """
    c = 0.5 * alpha
    U = problem.apply_Q(Yhat)            # n×p
    V = Yhat                             # n×p
    Wf = np.concatenate([U, V], axis=1)  # n×2p
    Zf = np.concatenate([V, U], axis=1)  # n×2p

    # Build coefficient matrix with optimized operations
    p = V.shape[1]
    Sf = np.block([[ c*np.eye(p),            np.zeros((p,p)) ],
                   [ np.zeros((p,p)),       -c*np.eye(p)     ]])
    
    # Use optimized matrix operations
    ZtW = Zf.T @ Wf  # Cache this computation
    M = np.eye(2*p) + Sf @ ZtW
    rhs_small = Sf @ (Zf.T @ X)
    
    # Use SciPy's optimized solve with symmetry assumption
    try:
        Minv_rhs = solve(M, rhs_small, assume_a='gen', overwrite_a=False, overwrite_b=False)
    except np.linalg.LinAlgError:
        Minv_rhs = np.linalg.lstsq(M, rhs_small, rcond=None)[0]
    
    W = X - Wf @ Minv_rhs

    # Optimized skew-symmetric operations
    VW = V.T @ W
    UW = U.T @ W
    Y = W - c * (U @ VW - V @ UW)
    
    # Optimized orthonormalization
    return polar_retract_optimized(Y)

def symmetric_cayley_predictor_optimized(problem, Xk, Xkm1, alpha_k, alpha_km1):
    """
    Optimized symmetric Cayley predictor
    """
    # Extrapolate on-manifold (optional)
    if Xkm1 is not None and alpha_km1 is not None and alpha_km1 > 0:
        lam = alpha_k / alpha_km1
        V = tangent_project(Xk, Xk - Xkm1)
        Xk_ex = polar_retract_optimized(Xk, lam * V)
    else:
        Xk_ex = Xk

    # Symmetric Cayley predictor
    Y0 = cayley_apply_QX_optimized(problem, Yhat=Xk_ex, X=Xk, alpha=alpha_k)
    return Y0

def feasible_newton_step_optimized(Yi, Hi):
    """Optimized feasibility enforcement after Newton correction"""
    return polar_retract_optimized(Yi + Hi)

class StiefelQuad:
    """Same StiefelQuad class as before"""
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

# ========== Optimized Woodbury operations ==========

def _Linv_apply_optimized(R, problem, Y, alpha, cache=None):
    """Optimized Woodbury preconditioner application"""
    if cache is None:
        n,p = problem.n, problem.p
        U = problem.apply_Q(Y)              # n×p
        V = Y                               # n×p
        W = np.concatenate([U, V], axis=1)  # n×2p
        Z = np.concatenate([V, U], axis=1)  # n×2p
        c = 0.5*alpha
        
        # Optimized coefficient matrix assembly
        Sinv = np.block([[ (1.0/c)*np.eye(p),             np.zeros((p,p)) ],
                         [ np.zeros((p,p)),               -(1.0/c)*np.eye(p) ]])
        M = Sinv + Z.T @ W                  # 2p×2p
        
        # Use optimized LU factorization
        try:
            P, L_lu, U_lu = scipy.linalg.lu_factor(M, overwrite_a=True)
            cache = (W, Z, (P, L_lu, U_lu), 'lu')
        except:
            try:
                # Fallback to optimized Cholesky if symmetric positive definite
                L_chol = scipy.linalg.cholesky(M, lower=True)
                cache = (W, Z, L_chol, 'chol')
            except:
                # Final fallback to direct inverse
                try:
                    Minv = solve(M, np.eye(M.shape[0]), assume_a='gen')
                    cache = (W, Z, Minv, 'inv')
                except:
                    Minv = np.linalg.pinv(M)
                    cache = (W, Z, Minv, 'pinv')
    
    # Extract cache components
    W, Z, Minv_data, method = cache

    # Apply optimized solve based on factorization type
    ZtR = Z.T @ R  # Cache this computation
    if method == 'lu':
        P, L_lu, U_lu = Minv_data
        temp = scipy.linalg.lu_solve((P, L_lu, U_lu), ZtR)
    elif method == 'chol':
        L_chol = Minv_data
        temp = scipy.linalg.solve_triangular(L_chol, ZtR, lower=True)
        temp = scipy.linalg.solve_triangular(L_chol.T, temp, lower=False)
    else:  # 'inv' or 'pinv'
        temp = Minv_data @ ZtR
    
    return R - W @ temp, cache

# ========== Core functions ==========

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

# ========== Main optimized solver ==========

def ICS_gravidy_fast_optimized(problem, X0,
                               alpha0=1.0,
                               tol_grad=1e-8,
                               max_outer=200,
                               max_inner=5,
                               gmres_tol=1e-8,
                               gmres_maxit=100,
                               use_smart_init=True,
                               verbose=True):
    """
    Optimized Fast ICS solver using SciPy's high-performance routines
    """
    n,p = X0.shape
    
    # Layer A: Optimized smart initialization
    if use_smart_init:
        # Use optimized spectral initialization if Q_list is available
        if hasattr(problem, 'Q_list'):
            X = spectral_init_avgQ_optimized(problem.Q_list, p)
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
            print(f"[ICS-fast-opt] it={k:3d} f={fval:.6e} ||grad||={gnorm:.2e} feas={feas:.2e} α={alpha:.2e} time={current_time:.2f}s")
        if gnorm <= tol_grad:
            break

        # Layer B: Optimized symmetric Cayley predictor
        Y = symmetric_cayley_predictor_optimized(problem, X, prev_X, alpha, prev_alpha)
        
        # Store for next iteration
        prev_X = X.copy()
        prev_alpha = alpha

        # ---- Layer C: Optimized inexact Newton with SciPy GMRES ----
        for it in range(max_inner):
            R = -_F_of_Y(problem, Y, X, alpha)
            Rn = np.linalg.norm(R, 'fro')
            # Eisenstat–Walker forcing (looser early, tighter later)
            eta = min(1e-1, max(1e-4, 0.1*Rn))
            
            # Build preconditioner L^{-1} at current Y (optimized Woodbury)
            Linv_cache = None
            def Aop_func(hvec):
                H = hvec.reshape(n, p, order='F')
                JH = _JF_action(problem, Y, H, X, alpha)
                Z, _ = _Linv_apply_optimized(JH, problem, Y, alpha, cache=Linv_cache)
                return Z.reshape(-1, order='F')
            
            # Wrap as LinearOperator for SciPy GMRES
            Aop = LinearOperator((n*p, n*p), matvec=Aop_func)
            
            b, Linv_cache = _Linv_apply_optimized(R, problem, Y, alpha, cache=Linv_cache)
            b = b.reshape(-1, order='F')
            
            # Use SciPy's GMRES only for large problems, otherwise use our custom GMRES
            if n*p > 1000:  # Use SciPy for large problems
                hvec, info = gmres(Aop, b, rtol=eta, maxiter=gmres_maxit, restart=min(50, gmres_maxit//2))
                if info != 0 and verbose:
                    print(f"    GMRES warning: info={info}")
            else:  # Use custom GMRES for smaller problems
                hvec, _ = _gmres_simple(Aop_func, b, tol=eta, maxiter=gmres_maxit)
            
            H = hvec.reshape(n, p, order='F')
            Y = feasible_newton_step_optimized(Y, H)  # Optimized retraction
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
