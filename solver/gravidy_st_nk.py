import numpy as np
import time
import scipy.linalg

def sym(M): return 0.5*(M+M.T)
def skew(M): return 0.5*(M-M.T)

class StiefelQuad:
    """
    Phi(X) = 0.5 * sum_j x_j^T Q_j x_j
    grad_R(X) = G - X sym(X^T G),  G = [Q_1 x_1, ..., Q_p x_p]
    A(Y) = G(Y) Y^T - Y G(Y)^T
    """
    def __init__(self, Q_list):
        self.Q_list = Q_list
        self.n = Q_list[0].shape[0]
        self.p = len(Q_list)

    def f(self, X):
        s = 0.0
        for j in range(self.p):
            xj = X[:, j]
            s += 0.5 * float(xj.T @ (self.Q_list[j] @ xj))
        return s

    def apply_Q(self, X):
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

def ICS_gravidy_NK(problem, X0, alpha0=1.0,
                   tol_grad=1e-8, newton_tol=1e-10,
                   max_outer=200, max_inner=10,
                   gmres_tol=1e-8, gmres_maxit=200,
                   armijo_c1=1e-4, grow=1.5, backtrack=0.5,
                   alpha_min=1e-8, alpha_max=1e8, verbose=True):
    """
    Same residual F, exact JF action, but solve JΔ = -F with GMRES (matrix-free).
    """
    n, p = X0.shape
    X = X0.copy()
    alpha = alpha0

    def vecF(M): return M.reshape(-1, order='F')
    def unvecF(v): return v.reshape((n, p), order='F')

    def F_of_Y(Y, Xk, a):
        A = problem.A_skew(Y)
        I = np.eye(n)
        return (I + 0.5*a*A) @ Y - (I - 0.5*a*A) @ Xk

    def JF_action(Y, H, Xk, a):
        A_Y  = problem.A_skew(Y)
        G_Xk = problem.apply_Q(Xk)
        G_H  = problem.apply_Q(H)
        W = (G_Xk @ H.T - H @ G_Xk.T) + (G_H @ Xk.T - Xk @ G_H.T)
        return (np.eye(n) + 0.5*a*A_Y) @ H + 0.5*a * (W @ (Y + Xk))

    # Improved GMRES with better numerical stability
    def gmres_matfree(Aop, b, tol=1e-8, maxiter=200):
        N = b.size
        beta = np.linalg.norm(b)
        if beta == 0.0:
            return np.zeros_like(b), 0
        V = np.zeros((N, maxiter+1))
        H = np.zeros((maxiter+1, maxiter))
        V[:, 0] = b / beta
        e1 = np.zeros(maxiter+1); e1[0] = beta
        
        for j in range(maxiter):
            w = Aop(V[:, j])
            # Modified Gram-Schmidt with reorthogonalization
            for i in range(j+1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]
            # Reorthogonalization for better stability
            for i in range(j+1):
                h_ij = np.dot(V[:, i], w)
                H[i, j] += h_ij
                w -= h_ij * V[:, i]
            
            H[j+1, j] = np.linalg.norm(w)
            if H[j+1, j] < 1e-12:  # Early termination if breakdown
                break
            V[:, j+1] = w / H[j+1, j]
            
            # Solve least squares problem more efficiently
            Hj = H[:j+2, :j+1]
            try:
                y = np.linalg.solve(Hj.T @ Hj, Hj.T @ e1[:j+2])
            except np.linalg.LinAlgError:
                y, *_ = np.linalg.lstsq(Hj, e1[:j+2], rcond=None)
            
            rnorm = np.linalg.norm(e1[:j+2] - Hj @ y)
            if rnorm <= tol * beta:
                return V[:, :j+1] @ y, 0
        
        # Final solve if maxiter reached
        Hj = H[:maxiter+1, :maxiter]
        try:
            y = np.linalg.solve(Hj.T @ Hj, Hj.T @ e1[:maxiter+1])
        except np.linalg.LinAlgError:
            y, *_ = np.linalg.lstsq(Hj, e1[:maxiter+1], rcond=None)
        return V[:, :maxiter] @ y, maxiter

    history = []
    t0 = time.time()
    for k in range(max_outer):
        g = problem.grad_riem(X)
        gnorm = np.linalg.norm(g, 'fro')
        feas = np.linalg.norm(X.T @ X - np.eye(p), 'fro')
        fval = problem.f(X)
        current_time = time.time() - t0
        history.append((k, fval, gnorm, feas, current_time))
        if verbose:
            print(f"[ICS-NK] it={k:3d} f={fval:.6e} ||grad||={gnorm:.2e} feas={feas:.2e} alpha={alpha:.2e}")
        if gnorm <= tol_grad:
            break

        Y = X.copy()
        for it in range(max_inner):
            R = -F_of_Y(Y, X, alpha)
            if np.linalg.norm(R, 'fro') <= newton_tol * (1.0 + np.linalg.norm(X, 'fro')):
                break
                
            # Better preconditioner: block-diagonal approximation
            # For each column, solve (I + a/2 A_ii) h_i = r_i independently
            A_Y = problem.A_skew(Y)
            L = np.eye(problem.n) + 0.5*alpha*A_Y
            
            # Use LU decomposition for better stability and speed
            try:
                P, L_lu, U = scipy.linalg.lu_factor(L)
                def precond_solve(B):
                    return scipy.linalg.lu_solve((P, L_lu, U), B)
            except:
                # Fallback to direct inverse
                try:
                    L_inv = np.linalg.inv(L)
                    def precond_solve(B):
                        return L_inv @ B
                except:
                    L_inv = np.linalg.inv(L + 1e-12*np.eye(problem.n))
                    def precond_solve(B):
                        return L_inv @ B

            def K_mv(h_vec):
                H = unvecF(h_vec)
                JH = JF_action(Y, H, X, alpha)
                Z = precond_solve(JH)
                return vecF(Z)

            b = vecF(precond_solve(R))
            h, _ = gmres_matfree(K_mv, b, tol=gmres_tol, maxiter=gmres_maxit)
            Y = Y + unvecF(h)

        X_trial = Y
        f_new = problem.f(X_trial)
        if f_new <= fval - 1e-4 * alpha * (gnorm ** 2):
            X = X_trial
            alpha = min(alpha * grow, alpha_max)
        else:
            alpha = max(alpha * backtrack, alpha_min)
            continue

    return X, history
