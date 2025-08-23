import numpy as np
import time

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

# ---- Helpers: vectorization (Fortran order) ----
def vecF(M): return M.reshape(-1, order='F')
def unvecF(v, n, p): return v.reshape((n, p), order='F')

# ---- F(Y) and exact Fréchet derivative JF[Y][H] ----
def F_of_Y(problem, Y, Xk, a):
    n = problem.n
    A = problem.A_skew(Y)
    I = np.eye(n)
    return (I + 0.5*a*A) @ Y - (I - 0.5*a*A) @ Xk

def JF_action(problem, Y, H, Xk, a):
    """
    JF[Y][H] = (I + a/2 A(Y)) H + (a/2) * W * (Y + Xk),
      W = (G(Xk)H^T - H G(Xk)^T) + (G(H) Xk^T - Xk G(H)^T)
    """
    n = problem.n
    A_Y  = problem.A_skew(Y)
    G_Xk = problem.apply_Q(Xk)
    G_H  = problem.apply_Q(H)
    W = (G_Xk @ H.T - H @ G_Xk.T) + (G_H @ Xk.T - Xk @ G_H.T)
    return (np.eye(n) + 0.5*a*A_Y) @ H + 0.5*a * (W @ (Y + Xk))

def build_dense_J(problem, Y, Xk, a):
    """Build dense (np x np) Jacobian by columns via JF_action on basis."""
    n, p = problem.n, problem.p
    N = n*p
    J = np.zeros((N, N))
    # basis vectors in column-major order
    for col in range(N):
        e = np.zeros(N); e[col] = 1.0
        E = unvecF(e, n, p)
        JE = JF_action(problem, Y, E, Xk, a)
        J[:, col] = vecF(JE)
    return J

def ICS_gravidy_NR_dense(problem,
                         X0,
                         alpha0=1.0,
                         tol_grad=1e-8,
                         newton_tol=1e-10,
                         max_outer=200,
                         max_inner=10,
                         armijo_c1=1e-4,
                         grow=1.5,
                         backtrack=0.5,
                         alpha_min=1e-8,
                         alpha_max=1e8,
                         verbose=True):
    """
    Implicit Cayley–Stiefel with *direct* Newton linear solve.
    Best for moderate N = n*p (say up to a few thousands).
    """
    n, p = X0.shape
    X = X0.copy()
    alpha = alpha0

    f = problem.f
    history = []
    t0 = time.time()

    for k in range(max_outer):
        g = problem.grad_riem(X)
        gnorm = np.linalg.norm(g, 'fro')
        feas = np.linalg.norm(X.T @ X - np.eye(p), 'fro')
        fval = f(X)
        current_time = time.time() - t0
        history.append((k, fval, gnorm, feas, current_time))
        if verbose:
            print(f"[ICS-NR] it={k:3d} f={fval:.6e} ||grad||={gnorm:.2e} feas={feas:.2e} alpha={alpha:.2e}")

        if gnorm <= tol_grad:
            break

        # ----- Inner Newton (exact J, direct solve) -----
        Y = X.copy()
        ok = False
        for it in range(max_inner):
            R = -F_of_Y(problem, Y, X, alpha)          # right-hand side
            rnorm = np.linalg.norm(R, 'fro')
            if rnorm <= newton_tol * (1.0 + np.linalg.norm(X, 'fro')):
                ok = True
                break

            # Build dense Jacobian J and solve J vec(H) = vec(R)
            J = build_dense_J(problem, Y, X, alpha)
            rhs = vecF(R)
            try:
                h = np.linalg.solve(J, rhs)
            except np.linalg.LinAlgError:
                # Fallback to least-squares if ill-conditioned
                h, *_ = np.linalg.lstsq(J, rhs, rcond=None)
            H = unvecF(h, n, p)
            Y = Y + H

        X_trial = Y
        f_new = f(X_trial)
        # Armijo acceptance on objective
        if f_new <= fval - armijo_c1 * alpha * (gnorm ** 2):
            X = X_trial
            alpha = min(alpha * grow, alpha_max)
        else:
            alpha = max(alpha * backtrack, alpha_min)
            # retry with smaller alpha
            continue

    return X, history
