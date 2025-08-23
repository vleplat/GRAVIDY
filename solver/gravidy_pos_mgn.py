"""
GRAVIDY–pos solver for NNLS / positive-orthant LS:
Implicit Euler in reparameterized coordinates x = g(u) with g_i(u_i) = exp(u_i) by default.

Inner solvers:
- MGN (recommended): solves (J^T J + M I) h = -J^T F (SPD, robust near active faces)
- Newton       : solves J h = -F (faster when well conditioned)

Consistent with paper's subsection:
  F(u) = u - u_k + eta * grad_x Phi(g(u))
  J(u) = I + eta * H * diag(g'(u)), H = A^T A
"""

import numpy as np
import time


# ---------- Utilities ----------
def safe_exp(u):
    """Clamp to avoid overflow/underflow; keeps positivity mapping numerically stable."""
    return np.exp(np.clip(u, -40.0, 40.0))


class PositiveLeastSquares:
    """Simple problem wrapper: min_x>=0 0.5||A x - b||^2."""
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.n = A.shape[1]
        self.H = A.T @ A  # Hessian
        self.Ab = A.T @ b

    def f(self, x):
        r = self.A @ x - self.b
        return 0.5 * float(r @ r)

    def grad(self, x):
        return self.H @ x - self.Ab


# ---------- Core residuals (consistent with LaTeX) ----------
def F_val(u, u_k, eta, prob, gprime='exp'):
    """
    F(u) = u - u_k + eta * grad_x Phi(g(u)), with x=g(u)
    grad_x Phi(x) = A^T(Ax - b) = H x - A^T b  (here we use prob.grad(x))
    """
    x = safe_exp(u) if gprime == 'exp' else None  # extend if you add other maps
    return (u - u_k) + eta * prob.grad(x)


def J_mat(u, eta, prob, gprime='exp'):
    """
    J(u) = I + eta * H * diag(g'(u)).
    For g_i(u_i)=exp(u_i), diag(g'(u))=diag(x)=diag(exp(u)).
    """
    if gprime == 'exp':
        x = safe_exp(u)
        # Column-scaling of H by x: H @ diag(x)
        return np.eye(prob.n) + eta * (prob.H @ np.diag(x))
    else:
        raise NotImplementedError("Only exp() mapping implemented here.")


# ---------- Inner solver: Modified Gauss–Newton (recommended) ----------
def gravidy_pos_step_mgn(u_k, eta, prob, tol=1e-10, max_iter=50,
                         M=None, M_min=1e-12, M_max=1e12, growth=2.0):
    """
    MGN inner solve for F(u)=0:
      (J^T J + M I) h = - J^T F
    Accept if residual decreases; adapt M multiplicatively.
    """
    n = prob.n
    u = u_k.copy()

    def residual(u):
        F = F_val(u, u_k, eta, prob, gprime='exp')
        return F, np.linalg.norm(F)

    # initialize damping if not provided (cheap spectral proxy)
    if M is None:
        # rough scale: ||H|| * median(x)
        x0 = safe_exp(u_k)
        Hnorm = np.linalg.norm(prob.H, 2)
        M = max(1e-6, Hnorm * max(1e-3, float(np.median(x0))))

    for _ in range(max_iter):
        F, nF = residual(u)
        if nF <= tol:
            break

        J = J_mat(u, eta, prob, gprime='exp')

        # Solve SPD normal eqs (J^T J + M I) h = - J^T F by Cholesky
        JTJ = J.T @ J
        rhs = - J.T @ F

        accepted = False
        inner_tries = 0
        while not accepted and inner_tries < 5:
            try:
                L = np.linalg.cholesky(JTJ + M * np.eye(n))
                y = np.linalg.solve(L, rhs)
                h = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                # if factorization fails, increase damping and retry
                M = min(M_max, M * growth)
                inner_tries += 1
                continue

            u_trial = u + h
            F_trial, nF_trial = residual(u_trial)

            if nF_trial < nF:  # success
                u = u_trial
                # decrease damping (but keep a floor)
                M = max(M_min, M / growth)
                accepted = True
            else:
                # increase damping and retry (short-circuit line search)
                M = min(M_max, M * growth)
                inner_tries += 1

        # if we couldn't accept the step after retries, do a small safeguarded step
        if not accepted:
            u = u - (rhs / (np.linalg.norm(JTJ, 2) + M + 1e-12))  # tiny gradient-like fallback

    x_next = safe_exp(u)
    return u, x_next


# ---------- Inner solver: pure Newton (optional) ----------
def gravidy_pos_step_newton(u_k, eta, prob, tol=1e-10, max_newton=50, backtrack_beta=0.5):
    """
    Newton on J(u) h = -F(u), with backtracking on m(u)=0.5||F(u)||^2.
    Uses the *consistent* J(u)=I + eta * H * diag(exp(u)) (no extra terms).
    """
    def merit(u):
        F = F_val(u, u_k, eta, prob, gprime='exp')
        return 0.5 * float(F @ F)

    u = u_k.copy()
    n = prob.n
    for _ in range(max_newton):
        F = F_val(u, u_k, eta, prob, gprime='exp')
        nF = np.linalg.norm(F)
        if nF < tol:
            break

        J = J_mat(u, eta, prob, gprime='exp')
        # Solve J h = -F (symmetric-indefinite possible)
        try:
            h = -np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            h = -np.linalg.pinv(J) @ F

        # Armijo backtracking on merit
        m0 = merit(u)
        alpha = 1.0
        while True:
            u_trial = u + alpha * h
            if merit(u_trial) <= m0 or alpha < 1e-12:
                break
            alpha *= backtrack_beta
        u = u_trial

    x_next = safe_exp(u)
    return u, x_next


# ---------- Outer driver ----------
def GRAVIDY_pos(problem, eta=30.0, max_outer=400, tol_grad=1e-10, inner='mgn',
                verbose=False):
    """
    GRAVIDY–pos outer loop. Choose inner='mgn' (default) or 'newton'.
    """
    prob = problem
    n = prob.n

    # start from u=0 => x=1 (neutral positive point)
    u = np.zeros(n)
    x = safe_exp(u)

    history = []
    t0 = time.time()

    for k in range(max_outer):
        obj_val = prob.f(x)
        grad = prob.grad(x)
        grad_norm = np.linalg.norm(grad)
        t = time.time() - t0
        history.append((k, obj_val, grad_norm, t))

        if verbose and (k % 50 == 0 or k == max_outer - 1):
            print(f"[GRAVIDY-pos/{inner}] it={k:4d}  f={obj_val:.6e}  ||grad||={grad_norm:.3e}  t={t:.2f}s")

        if grad_norm <= tol_grad:
            break

        if inner == 'mgn':
            u, x = gravidy_pos_step_mgn(u, eta, prob, tol=1e-10, max_iter=50)
        elif inner == 'newton':
            u, x = gravidy_pos_step_newton(u, eta, prob, tol=1e-10, max_newton=50)
        else:
            raise ValueError("inner must be 'mgn' or 'newton'")

    return x, history


# ---------- Tiny smoke test ----------
if __name__ == "__main__":
    np.random.seed(0)
    m, n = 120, 80
    A = np.abs(np.random.randn(m, n)) + 0.05 * np.eye(m, n)  # nonnegative-ish
    x_star = np.abs(np.random.randn(n))
    b = A @ x_star

    prob = PositiveLeastSquares(A, b)

    # MGN
    x_mgn, hist_mgn = GRAVIDY_pos(prob, eta=30.0, inner='mgn', verbose=True)
    # Newton
    x_nt,  hist_nt  = GRAVIDY_pos(prob, eta=30.0, inner='newton', verbose=False)

    print("MGN   final f =", prob.f(x_mgn))
    print("Newton final f =", prob.f(x_nt))
