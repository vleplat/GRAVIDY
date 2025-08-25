"""
GRAVIDY–pos solver for NNLS / positive-orthant LS:
Non-pullback implicit Euler in reparameterized coordinates x = exp(u).

Inner solvers:
- MGN (paper default): solves (J^T J + λ I) h = -J^T F by matrix-free CG
- Newton: solves J h = -F via SPD transform and CG

Paper's formulation:
  F(u) = u - u_k + η ∇_x Φ(x(u)), x(u) = exp(u)
  J(u) = I + η ∇_x² Φ(x) Diag(x)
"""

import numpy as np
import time


# ---------- helpers ----------
def safe_exp(u):
    """Conservative clamp to keep positivity mapping numerically stable."""
    return np.exp(np.clip(u, -40.0, 40.0))


def _cg(matvec, b, tol=1e-8, maxit=200):
    """Conjugate Gradient for SPD operator given by matvec(v)."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = r @ r
    if rs == 0:
        return x
    bnorm2 = max(b @ b, 1e-30)
    for _ in range(maxit):
        Ap = matvec(p)
        denom = p @ Ap + 1e-30
        alpha = rs / denom
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        if rs_new <= (tol * tol) * bnorm2:
            break
        beta = rs_new / (rs + 1e-30)
        p = r + beta * p
        rs = rs_new
    return x


# ---------- non-pullback INNER: Newton via SPD transform ----------
def gravidy_pos_step_newton(u_k, eta, A, b, tol=1e-10, max_inner=50,
                            backtrack_beta=0.5, cg_tol=1e-8, cg_maxit=200):
    """
    Non-pullback Newton step: solve J(u) h = -F(u), where
      F(u) = u - u_k + eta * g(x), x=exp(u), g(x)=A^T(Ax-b)
      J(u) = I + eta * H * Diag(x), H=A^T A.

    We solve with the SPD transform:
      let D=Diag(x), h = D^{-1/2} v,
      (I + eta * D^{1/2} H D^{1/2}) v = - D^{1/2} F
    by CG, using only A/AT matvecs.
    """
    AT = A.T

    def merit(F):  # 0.5 ||F||^2
        return 0.5 * float(F @ F)

    u = u_k.copy()
    for _ in range(max_inner):
        x = safe_exp(u)
        Ax = A @ x
        r  = Ax - b
        g  = AT @ r                       # grad in x
        F  = (u - u_k) + eta * g          # residual in u
        nF = np.linalg.norm(F)
        if nF <= tol:
            break

        sqrtx = np.sqrt(x)
        rhs   = - sqrtx * F               # -D^{1/2} F

        # SPD matvec: (I + eta * D^{1/2} H D^{1/2}) v
        def spd_matvec(v):
            w  = sqrtx * v                # D^{1/2} v
            Aw = A @ w
            HTw = AT @ Aw                 # H * (D^{1/2} v)
            return v + eta * (sqrtx * HTw)  # v + eta * D^{1/2} H D^{1/2} v

        # solve for v, then recover h = D^{-1/2} v
        v = _cg(spd_matvec, rhs, tol=cg_tol, maxit=cg_maxit)
        h = v / (sqrtx + 1e-30)

        # Armijo backtracking on 0.5||F||^2
        m0 = merit(F)
        alpha = 1.0
        while True:
            u_trial = u + alpha * h
            x_trial = safe_exp(u_trial)
            F_trial = (u_trial - u_k) + eta * (AT @ (A @ x_trial) - AT @ b)
            if merit(F_trial) <= m0 or alpha < 1e-12:
                u = u_trial
                break
            alpha *= backtrack_beta

    x_next = safe_exp(u)
    return u, x_next


# ---------- non-pullback INNER: MGN (paper default) ----------
def gravidy_pos_step_mgn(u_k, eta, A, b, tol=1e-10, max_inner=50,
                         lm=1e-6, backtrack_beta=0.5):
    """
    Non-pullback MGN step: normal equations (J^T J + lm I) h = - J^T F,
    with matrix-free J and J^T using only A and A^T products.
    """
    AT = A.T

    def merit(F):  # 0.5 ||F||^2
        return 0.5 * float(F @ F)

    def cg(matvec, b, tol=1e-8, maxit=200):
        return _cg(matvec, b, tol, maxit)

    u = u_k.copy()
    for _ in range(max_inner):
        x  = safe_exp(u)
        Ax = A @ x
        r  = Ax - b
        g  = AT @ r                       # grad in x
        F  = (u - u_k) + eta * g
        nF = np.linalg.norm(F)
        if nF <= tol:
            break

        # J v = v + eta * A^T( A ( x ⊙ v ) )
        def Jv(v):
            return v + eta * (AT @ (A @ (x * v)))

        # J^T v = v + eta * x ⊙ ( A^T( A v ) )
        def JT(v):
            return v + eta * (x * (AT @ (A @ v)))

        rhs = -JT(F)

        # (J^T J + lm I) matvec for CG
        def JTJ(v):
            return JT(Jv(v)) + lm * v

        # Slightly relaxed tol when residual is large; tightens as ||F|| shrinks
        cg_tol = min(1e-2, 0.1*np.sqrt(nF) + 1e-12)
        h = cg(JTJ, rhs, tol=cg_tol, maxit=200)

        # Armijo backtracking on 0.5||F||^2
        m0 = merit(F)
        alpha = 1.0
        while True:
            u_trial = u + alpha * h
            x_trial = safe_exp(u_trial)
            F_trial = (u_trial - u_k) + eta * (AT @ (A @ x_trial) - AT @ b)
            if merit(F_trial) <= m0 or alpha < 1e-12:
                u = u_trial
                break
            alpha *= backtrack_beta

        # simple LM adaptation (optional)
        # if alpha == 1.0: lm = max(lm * 0.5, 1e-12)
        # else:            lm = min(lm * 2.0, 1e-2)

    x_next = safe_exp(u)
    return u, x_next


# ---------- Problem wrapper ----------
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


# ---------- outer loop ----------
def GRAVIDY_pos(problem, eta=30.0, max_outer=400, tol_grad=1e-10,
                inner='mgn', verbose=False):
    """
    Non-pullback GRAVIDY–pos outer loop.
      problem must provide A, b, n, f(x), grad(x).
    inner: 'mgn' (paper default) or 'newton'
    """
    A, b, n = problem.A, problem.b, problem.n
    u = np.zeros(n)          # x = 1 initial
    x = safe_exp(u)

    history = []
    t0 = time.time()

    for k in range(max_outer):
        obj_val = problem.f(x)
        grad    = problem.grad(x)
        grad_norm = np.linalg.norm(grad)
        history.append((k, obj_val, grad_norm, time.time() - t0))

        if verbose and (k % 50 == 0):
            print(f"[GRAVIDY-pos/{inner}] iter={k:4d} f={obj_val:.6e} ||grad||={grad_norm:.3e}")

        # Optional KKT residual for the orthant:
        # kkt = np.linalg.norm(x * grad, ord=np.inf)

        if grad_norm <= tol_grad:
            break

        if inner == 'mgn':
            u, x = gravidy_pos_step_mgn(u, eta, A, b, tol=1e-10)
        elif inner == 'newton':
            u, x = gravidy_pos_step_newton(u, eta, A, b, tol=1e-10)
        else:
            raise ValueError("inner must be 'mgn' (paper default) or 'newton'")

    return x, history


# ---------- Example run ----------
if __name__ == "__main__":
    np.random.seed(1)
    m, n = 150, 80
    A = np.random.randn(m, n)
    x_star = np.abs(np.random.randn(n))  # positive ground truth
    b = A @ x_star
    prob = PositiveLeastSquares(A, b)
    
    x_mgn, hist_mgn = GRAVIDY_pos(prob, eta=10.0, inner='mgn', verbose=True)
    x_nt, hist_nt = GRAVIDY_pos(prob, eta=10.0, inner='newton', verbose=False)
    
    print("MGN final f =", prob.f(x_mgn))
    print("Newton final f =", prob.f(x_nt))
