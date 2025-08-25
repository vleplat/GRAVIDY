"""
GRAVIDY–box solver for box-constrained least squares:
Non-pullback implicit Euler in reparameterized coordinates x = lo + (hi-lo) * sigmoid(z).

Inner solvers:
- MGN (paper default): solves (J^T J + λ I) h = -J^T F by matrix-free CG
- Newton: solves J h = -F via SPD transform and CG

Paper's formulation:
  F(z) = z - z_k + η ∇_x Φ(x(z)), x(z) = lo + (hi-lo) * sigmoid(z)
  J(z) = I + η ∇_x² Φ(x) Diag(g'(z))
"""

import numpy as np
import time


# -------- numerically-stable sigmoid and box map --------
def sigmoid(z):
    zc = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-zc))


def box_x(z, lo, hi):
    s = sigmoid(z)
    return lo + (hi - lo) * s


def box_gprime(z, lo, hi):
    s = sigmoid(z)
    return (hi - lo) * s * (1.0 - s)  # elementwise > 0 since hi>lo


# -------- tiny CG (SPD) --------
def _cg(matvec, b, tol=1e-8, maxit=200):
    x = np.zeros_like(b)
    r = b - matvec(x)
    p = r.copy()
    rs = r @ r
    bnorm2 = max(b @ b, 1e-30)
    if rs <= (tol * tol) * bnorm2:
        return x
    for _ in range(maxit):
        Ap = matvec(p)
        denom = p @ Ap + 1e-30
        alpha = rs / denom
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        if rs_new <= (tol * tol) * bnorm2:
            break
        p = r + (rs_new / (rs + 1e-30)) * p
        rs = rs_new
    return x


# -------- inner: Newton (non-pullback, paper-faithful) --------
def gravidy_box_step_newton(z_k, eta, A, b, lo, hi,
                            tol=1e-10, max_newton=50,
                            backtrack_beta=0.5, cg_tol=1e-8, cg_maxit=200):
    """
    Solve F(z)=0 with F(z)=z - z_k + eta * grad_x Phi(x(z)),
    Phi(x)=0.5||Ax-b||^2,  x(z)=lo + (hi-lo)*sigmoid(z).
    J = I + eta * H * D,   H=A^T A,  D=diag(g'(z)).
    Newton system solved via SPD transform: (I + eta * sqrt(D) H sqrt(D)) v = - sqrt(D) F,
    then h = D^{-1/2} v.
    """
    AT = A.T

    def F_val(z):
        x = box_x(z, lo, hi)
        r = A @ x - b
        g = AT @ r
        return (z - z_k) + eta * g

    def merit(F):
        return 0.5 * float(F @ F)

    z = z_k.copy()
    for _ in range(max_newton):
        x = box_x(z, lo, hi)
        r = A @ x - b
        g = AT @ r
        F = (z - z_k) + eta * g
        nF = np.linalg.norm(F)
        if nF <= tol:
            break

        gp = box_gprime(z, lo, hi)
        sqrtgp = np.sqrt(np.maximum(gp, 0.0))
        rhs = - sqrtgp * F  # -D^{1/2} F

        # SPD operator: v -> v + eta * sqrt(D) H sqrt(D) v
        def spd_matvec(v):
            w = sqrtgp * v                   # sqrt(D) v
            Aw = A @ w
            HTw = AT @ Aw                    # H * sqrt(D) v
            return v + eta * (sqrtgp * HTw)  # v + eta * sqrt(D) H sqrt(D) v

        v = _cg(spd_matvec, rhs, tol=cg_tol, maxit=cg_maxit)
        h = v / (sqrtgp + 1e-30)            # h = D^{-1/2} v

        # Armijo backtracking on 0.5||F||^2
        m0 = merit(F)
        alpha = 1.0
        while True:
            z_trial = z + alpha * h
            F_trial = F_val(z_trial)
            if merit(F_trial) <= m0 or alpha < 1e-12:
                z = z_trial
                break
            alpha *= backtrack_beta

    return z, box_x(z, lo, hi)


# -------- inner: MGN (matrix-free, non-pullback) --------
def gravidy_box_step_mgn(z_k, eta, A, b, lo, hi, tol=1e-10, max_iter=50,
                         lm=1e-6, backtrack_beta=0.5):
    """
    Modified Gauss–Newton on (J^T J + lm I) h = - J^T F,
    with Jv = v + eta * A^T( A ( D v ) ),  JT v = v + eta * D ( A^T (A v) ).
    """
    AT = A.T

    def F_val(z):
        x = box_x(z, lo, hi)
        return (z - z_k) + eta * (AT @ (A @ x - b))

    def merit(F):
        return 0.5 * float(F @ F)

    z = z_k.copy()
    for _ in range(max_iter):
        x = box_x(z, lo, hi)
        gp = box_gprime(z, lo, hi)
        Dv = lambda v: gp * v

        F = F_val(z)
        nF = np.linalg.norm(F)
        if nF <= tol:
            break

        def Jv(v):
            return v + eta * (AT @ (A @ Dv(v)))

        def JT(v):
            return v + eta * Dv(AT @ (A @ v))

        rhs = - JT(F)

        def JTJ(v):
            return JT(Jv(v)) + lm * v

        # relaxed tol when far, tighter when close
        cg_tol = min(1e-2, 0.1*np.sqrt(max(nF, 1e-30)))
        h = _cg(JTJ, rhs, tol=cg_tol, maxit=200)

        # Armijo backtracking on 0.5||F||^2
        m0 = merit(F)
        alpha = 1.0
        while True:
            z_trial = z + alpha * h
            F_trial = F_val(z_trial)
            if merit(F_trial) <= m0 or alpha < 1e-12:
                z = z_trial
                break
            alpha *= backtrack_beta

        # optional LM adaptation:
        # if alpha == 1.0: lm = max(lm * 0.5, 1e-12)
        # else:            lm = min(lm * 2.0, 1e-2)

    return z, box_x(z, lo, hi)


# -------- Problem wrapper --------
class BoxLeastSquares:
    """Simple problem wrapper: min_x 0.5||A x - b||^2 s.t. lo <= x <= hi."""
    def __init__(self, A, b, lo, hi):
        self.A = A
        self.b = b
        self.lo = lo
        self.hi = hi
        self.n = A.shape[1]
        self.H = A.T @ A  # Hessian
        self.Ab = A.T @ b

    def f(self, x):
        r = self.A @ x - self.b
        return 0.5 * float(r @ r)

    def grad(self, x):
        return self.H @ x - self.Ab


# -------- outer driver --------
def GRAVIDY_box(problem, eta=10.0, max_outer=200, tol_grad=1e-10, inner='mgn', verbose=False):
    """
    Non-pullback GRAVIDY–box outer loop (paper-faithful).
    problem must provide A, b, lo, hi, n, f(x), grad(x).
    inner: 'newton' or 'mgn'
    """
    A, b, lo, hi, n = problem.A, problem.b, problem.lo, problem.hi, problem.n
    z = np.zeros(n)                 # midpoint in the box
    x = box_x(z, lo, hi)

    history = []
    t0 = time.time()
    for k in range(max_outer):
        obj = problem.f(x)
        g = problem.grad(x)
        ng = np.linalg.norm(g)
        history.append((k, obj, ng, time.time() - t0))

        if verbose and (k % 50 == 0):
            print(f"[GRAVIDY-box/{inner}] it={k:4d}  f={obj:.6e}  ||grad||={ng:.3e}")

        if ng <= tol_grad:
            break

        if inner == 'newton':
            z, x = gravidy_box_step_newton(z, eta, A, b, lo, hi, tol=1e-10)
        elif inner == 'mgn':
            z, x = gravidy_box_step_mgn(z, eta, A, b, lo, hi, tol=1e-10)
        else:
            raise ValueError("inner must be 'newton' or 'mgn'")

    return x, history


# -------- Example run --------
if __name__ == "__main__":
    np.random.seed(1)
    m, n = 150, 80
    A = np.random.randn(m, n)
    lo = -1.0 * np.ones(n)
    hi = 2.0 * np.ones(n)
    x_star = lo + (hi - lo) * np.random.rand(n)
    b = A @ x_star
    prob = BoxLeastSquares(A, b, lo, hi)
    
    x_mgn, hist_mgn = GRAVIDY_box(prob, eta=10.0, inner='mgn', verbose=True)
    x_nt, hist_nt = GRAVIDY_box(prob, eta=10.0, inner='newton', verbose=False)
    
    print("MGN final f =", prob.f(x_mgn))
    print("Newton final f =", prob.f(x_nt))
