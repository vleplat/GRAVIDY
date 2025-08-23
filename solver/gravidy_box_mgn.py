"""
GRAVIDY–box solver for box-constrained least squares:
min 0.5 ||A x - b||^2  s.t.  lo <= x <= hi.

Reparameterization: x(z) = lo + (hi - lo) * sigmoid(z)
Implicit step in z (pullback-style): F(z) = z - z_k + eta * D(z) * grad f(x(z)) = 0
with D(z)=diag(g'(z)), g'(z)=(hi-lo)*sigmoid(z)*(1-sigmoid(z))

Inner solvers:
- Newton  : solves J h = -F, with Armijo on 0.5||F||^2
- MGN     : solves (J^T J + M I) h = - J^T F (SPD) with adaptive damping M

Both use the same Jacobian:
  J(z) = I + eta * [ diag(g''(z) ∘ grad f(x(z))) + D(z) H D(z) ],
  H = A^T A
"""

import numpy as np
import time


# ---------- Problem wrapper ----------
class BoxLeastSquares:
    def __init__(self, A, b, lo, hi):
        self.A = A
        self.b = b
        self.lo = lo
        self.hi = hi
        self.n = A.shape[1]
        self.H = A.T @ A
        self.Ab = A.T @ b

    def f(self, x):
        r = self.A @ x - self.b
        return 0.5 * float(r @ r)

    def grad(self, x):
        return self.H @ x - self.Ab


# ---------- Stable sigmoid ----------
def sigmoid(z):
    zc = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-zc))


# ---------- Mapping and derivatives ----------
def box_map_x(z, lo, hi):
    s = sigmoid(z)
    return lo + (hi - lo) * s

def box_map_gprime(z, lo, hi):
    s = sigmoid(z)
    return (hi - lo) * s * (1.0 - s)

def box_map_gsecond(z, lo, hi):
    s = sigmoid(z)
    return (hi - lo) * s * (1.0 - s) * (1.0 - 2.0 * s)


# ---------- Residual and Jacobian (your formulation) ----------
def F_box(z, z_k, eta, prob: BoxLeastSquares):
    x = box_map_x(z, prob.lo, prob.hi)
    gp = box_map_gprime(z, prob.lo, prob.hi)
    return z - z_k + eta * (gp * prob.grad(x))

def J_box(z, eta, prob: BoxLeastSquares):
    x = box_map_x(z, prob.lo, prob.hi)
    gp = box_map_gprime(z, prob.lo, prob.hi)
    gpp = box_map_gsecond(z, prob.lo, prob.hi)
    D = np.diag(gp)
    # symmetric Jacobian of F(z) in this pullback-style formulation
    return np.eye(prob.n) + eta * (np.diag(gpp * prob.grad(x)) + D @ prob.H @ D)


# ---------- Inner solver: Newton (your code, tidied) ----------
def gravidy_box_step_newton(z_k, eta, prob: BoxLeastSquares,
                            tol=1e-10, max_newton=50, backtrack_beta=0.5):
    def merit(z):
        F = F_box(z, z_k, eta, prob)
        return 0.5 * float(F @ F)

    z = z_k.copy()
    for _ in range(max_newton):
        F = F_box(z, z_k, eta, prob)
        nF = np.linalg.norm(F)
        if nF < tol:
            break

        J = J_box(z, eta, prob)
        try:
            h = -np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            # tiny Tikhonov if needed
            h = -np.linalg.solve(J + 1e-10 * np.eye(prob.n), F)

        # Armijo backtracking on m(z)=0.5||F||^2
        m0, alpha = merit(z), 1.0
        while True:
            z_trial = z + alpha * h
            if merit(z_trial) <= m0 or alpha < 1e-12:
                break
            alpha *= backtrack_beta
        z = z_trial

    x_next = box_map_x(z, prob.lo, prob.hi)
    return z, x_next


# ---------- Inner solver: Modified Gauss–Newton (recommended near active bounds) ----------
def gravidy_box_step_mgn(z_k, eta, prob: BoxLeastSquares,
                         tol=1e-10, max_iter=50, M=None,
                         growth=2.0, M_min=1e-12, M_max=1e12):
    """
    MGN on (J^T J + M I) h = - J^T F, with multiplicative damping adaptation:
    - if residual decreases, M <- max(M_min, M/growth)
    - else increase M <- min(M_max, M*growth) and retry
    """
    def residual(z):
        F = F_box(z, z_k, eta, prob)
        return F, np.linalg.norm(F)

    # init
    z = z_k.copy()
    if M is None:
        # rough scale from H and typical g' magnitude near the middle of the box
        Hn = np.linalg.norm(prob.H, 2)
        M = max(1e-6, Hn * 1e-1)

    for _ in range(max_iter):
        F, nF = residual(z)
        if nF <= tol:
            break

        J = J_box(z, eta, prob)
        JTJ = J.T @ J
        rhs = - J.T @ F

        accepted = False
        tries = 0
        while not accepted and tries < 5:
            try:
                L = np.linalg.cholesky(JTJ + M * np.eye(prob.n))
                y = np.linalg.solve(L, rhs)
                h = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                M = min(M_max, M * growth)
                tries += 1
                continue

            z_trial = z + h
            _, nF_trial = residual(z_trial)
            if nF_trial < nF:
                z = z_trial
                M = max(M_min, M / growth)
                accepted = True
            else:
                M = min(M_max, M * growth)
                tries += 1

        if not accepted:
            # very small fallback step if nothing was accepted
            z = z - rhs / (np.linalg.norm(JTJ, 2) + M + 1e-12)

    x_next = box_map_x(z, prob.lo, prob.hi)
    return z, x_next


# ---------- Outer driver ----------
def GRAVIDY_box(problem: BoxLeastSquares, eta=10.0, max_outer=200, tol_grad=1e-10,
                inner='mgn', verbose=False):
    """
    GRAVIDY–box outer loop.
    inner: 'mgn' (default) or 'newton'
    """
    prob = problem
    n = prob.n

    # start from box midpoint: z=0 => x(mid) = lo + (hi-lo)*0.5
    z = np.zeros(n)
    x = box_map_x(z, prob.lo, prob.hi)

    history = []
    t0 = time.time()

    for k in range(max_outer):
        obj = prob.f(x)
        g = prob.grad(x)
        ng = np.linalg.norm(g)
        elapsed = time.time() - t0

        history.append((k, obj, ng, elapsed))
        if verbose and (k % 50 == 0 or k == max_outer - 1):
            print(f"[GRAVIDY-box/{inner}] it={k:4d}  f={obj:.6e}  ||grad||={ng:.3e}  t={elapsed:.2f}s")

        if ng <= tol_grad:
            break

        if inner == 'newton':
            z, x = gravidy_box_step_newton(z, eta, prob, tol=1e-10, max_newton=50)
        elif inner == 'mgn':
            z, x = gravidy_box_step_mgn(z, eta, prob, tol=1e-10, max_iter=50)
        else:
            raise ValueError("inner must be 'newton' or 'mgn'")

    return x, history


# ---------- Example run ----------
if __name__ == "__main__":
    np.random.seed(1)
    m, n = 150, 80
    A = np.random.randn(m, n)
    # make the target inside a box
    lo = -1.0 * np.ones(n)
    hi = 2.0 * np.ones(n)
    x_star = lo + (hi - lo) * np.random.rand(n)
    b = A @ x_star

    prob = BoxLeastSquares(A, b, lo, hi)

    x_mgn, hist_mgn = GRAVIDY_box(prob, eta=10.0, inner='mgn', verbose=True)
    x_nt,  hist_nt  = GRAVIDY_box(prob, eta=10.0, inner='newton', verbose=False)

    print("MGN   final f =", prob.f(x_mgn))
    print("Newton final f =", prob.f(x_nt))
