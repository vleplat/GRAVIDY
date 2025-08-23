import time
import numpy as np
from utils.stiefel_utils import retraction_qr

def RGD_QR(problem,
           X0,
           alpha0=1.0,
           max_iters=2000,
           tol_grad=1e-8,
           armijo_c1=1e-4,
           backtrack=0.5,
           grow=1.5,
           alpha_min=1e-12,
           alpha_max=1e3,
           verbose=False):
    """
    Riemannian gradient descent with QR retraction.
    """
    X = X0.copy()
    f = problem.f
    n, p = X.shape
    hist = {"it": [], "time": [], "f": [], "feas": [], "grad_norm": []}
    t0 = time.time()
    alpha = alpha0

    for k in range(max_iters):
        gR = problem.grad_riem(X)
        gnorm = np.linalg.norm(gR, 'fro')
        feas = np.linalg.norm(X.T @ X - np.eye(p), 'fro')
        hist["it"].append(k)
        hist["time"].append(time.time() - t0)
        hist["f"].append(f(X))
        hist["feas"].append(feas)
        hist["grad_norm"].append(gnorm)

        if verbose and (k % 10 == 0):
            print(f"[RGD] it={k:4d} f={hist['f'][-1]:.6e} ||grad||={gnorm:.3e} feas={feas:.2e} alpha={alpha:.2e}")

        if gnorm <= tol_grad:
            break

        # Trial
        Y = X - alpha * gR
        X_trial = retraction_qr(X, -alpha * gR)
        f_old = hist["f"][-1]
        f_new = f(X_trial)
        if f_new <= f_old - armijo_c1 * alpha * (gnorm ** 2):
            X = X_trial
            alpha = min(alpha * grow, alpha_max)
        else:
            alpha = max(alpha * backtrack, alpha_min)

    return X, hist
