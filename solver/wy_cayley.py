import time
import numpy as np

def WY_cayley(problem,
              X0,
              alpha0=1.0,
              max_iters=200,
              tol_grad=1e-8,
              armijo_c1=1e-4,
              backtrack=0.5,
              grow=1.5,
              alpha_min=1e-8,
              alpha_max=1e6,
              verbose=False):
    """
    Wen–Yin (2013) feasible Cayley step:
      Q = (I + a/2 A(Xk))^{-1} (I - a/2 A(Xk))  (orthogonal)
      X_{k+1} = Q X_k
    with Armijo backtracking on f.
    """
    X = X0.copy()
    f = problem.f
    n, p = X.shape
    hist = {"it": [], "time": [], "f": [], "feas": []}
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

        if verbose:
            print(f"[WY]  it={k:3d} f={hist['f'][-1]:.6e} ||grad||={gnorm:.3e} feas={feas:.2e} alpha={alpha:.2e}")

        if gnorm <= tol_grad:
            break

        A = problem.A_skew(X)
        I = np.eye(n)
        rhs = (I - 0.5 * alpha * A) @ X
        L = (I + 0.5 * alpha * A)
        try:
            QX = np.linalg.solve(L, rhs)
        except np.linalg.LinAlgError:
            QX = np.linalg.solve(L + 1e-12 * np.eye(n), rhs)

        # Armijo acceptance
        f_old = hist["f"][-1]
        f_new = f(QX)
        if f_new <= f_old - armijo_c1 * alpha * (gnorm ** 2):
            X = QX
            alpha = min(alpha * grow, alpha_max)
        else:
            alpha = max(alpha * backtrack, alpha_min)

    return X, hist
