import numpy as np

class Anderson:
    """
    Minimal Anderson(m) accelerator for matrix fixed-point Y = T(Y).
    Stores m residual differences across iterates; uses least squares mixing.
    """
    def __init__(self, m=5, lam=1e-8):
        self.m = m
        self.lam = lam
        self.F_hist = []
        self.dY_hist = []

    def reset(self):
        """Reset the acceleration history."""
        self.F_hist.clear()
        self.dY_hist.clear()

    def step(self, Y, T_Y):
        """Perform one Anderson acceleration step."""
        # F = T(Y) - Y
        F = T_Y - Y
        m = self.m
        if len(self.F_hist) < m:
            self.F_hist.append(F.copy())
            self.dY_hist.append((T_Y - Y).copy())
            return T_Y
        # Build least-squares system for mixing coefficients
        F_stack = np.stack([F_i.ravel() for F_i in self.F_hist[-m:]], axis=1)  # (n*p) x m
        rhs = F.ravel()
        # Solve (F_stack^T F_stack + lam I) c = F_stack^T rhs
        M = F_stack.T @ F_stack + self.lam * np.eye(m)
        b = F_stack.T @ rhs
        try:
            c = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            c = np.zeros(m)
        dY = sum(ci * d for ci, d in zip(c, self.dY_hist[-m:]))
        Y_new = Y + (T_Y - Y) - dY.reshape(Y.shape)
        self.F_hist.append(F.copy())
        self.dY_hist.append((T_Y - Y).copy())
        return Y_new
