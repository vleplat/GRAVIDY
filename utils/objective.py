import numpy as np
from .stiefel_utils import sym

class StiefelQuad:
    """
    Phi(X) = 0.5 * sum_j x_j^T Q_j x_j
    Gradient (Euclidean): G = [Q_1 x_1, ..., Q_p x_p]
    Riemannian grad (canonical): grad_R = G - X * sym(X^T G)
    """
    def __init__(self, Q_list):
        self.Q_list = Q_list
        self.n = Q_list[0].shape[0]
        self.p = len(Q_list)

    def apply_Q(self, X):
        """Columnwise application of Q matrices."""
        G = np.empty_like(X)
        for j in range(self.p):
            G[:, j] = self.Q_list[j] @ X[:, j]
        return G

    def f(self, X):
        """Compute objective value."""
        val = 0.0
        for j in range(self.p):
            xj = X[:, j]
            val += 0.5 * float(xj.T @ (self.Q_list[j] @ xj))
        return val

    def grad_euclid(self, X):
        """Compute Euclidean gradient."""
        return self.apply_Q(X)

    def grad_riem(self, X):
        """Compute Riemannian gradient."""
        G = self.grad_euclid(X)
        return G - X @ sym(X.T @ G)

    def A_skew(self, X):
        """Compute A(X) = G X^T - X G^T (skew-symmetric)."""
        G = self.grad_euclid(X)
        return G @ X.T - X @ G.T
