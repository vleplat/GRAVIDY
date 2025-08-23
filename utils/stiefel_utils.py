import numpy as np

def rand_stiefel(n, p, seed=0):
    """Generate a random point on the Stiefel manifold St(n,p)."""
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n, p))
    Q, R = np.linalg.qr(Y)
    # Ensure det(R) positive on diagonal blocks for numeric stability
    d = np.sign(np.diag(R))
    d[d == 0.0] = 1.0
    Q = Q @ np.diag(d)
    return Q

def sym(M):
    """Extract symmetric part of matrix M."""
    return 0.5 * (M + M.T)

def skew(M):
    """Extract skew-symmetric part of matrix M."""
    return 0.5 * (M - M.T)

def retraction_qr(X, H):
    """First-order retraction: QR of X+H."""
    Q, R = np.linalg.qr(X + H)
    d = np.sign(np.diag(R))
    d[d == 0.0] = 1.0
    Q = Q @ np.diag(d)
    return Q
