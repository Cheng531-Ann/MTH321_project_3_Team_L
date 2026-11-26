"""Free-boundary extraction from the obstacle slack."""

from __future__ import annotations

import numpy as np


def extract_free_boundary(S, V, g, slack_tol=1e-10):
    """Return (point estimate, left bracket, right bracket) of B(tau).

    ``j`` is the last contact node (V_i = g_i) and ``j + 1`` the first
    continuation node.  In the continuation region the slack behaves locally
    like c (S - B)^2, so the square root of the slack is near-linear and is
    interpolated between nodes ``j + 1`` and ``j + 2``.  The estimate is
    clamped to the bracket [S_j, S_{j+1}].
    """
    slack = V - g
    # The exercise region is a prefix of the grid, and B(tau) < K always.  The
    # right boundary node (S = Smax) has V = g = 0 by construction but is a
    # truncation boundary, not a contact node.  Deep out-of-the-money nodes
    # have g = 0 with a tiny-but-positive continuation value, so they are not
    # contact nodes either.  We therefore search only nodes with S < K.
    n = len(S)
    strike = g[0]  # g = (K - S)^+ so g(0) = K
    interior = np.arange(n - 1)              # indices 0..n-2
    below_strike = S[interior] < strike
    idx = interior[below_strike]
    threshold = slack_tol * max(1.0, float(strike))
    contact = idx[slack[idx] <= threshold]

    if contact.size == 0:
        return S[0], S[0], S[0]

    j = int(contact[-1])
    if j >= n - 1:
        return S[-1], S[-1], S[-1]

    w = np.sqrt(np.maximum(V - g, 0.0))
    h = S[1] - S[0]

    if j + 2 < n and w[j + 2] - w[j + 1] > 0.0:
        B = S[j + 1] - h * w[j + 1] / (w[j + 2] - w[j + 1])
    else:
        B = 0.5 * (S[j] + S[j + 1])

    B = float(np.clip(B, S[j], S[j + 1]))
    return B, S[j], S[j + 1]
