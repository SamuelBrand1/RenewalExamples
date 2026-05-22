"""Liu-West (2001) shrink-jitter kernel for the parameter particles.

For each parameter particle ``theta_i`` and current normalized weights ``w_i``,
let ``m = sum_i w_i theta_i`` and ``V = sum_i w_i (theta_i - m)(theta_i - m)^T``.
Set ``a = sqrt(1 - h^2)``.  Then

    theta_i_new = a * theta_i + (1 - a) * m + eps_i,   eps_i ~ N(0, h^2 V).

The shrinkage toward ``m`` exactly cancels the variance inflation that ``eps_i``
introduces, so the kernel preserves the mean and (asymptotically) the covariance
of the weighted theta cloud.  ``h`` controls the per-step jitter scale — typical
``h ∈ [0.05, 0.2]``.

``shrink_jitter`` is generic over the parameter pytree: any NamedTuple (or
pytree) whose leaves are ``(N,)``-shaped arrays is supported.  The leaves are
treated as the components of a ``d``-dimensional parameter vector in
declaration order — d is inferred from the number of leaves.  This is what
lets the four PF model variants (A, B, C, D, with d ∈ {3, 4}) share one
implementation.
"""

from __future__ import annotations

from typing import TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

# Liu-West cloud dim for Model A — retained for back-compat with external
# callers; the shrink-jitter kernel itself no longer needs it.
PARAM_DIM = 3


P = TypeVar("P")


def weighted_moments(P: Array, log_weights: Array) -> tuple[Array, Array]:
    """Return ``(mean, cov)`` of the weighted particle cloud.

    ``P`` has shape ``(N, d)``; ``log_weights`` shape ``(N,)``.
    """
    logZ = jax.scipy.special.logsumexp(log_weights)
    w = jnp.exp(log_weights - logZ)  # (N,)
    m = jnp.sum(w[:, None] * P, axis=0)  # (d,)
    diff = P - m
    cov = (w[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
    # Add a tiny ridge for stability when the cloud is degenerate.
    cov = cov + 1e-12 * jnp.eye(cov.shape[0])
    return m, cov


def shrink_jitter(
    params: P,
    log_weights: Array,
    h: float,
    key: Array,
) -> P:
    """Apply one Liu-West shrink-and-jitter step to ``params``.

    ``params`` is any pytree (typically a NamedTuple) whose leaves are
    ``(N,)``-shaped arrays.  The leaves are stacked into an ``(N, d)`` matrix
    in flatten order, the Liu-West update is applied, and the result is
    unflattened back into a pytree of the same structure.

    At ``h=0`` this is the identity map (no shrink, no jitter), which is useful
    as a degenerate case in tests.
    """
    leaves, treedef = jax.tree.flatten(params)
    Pmat = jnp.stack(leaves, axis=-1)  # (N, d)
    d = Pmat.shape[-1]
    m, V = weighted_moments(Pmat, log_weights)
    a = jnp.sqrt(1.0 - h * h)
    shrunk = a * Pmat + (1.0 - a) * m[None, :]
    Lc = jnp.linalg.cholesky(V)
    z = jr.normal(key, (Pmat.shape[0], d))
    eps = h * (z @ Lc.T)
    P_new = shrunk + eps
    new_leaves = [P_new[..., i] for i in range(d)]
    return jax.tree.unflatten(treedef, new_leaves)
