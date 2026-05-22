"""Test-wide configuration: enable float64 before any JAX use."""

import jax

jax.config.update("jax_enable_x64", True)
