"""Hand cases for ``sc_empty_perturber_probe.py``: the algebra of an empty
SC perturber populated at first order (docs/hessian-theory.md, section 5.6).

An empty tuple's energy along ``s u + t v`` is ``F(s x_u + t x_v)`` with
``F(x) = -q_N(x)^2 / (q_N(x) Delta + q_H(x))``, ``q_N(x) = x.S.x`` and
``q_H(x) = x.K.x``. The cases build that from random ``S`` (positive
semi-definite), ``K`` and first-order vectors, and check what the probe and
the theory claim:

* the four-point mixed difference of ``F`` is independent of the step and
  equals the probe's ``polar``;
* with ``K = eta S`` on the plane ``F`` is a quadratic form: ``rho`` is zero,
  the bilinear sixth term ``-N^{uv} / (Delta + eta)`` equals ``polar``, it is
  K4 at the ray-limit multipliers, and the parallelogram law holds;
* otherwise ``rho`` and the parallelogram defect are not zero, ``rho`` does
  not change when either direction is rescaled, and a parallel pair has
  ``rho = 0`` whatever ``K`` is;
* :func:`dotted` is the derivative of :func:`rotate`;
* the transcription reproduces a committed golden's class energies, and every
  guarded tuple of N2 CAS(4,4) is empty by its integrals.

Synthetic inputs except the last case, which reads golden/n2_ccpvdz_cas44 (a
second of CPU). No card.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sc_empty_perturber_probe as probe
from scipy.linalg import expm

REPO = Path(__file__).resolve().parents[2]
DIM, NT = 6, 5  # perturber-space dimension, tuples
DELTA = np.array([1.3, 2.1, 0.9, 3.4, 1.7])


def _spd(rng, n, rank=None):
    a = rng.standard_normal((n, rank or n))
    return a @ a.T


def _tables(S, K, xu, xv):
    """The probe's cross tables for tuples whose first-order vectors are the
    rows of ``xu`` / ``xv``: ``n[u, v] = x_u.S.x_v`` and so on."""
    def f(M, a, b):
        return np.einsum("ti,ij,tj->t", a, M, b)
    return dict(nuu=f(S, xu, xu), nvv=f(S, xv, xv), nuv=f(S, xu, xv),
                huu=f(K, xu, xu), hvv=f(K, xv, xv), huv=f(K, xu, xv))


def _F(S, K, x, d):
    qn, qh = np.einsum("ti,ij,tj->t", x, S, x), np.einsum("ti,ij,tj->t", x, K, x)
    return -qn**2 / (qn * d + qh)


@pytest.fixture
def setup():
    rng = np.random.default_rng(190)
    S = _spd(rng, DIM, rank=4)
    K = 0.3 * _spd(rng, DIM)
    xu, xv = rng.standard_normal((NT, DIM)), rng.standard_normal((NT, DIM))
    return rng, S, K, xu, xv


@pytest.mark.parametrize("h", [1.0, 1e-2, 1e-4])
def test_stencil_is_the_polarisation_at_every_step(setup, h):
    _rng, S, K, xu, xv = setup

    def E(s, t):
        return _F(S, K, s * xu + t * xv, DELTA).sum()
    stencil = (E(h, h) - E(h, -h) - E(-h, h) + E(-h, -h)) / (4 * h * h)
    six = probe.sixth_term(_tables(S, K, xu, xv), DELTA)
    assert stencil == pytest.approx(six["polar"], rel=1e-9)


def test_proportional_is_bilinear(setup):
    _rng, S, _K, xu, xv = setup
    eta = 0.7
    T = _tables(S, eta * S, xu, xv)
    rho = probe.proportionality(T, DELTA)
    six = probe.sixth_term(T, DELTA)
    assert np.abs(rho).max() < 1e-14
    assert six["bilinear"] == pytest.approx(six["polar"], rel=1e-12)
    assert abs(six["parallelogram"]) < 1e-12 * abs(six["polar"])
    # K4 at the ray-limit multipliers: P = 2T + Delta T^2, Q = T^2, T = -1/(Delta + eta)
    t = -1.0 / (DELTA + eta)
    p, q = 2 * t + DELTA * t * t, t * t
    k4 = (p * 2 * T["nuv"] + q * 2 * T["huv"]).sum()
    assert k4 == pytest.approx(six["bilinear"], rel=1e-12)


def test_sijrs_limit_multiplier(setup):
    """H = 0: the sixth term is K4 at P = -1/Delta, the class's own multiplier."""
    _rng, S, _K, xu, xv = setup
    T = _tables(S, 0 * S, xu, xv)
    six = probe.sixth_term(T, DELTA)
    assert six["bilinear"] == pytest.approx((-2 * T["nuv"] / DELTA).sum(), rel=1e-12)
    assert six["polar"] == pytest.approx(six["bilinear"], rel=1e-12)


def test_not_proportional_is_not_a_quadratic_form(setup):
    _rng, S, K, xu, xv = setup
    T = _tables(S, K, xu, xv)
    rho = probe.proportionality(T, DELTA)
    six = probe.sixth_term(T, DELTA)
    assert rho.min() > 1e-3
    assert abs(six["parallelogram"]) > 1e-6 * abs(six["polar"])
    # rho is a property of the plane: rescaling either direction leaves it
    rho2 = probe.proportionality(_tables(S, K, 3 * xu, -0.5 * xv), DELTA)
    np.testing.assert_allclose(rho2, rho, rtol=1e-12)


def test_parallel_directions_are_proportional_whatever_K(setup):
    _rng, S, K, xu, _xv = setup
    T = _tables(S, K, xu, -2.5 * xu)
    assert probe.proportionality(T, DELTA).max() < 1e-13
    six = probe.sixth_term(T, DELTA)
    assert six["bilinear"] == pytest.approx(six["polar"], rel=1e-12)


def test_one_sided_population_has_no_mixed_term(setup):
    _rng, S, K, xu, _xv = setup
    T = _tables(S, K, xu, np.zeros_like(xu))
    assert probe.proportionality(T, DELTA).max() == 0.0
    six = probe.sixth_term(T, DELTA)
    assert six["bilinear"] == 0.0 and six["polar"] == 0.0


def test_dotted_is_the_derivative_of_rotate():
    rng = np.random.default_rng(7)
    nc, nv, na = 3, 4, 2
    shapes = {"h2e_v_Sr": (nv, na, na, na), "h1e_v_Sr": (nv, na), "h2e_v_Si": (na, na, nc, na),
              "h1e_v_Si": (na, nc), "cvcv": (nc, nv, nc, nv), "h2e_v_Sijr": (nv, na, nc, nc),
              "h2e_v_Srsi": (nv, nv, nc, na), "h2e_v_Srs": (nv, nv, na, na),
              "h2e_v_Sij": (na, na, nc, nc), "h2e_v1_Sir": (nv, na, nc, na),
              "h2e_v2_Sir": (nv, na, na, nc), "h1e_v_Sir": (nv, nc)}
    assert set(shapes) == set(probe.AXES)
    blocks = {k: rng.standard_normal(s) for k, s in shapes.items()}
    kc, kv = probe.antisym(rng, nc), probe.antisym(rng, nv)
    h = 1e-5
    plus = probe.rotate(blocks, expm(h * kc), expm(h * kv))
    minus = probe.rotate(blocks, expm(-h * kc), expm(-h * kv))
    dot = probe.dotted(blocks, kc, kv)
    for k in shapes:
        np.testing.assert_allclose((plus[k] - minus[k]) / (2 * h), dot[k], atol=1e-9)


def test_cas44_transcription_and_census():
    case = probe.Case(REPO / "golden" / "n2_ccpvdz_cas44.nevpt2gold")
    np.testing.assert_allclose(case.class_energies(), case.st.a["class_energies"], atol=1e-12)
    for cls in probe.GOLDEN_ORDER:
        assert (case.empty[cls] == case.guard[cls]).all(), cls
    assert int(case.guard["Sr"].sum()) == 11 and int(case.guard["Sij"].sum()) == 0
