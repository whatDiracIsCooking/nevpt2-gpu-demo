"""Hand cases for ``pc_orbital_hessian_probe.py``: the algebra of PC's orbital
Hessian blocks (docs/hessian-theory.md, sections 6.4-6.8).

The synthetic family is ``test_pc_null_mode_probe``'s -- a Gram metric
``S(c) = Phi(c)^T Phi(c)`` of constant rank whose null space turns with
``c``, a Dyall block annihilating it at the base point only, ``b = S x +
S_1 h1`` -- given an ORBITAL parameter ``s`` that, like a rotation of the
integrals at fixed CI, moves ``x``, ``h1`` and ``K`` but not ``S`` or
``S_1``. ``K``'s orbital tangent couples the dropped space from one side, as
a Dyall block at dotted integrals does. The cases check what the theory
claims:

* along the orbital parameter the first derivative is the two-term formula
  at ANY CI vector -- no projector term, the kept span being fixed;
* the mixed (orbital, CI) second derivative is section 6.1's terms PLUS one
  rotation term, ``-2 rho.S_u.P.A_w.x`` at the base point, and is wrong
  without it;
* an empty tuple (``b = 0``) along an intra-space pair is an exact quadratic
  form: its four-point mixed difference is step-independent and equals the
  bilinear closed form, with no sixth term and nothing to refuse;
* :func:`second_dotted` and :func:`eps_second` are the mixed derivatives of
  the rotations they claim to differentiate;
* on golden/n2_ccpvdz_cas44 (a second of CPU), the active-rotation gauge
  ``Eddot_kc(w, u) = Eddot_cc(w^ c, u) + Edot(w^ u)`` holds for Sr along a
  singlet ``u`` to round-off -- and fails without the CI side's rotation
  terms; and Sr's ``H_kk`` along two active directions, with no rotation
  term, matches a ladder of the rotated energy and the BCH gauge
  ``Eddot_kk(v, w) + Edot([v, w]) / 2 = Eddot_kc(v, w^ c)``.

No card.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pc_null_mode_probe as pnm
import pc_orbital_hessian_probe as probe
import pytest
import sc_empty_perturber_probe as sep
from scipy.linalg import expm
from test_pc_null_mode_probe import DELTA, NC, NCOL, NROW, Family

REPO = Path(__file__).resolve().parents[2]


class Orbital:
    """``Family`` with an orbital parameter: ``x + s x_w``, ``h1 + s h1_w`` and
    ``K + s K_w(c)``, ``K_w(c) = sym(Phi(c)^T G Xi)`` -- linear in ``c``,
    zero on S's dropped space from the bra side only."""

    def __init__(self, seed: int):
        self.fam = Family(seed)
        rng = np.random.default_rng(100 + seed)
        self.G = rng.standard_normal((NROW, NROW))
        self.Xi = rng.standard_normal((NROW, NCOL))
        self.xw = rng.standard_normal(NCOL)
        self.h1w = rng.standard_normal(1)

    def phi(self, c):
        return np.einsum("k,knd->nd", c, self.fam.O)

    def Kw(self, c):
        return pnm.sym(self.phi(c).T @ self.G @ self.Xi)

    def energy(self, c, s, nd):
        S, K, S1 = self.fam.mats(c)
        S, K = pnm.sym(S), pnm.sym(K) + s * self.Kw(c)
        _, _, Vk, _ = pnm.split(S, nd)
        b = S @ (self.fam.x + s * self.xw) + S1 @ (self.fam.h1 + s * self.h1w)
        return b @ pnm.restricted_inverse(K + DELTA * S, Vk) @ b

    def first(self, c, nd):
        S, K, S1 = self.fam.mats(c)
        S, K = pnm.sym(S), pnm.sym(K)
        _, _, Vk, _ = pnm.split(S, nd)
        b = S @ self.fam.x + S1 @ self.fam.h1
        bw = S @ self.xw + S1 @ self.h1w
        return probe.orbital_first(b, bw, self.Kw(c), pnm.restricted_inverse(K + DELTA * S, Vk))


def _fd(f, pows=(8, 9, 10)):
    fd = [(f(2.0**-p) - f(-(2.0**-p))) / (2 * 2.0**-p) for p in pows]
    r1 = [(4 * fd[i + 1] - fd[i]) / 3 for i in range(len(fd) - 1)]
    return (16 * r1[1] - r1[0]) / 15


ND = NCOL - NROW
U_DIR = np.array([0.0, 0.8, -0.6])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_orbital_first_derivative_needs_no_projector_term(seed):
    orb = Orbital(seed)
    for c in (orb.fam.c0, orb.fam.c0 + 0.2 * U_DIR):  # at the base point and off it
        ref = _fd(lambda h, c=c: orb.energy(c, h, ND))
        assert orb.first(c, ND) == pytest.approx(ref, rel=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mixed_block_needs_the_rotation_term(seed):
    orb = Orbital(seed)
    fam, u = orb.fam, U_DIR
    cf = fam.coeffs(u, np.zeros(NC))
    S, K = pnm.sym(cf["S"]), pnm.sym(cf["K"])
    s, Vn, Vk, _ = pnm.split(S, ND)
    A = K + DELTA * S
    M = pnm.restricted_inverse(A, Vk)
    x, h1 = fam.x, fam.h1
    Su, Ku = pnm.sym(cf["Su"]), pnm.sym(cf["Ku"])
    b = S @ x + cf["S1"] @ h1
    bw = S @ orb.xw + cf["S1"] @ orb.h1w
    bu = Su @ x + cf["S1u"] @ h1
    bwu = Su @ orb.xw + cf["S1u"] @ orb.h1w
    Kwu = pnm.sym(orb.phi(u).T @ orb.G @ orb.Xi)  # K_w is linear in c
    base, rot, rot_eig = probe.orbital_mixed(b, bw, bu, bwu, A, orb.Kw(fam.c0),
                                             Ku + DELTA * Su, Kwu, M, pnm.kept_pinv(s, Vk),
                                             Su, Vn)
    ref = _fd(lambda h: orb.first(fam.c0 + h * u, ND))
    assert base + rot == pytest.approx(ref, rel=1e-7)
    assert abs(base - ref) > 1e-3 * abs(ref)
    assert rot == pytest.approx(rot_eig, rel=1e-9)  # P A = 0 at the base point


@pytest.mark.parametrize("seed", [0, 1])
def test_empty_tuple_is_an_exact_quadratic_form(seed):
    rng = np.random.default_rng(seed)
    d = 5
    R = rng.standard_normal((d, d))
    S = R @ R.T
    K = pnm.sym(rng.standard_normal((d, d))) + 4 * np.eye(d)
    M = np.linalg.inv(K + DELTA * S)
    bu, bv = rng.standard_normal(d), rng.standard_normal(d)
    z = np.zeros(d)

    def energy(s, t):  # an empty tuple populated at first order, b = s b_u + t b_v
        b = s * bu + t * bv
        return -b @ M @ b

    want = -probe.pair_second(z, z, bu, bu, bv, bv, z, z, 0 * S, M)
    assert want == pytest.approx(-2 * bu @ M @ bv, rel=1e-12)
    for h in (0.5, 1e-2):  # step-independent AND equal to the bilinear form
        lad = (energy(h, h) - energy(h, -h) - energy(-h, h) + energy(-h, -h)) / (4 * h * h)
        assert lad == pytest.approx(want, rel=1e-10)
    # the parallelogram law of the directional second derivatives holds exactly
    d2 = {k: 2 * energy(*v) for k, v in {"p": (1, 1), "m": (1, -1), "u": (1, 0),
                                         "v": (0, 1)}.items()}
    assert d2["p"] + d2["m"] - 2 * d2["u"] - 2 * d2["v"] == pytest.approx(
        0.0, abs=1e-12 * abs(d2["p"]))


def test_second_dotted_and_eps_second_are_mixed_derivatives():
    rng = np.random.default_rng(7)
    nc, nv = 3, 4
    X = rng.standard_normal((nv, 2, nc, nv))
    axes = {"X": ((0, "v"), (2, "c"), (3, "v"))}
    ku = {"c": sep.antisym(rng, nc), "v": sep.antisym(rng, nv)}
    kv = {"c": sep.antisym(rng, nc), "v": sep.antisym(rng, nv)}

    def rot(s, t):
        U = {k: expm(s * ku[k] + t * kv[k]) for k in ("c", "v")}
        Y = X
        for a, sp in axes["X"]:
            Y = sep.contract(Y, a, U[sp])
        return Y

    h = 1e-4
    fd = (rot(h, h) - rot(h, -h) - rot(-h, h) + rot(-h, -h)) / (4 * h * h)
    got = probe.second_dotted({"X": X}, axes, ku, kv)["X"]
    assert np.abs(got - fd).max() < 1e-6 * np.abs(got).max()

    e = rng.standard_normal(nv)

    def er(s, t):
        return sep.rotate_eps(e, expm(s * ku["v"] + t * kv["v"]))

    fd = (er(h, h) - er(h, -h) - er(-h, h) + er(-h, -h)) / (4 * h * h)
    got = probe.eps_second(e, ku["v"], kv["v"])
    assert np.abs(got - fd).max() < 1e-6 * np.abs(got).max()


def test_gauge_identity_on_cas44_sr():
    st = pnm.State(REPO / "golden" / "n2_ccpvdz_cas44.nevpt2gold")
    rng = np.random.default_rng(199)
    w = sep.antisym(rng, st.n)
    stw = probe.act_dotted(st, w)
    Dc, Dcw = st.Q(st.c), stw.Q(st.c)
    m = probe.Mixed("Sr", st, stw, Dc, Dcw)
    u = pnm.direction(st, "singlet", rng, None)
    (base, rot, _), pkx = m.mixed(st.B(st.c, u), stw.B(st.c, u))
    assert pkx > 1e-2  # K_w couples Sr's dropped (rank-bound) space
    assert m.lift(st.B(st.c, u), st.Q(u)) < 1e-10  # and u does not lift it
    gc, gu = probe.ci_generator(st, w, st.c), probe.ci_generator(st, w, u)
    dv = pnm.Derivs("Sr", st, Dc, st.B(st.c, gc), st.B(st.c, u), st.B(gc, u), st.Q(gc))
    hb, hr, _, _ = dv.hessian("Sr", st)
    yu = pnm.edot_at("Sr", st, Dc, pnm.comb([(2, st.B(st.c, gu))]), m.nd, True)
    scale = abs(base + rot)
    assert abs(base + rot - (hb + hr + yu)) < 1e-10 * scale
    assert abs(rot) > 1e-3 * scale  # the orbital side's own rotation term
    assert abs(base + rot - (hb + yu)) > 1e-4 * scale  # the CI side's, along w^ c


def test_orbital_block_has_no_rotation_term_on_cas44():
    """``H_kk`` along two active directions, with no projector or rotation
    term, against a four-point ladder of the rotated energy, and the BCH gauge
    ``Eddot_kk(v, w) + Edot([v, w]) / 2 = Eddot_kc(v, w^ c)`` -- whose right
    side carries a rotation term the left does not."""
    st = pnm.State(REPO / "golden" / "n2_ccpvdz_cas44.nevpt2gold")
    rng = np.random.default_rng(5)
    v, w = sep.antisym(rng, st.n), sep.antisym(rng, st.n)
    stv, stw, stvw = probe.act_dotted(st, v), probe.act_dotted(st, w), probe.act_second(st, v, w)
    Dc = st.Q(st.c)
    legs = ((stv, stv.Q(st.c)), (stw, stw.Q(st.c)), (stvw, stvw.Q(st.c)))
    an = probe.orbital_block("Sr", st, Dc, legs)
    nd = pnm.split(pnm.sym(pnm.build("Sr", Dc, st)[0]))[3]

    def energy(s, t):
        return probe.rotated_energy(st, "Sr", expm(s * v + t * w), nd)

    lad = probe.richardson([(energy(h, h) - energy(h, -h) - energy(-h, h) + energy(-h, -h))
                            / (4 * h * h) for h in (2.0**-6, 2.0**-7, 2.0**-8)])
    assert lad == pytest.approx(an, rel=1e-7)
    gc = probe.ci_generator(st, w, st.c)
    base, rot, _ = probe.Mixed("Sr", st, stv, Dc, legs[0][1]).mixed(
        st.B(st.c, gc), stv.B(st.c, gc))[0]
    comm = probe.act_dotted(st, v @ w - w @ v)
    half = 0.5 * probe.Mixed("Sr", st, comm, Dc, comm.Q(st.c)).first()
    assert an + half == pytest.approx(base + rot, rel=1e-10)
    assert abs(rot) > 1e-3 * abs(base + rot)
