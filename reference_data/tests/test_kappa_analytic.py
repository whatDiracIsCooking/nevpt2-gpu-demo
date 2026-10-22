"""Hand cases for ``nevpt2_grad.py``'s analytic ``kappa``-direction objects
(#250): ``E_CASCI``'s slope ``A_casci``, the constraint Jacobian ``M`` and
``Q = d(H c)/dkappa``, each linear in the MO Hamiltonian's rotation
derivative ``[I, kappa]`` (``kappa_analytic``), against the sweep's central
differences (``kappa_sweep``) that they replace.

On the committed N2/cc-pVDZ CAS(4,4) states, CASCI-on-RHF and CASSCF, with
``E_corr`` dropped (its half is the dump's ``Y``, held to the integral
adjoint in every run that reads one). Seconds of CPU, no card. The whole-run
A/B, the guess-to-guess spread it removes and the walls are
docs/testing.md, "The analytic kappa terms".

Tolerances are written out here, not read from the module.
"""

from __future__ import annotations

import nevpt2_grad as ng
import numpy as np
import pytest

#: The sweep at KAPPA_STEP against the analytic objects: its round-off over
#: 2 * 2^-17 on entries of up to ~20 Eh (measured 1.5e-8 on N2's M).
SWEEP_TOL = 1e-7
#: Two gradients, one per set of objects, on one state: the sweep's error
#: propagated through the solve.
GRAD_TOL = 1e-7


@pytest.fixture(scope="module", params=["casci", "casscf"])
def objects(request):
    spec = dict(ng.CASES["n2_ccpvdz_cas44"])
    if request.param == "casscf":
        st = ng.build_state("n2_ccpvdz_cas44", spec, reference="casscf",
                            mo_from=ng.REPO / "golden" / "n2_ccpvdz_cas44_casscf.mogold")
    else:
        st = ng.build_state("n2_ccpvdz_cas44", spec)
    fn = ng.Functional(st)
    cons = ng.constraints(st)
    an = ng.kappa_analytic(fn, st, cons, log=lambda *a: None)
    sweeps = {h: ng.kappa_sweep(fn, st, cons, with_corr=False, step=h, log=lambda *a: None)
              for h in (ng.KAPPA_STEP, 2.0 ** -10, 2.0 ** -12)}
    cols = [m for m, (kind, _p, _q) in enumerate(cons) if kind == "f" or not st.casscf]
    return st, fn, cons, an, sweeps, cols


def test_the_objects_are_the_sweeps(objects):
    st, _fn, _cons, an, sweeps, cols = objects
    sw = sweeps[ng.KAPPA_STEP]
    assert np.abs(an["A_casci"] - sw["A_casci"]).max() < SWEEP_TOL
    assert np.abs(an["M"][:, cols] - sw["M"][:, cols]).max() < SWEEP_TOL
    assert np.abs(an["Q"] - sw["Q"]).max() < SWEEP_TOL
    # nothing here is vacuous: the Jacobian and (CASCI) the source term carry weight
    assert np.abs(an["M"][:, cols]).max() > 1.0
    if not st.casscf:
        assert np.abs(an["A_casci"]).max() > 1e-3 and np.abs(an["Q"]).max() > 1e-2


def test_the_sweep_converges_onto_them_as_h_squared(objects):
    """Above round-off the sweep's error is its h^2 truncation: two halvings
    of the step take it down ~16-fold, so the analytic objects are what the
    quotient converges TO, not merely near it."""
    _st, _fn, _cons, an, sweeps, cols = objects
    coarse, fine = sweeps[2.0 ** -10], sweeps[2.0 ** -12]
    e_coarse = np.abs(an["M"][:, cols] - coarse["M"][:, cols]).max()
    e_fine = np.abs(an["M"][:, cols] - fine["M"][:, cols]).max()
    assert 10.0 < e_coarse / e_fine < 25.0, (e_coarse, e_fine)


def test_casscf_slope_is_gen_g_hops_gradient(objects):
    """On the CASSCF state ``A_casci`` along (S1)'s rotations is
    ``gen_g_hop``'s ``g_orb``, with the sign ``rhs_full`` reads it with
    (a Givens rotation by ``t`` along ``(p, q)`` is the step ``x_(q,p) =
    -t``); along (O2)'s it vanishes."""
    st, fn, cons, an, _sweeps, _cols = objects
    if not st.casscf:
        pytest.skip("the CASSCF branch's identity")
    g_all, _h, _d = fn.casscf_g_hop(st.mo_coeff, st.ci)
    var = fn.casscf_variables()
    pos = {(int(j), int(i)): n for n, (i, j) in enumerate(var)}
    a = an["A_casci"]
    s = [(p, q) for kind, p, q in cons if kind == "S"]
    gap = max(abs(a[p, q] + g_all[pos[(p, q)]]) for p, q in s)
    assert gap < 1e-12
    assert max(abs(a[p, q]) for kind, p, q in cons if kind == "f") < 1e-12


def test_the_density_response_is_needed(objects):
    """``f`` is not covariant across the core/active boundary: the Jacobian
    without the index-space density's response, ``[f, kappa]`` alone, misses
    the sweep by far more than its round-off. The comparison can fail."""
    st, _fn, cons, an, sweeps, _cols = objects
    h, g = ng.full_mo_hamiltonian(ng.Functional(st), st.mo_coeff)
    pairs = ng.kappa_pairs(st)
    idx = [m for m, (kind, _p, _q) in enumerate(cons) if kind == "f"]
    rows = np.array([cons[m][1] for m in idx])
    cols = np.array([cons[m][2] for m in idx])
    gamma = ng.Functional(st).reference_dm1(st)
    d_f = np.zeros((st.nmo, st.nmo))
    d_f[st.core, st.core] = 2.0 * np.eye(st.ncore)
    d_f[st.act, st.act] = gamma
    fock = (h + np.einsum("abrs,rs->ab", g, d_f) - 0.5 * np.einsum("arsb,rs->ab", g, d_f))
    # [f, kappa] alone: the response to the moving projector dropped
    comm_only = ng.fock_rotation_derivatives(fock, np.zeros_like(g), np.zeros_like(d_f),
                                             pairs, rows, cols)
    full = ng.fock_rotation_derivatives(h, g, d_f, pairs, rows, cols)
    sw = sweeps[ng.KAPPA_STEP]["M"][:, idx]
    assert np.abs(full - sw).max() < SWEEP_TOL
    assert np.abs(comm_only - sw).max() > 1e-3


def test_the_gradient_is_the_sweeps(objects):
    """The CASCI-only (or CASSCF-only) gradient solved on each set of objects
    over one state: they agree within the sweep's error."""
    st, _fn, _cons, _an, _sweeps, _cols = objects
    src = ng.SourceTerms(kappa=np.zeros((st.nmo, st.nmo)), ci=np.zeros(st.ci.shape),
                         origin="test")
    out = {}
    for kappa in ng.KAPPA_TERMS:
        g = ng.solve_gradient(st, src, pc=False, with_corr=False, log=lambda *a: None,
                              kappa=kappa)
        out[kappa] = g
    d = np.abs(out["analytic"].de - out["quotient"].de).max()
    assert d < GRAD_TOL, d
    assert np.abs(out["analytic"].de).max() > 1e-3
    # the A/B residuals the quotient run reports, beside it
    r = out["quotient"].residuals
    assert max(r["kappa_ab_A_casci"], r["kappa_ab_M"], r["kappa_ab_Q"]) < SWEEP_TOL
