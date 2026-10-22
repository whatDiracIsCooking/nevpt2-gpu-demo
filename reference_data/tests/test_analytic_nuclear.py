"""Hand cases for ``nevpt2_grad.py``'s analytic nuclear term (#247): every
piece of ``Phi`` but ``E_corr`` as a density pair contracted with the frozen
path's derivative integrals (``Gradients.nuclear_densities``,
``integral_perturbation.path_gradient``).

On the committed N2/cc-pVDZ CAS(4,4) states, CASCI-on-RHF and CASSCF, with
SEEDED multipliers -- so every constraint carries weight without a sweep or
a solve -- and ``E_corr`` dropped (its half is the GPU's integral adjoint,
#246, whose own check is the dump's): the analytic term against the
frozen-path quotient of the same ``Phi``, Richardson-extrapolated, and the
densities against ``Phi``'s linear part rebuilt at a displaced geometry.
Seconds of CPU, no card. The whole-run A/B on the committed cases, with
the GPU's adjoint, is docs/testing.md, "The analytic nuclear term".

Tolerances are written out here, not read from the module.
"""

from __future__ import annotations

import nevpt2_grad as ng
import numpy as np
import pytest

#: The quotient's own error at NUC_STEP with the Richardson pass, on N2.
QUOTIENT_TOL = 1e-7
#: The densities' rebuild of Phi's linear part: an exact identity.
IDENTITY_TOL = 1e-9


def _seeded(g: ng.Gradients, seed: int) -> ng.Gradients:
    """``g`` with seeded multipliers (``z`` projected against the state, as
    the solve leaves it) and what the quotient path reads from ``prepare``."""
    st = g.st
    rng = np.random.default_rng(seed)
    g.lam = 0.05 * rng.standard_normal(g.n_constraint)
    z = rng.standard_normal(st.ci.size) * g.allowed
    c = st.ci.ravel()
    z = 0.05 * (z - float(c @ z) * c) / np.linalg.norm(z)
    g.z = z.reshape(st.ci.shape)
    g._fn_cache = {}
    g._S0_half = ng.sym_power(g.fn.ovlp, 0.5)
    return g


@pytest.fixture(scope="module", params=["casci", "casscf"])
def grads(request):
    spec = dict(ng.CASES["n2_ccpvdz_cas44"])
    src = None
    if request.param == "casscf":
        st = ng.build_state("n2_ccpvdz_cas44", spec, reference="casscf",
                            mo_from=ng.REPO / "golden" / "n2_ccpvdz_cas44_casscf.mogold")
    else:
        st = ng.build_state("n2_ccpvdz_cas44", spec)
    src = ng.SourceTerms(kappa=np.zeros((st.nmo, st.nmo)), ci=np.zeros(st.ci.shape),
                         origin="test")
    cls = ng.CasscfGradients if st.casscf else ng.Gradients
    return _seeded(cls(st, src, with_corr=False), 247)


def test_analytic_term_is_the_frozen_path_derivative(grads):
    """Both halves -- ``E_CASCI``'s and the seeded constraints' -- against
    the Richardson quotient of the same scalar along the frozen path."""
    ana = grads.analytic_nuclear()
    for half, kw in (("ham", dict(ham_only=True, lag_only=False)),
                     ("lag", dict(ham_only=False, lag_only=True))):
        quo, _res, _p = grads._nuclear_difference(**kw)
        assert np.abs(ana[half] - quo).max() < QUOTIENT_TOL, (half, np.abs(ana[half] - quo).max())
        assert np.abs(ana[half]).max() > 1e-4, f"{half} carries no weight: the test is vacuous"
    # a rigid translation moves no overlap: exact on the analytic side
    total = ana["ham"] + ana["lag"]
    assert np.abs(total.sum(axis=0)).max() < 1e-11


def test_densities_rebuild_phi_at_a_displaced_geometry(grads):
    grads.de = grads.analytic_nuclear()["ham"] + grads.analytic_nuclear()["lag"]
    for chk in ng.check_nuclear_densities(grads):
        assert chk.ok, chk.summary
        assert chk.numbers["err"] <= IDENTITY_TOL
        assert abs(chk.numbers["ref"]) > 1e-6, "the piece is zero: the identity is vacuous"


def test_a_dropped_multiplier_term_fails_the_identity(grads):
    """The identity must be able to fail: the constraints' densities with the
    CI multiplier's share left out miss Phi by far more than round-off."""
    grads.de = grads.analytic_nuclear()["ham"] + grads.analytic_nuclear()["lag"]
    saved = grads._analytic
    try:
        z = grads.z
        grads._analytic = None
        grads.z = np.zeros_like(z)
        dens = grads.nuclear_densities()
        grads.z = z
        grads._analytic = dict(saved, densities=dens)
        lag = [c for c in ng.check_nuclear_densities(grads) if "constraints" in c.name]
        assert lag and not lag[0].ok
    finally:
        grads._analytic = saved
