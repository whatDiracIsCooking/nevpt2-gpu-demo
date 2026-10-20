# References

## CAS reference

- B. O. Roos, P. R. Taylor, P. E. M. Siegbahn, *A complete active space SCF
  method (CASSCF) using a density matrix formulated super-CI approach*,
  Chem. Phys. **48**, 157 (1980).
  [doi:10.1016/0301-0104(80)80045-0](https://doi.org/10.1016/0301-0104(80)80045-0)

## NEVPT2

- K. G. Dyall, *The choice of a zeroth-order Hamiltonian for second-order
  perturbation theory with a complete active space self-consistent-field
  reference function*, J. Chem. Phys. **102**, 4909 (1995).
  [doi:10.1063/1.469539](https://doi.org/10.1063/1.469539)
- C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu,
  *Introduction of n-electron valence states for multireference perturbation
  theory*, J. Chem. Phys. **114**, 10252 (2001).
  [doi:10.1063/1.1361246](https://doi.org/10.1063/1.1361246)
- C. Angeli, R. Cimiraglia, J.-P. Malrieu, *N-electron valence state
  perturbation theory: a fast implementation of the strongly contracted
  variant*, Chem. Phys. Lett. **350**, 297 (2001).
  [doi:10.1016/S0009-2614(01)01303-3](https://doi.org/10.1016/S0009-2614(01)01303-3)
- C. Angeli, R. Cimiraglia, J.-P. Malrieu, *n-electron valence state
  perturbation theory: A spinless formulation and an efficient implementation
  of the strongly contracted and of the partially contracted variants*,
  J. Chem. Phys. **117**, 9138 (2002).
  [doi:10.1063/1.1515317](https://doi.org/10.1063/1.1515317)

## Analytic gradients

- J. W. Park, *Analytical gradient theory for strongly contracted (SC-) and
  partially contracted (PC-) N-electron valence state perturbation theory
  (NEVPT2)*, J. Chem. Theory Comput. **15**, 5417 (2019).
  [doi:10.1021/acs.jctc.9b00762](https://doi.org/10.1021/acs.jctc.9b00762),
  [arXiv:1907.10180](https://arxiv.org/abs/1907.10180). The paper
  [`gradient-theory.md`](gradient-theory.md) derives from; its equation
  numbers are the ones that document cites.
  The equation numbers `src/gradient/` cites are this paper's too: the SC
  amplitude and multipliers are Eqs. 12 and 36-38, the pseudodensities
  Eqs. 41-47, and the consistency identity its ctest entries assert is Eq. 40.

- Y. Nishimoto, *Analytic first-order derivatives of partially contracted
  n-electron valence state second-order perturbation theory (PC-NEVPT2)*,
  J. Chem. Phys. **151**, 114103 (2019).
  [doi:10.1063/1.5115819](https://doi.org/10.1063/1.5115819). An independent
  PC-NEVPT2 first-derivative implementation, contemporaneous with Park's. Cited
  by [`hessian-theory.md`](hessian-theory.md), §8 for its *second* derivatives:
  they are seminumerical — a stencil over the analytic gradient — which is the
  state of the art for this class of method and the reason that document's
  level 3 has no analytic oracle to be checked against. Also cited for its
  threshold on linear dependence ("accuracy of the gradient is not
  affected"; continuity "has not been investigated carefully"), by
  [`gradient-theory.md`](gradient-theory.md), §4.5 and
  [`hessian-theory.md`](hessian-theory.md), §6.2.
- Y. Nishimoto, *Analytic first-order derivatives of CASPT2 with IPEA
  shift*, J. Chem. Phys. **158**, 174112 (2023).
  [doi:10.1063/5.0147611](https://doi.org/10.1063/5.0147611). Its §II B 3
  and Eqs. 55–63 couple the linearly dependent and independent spaces. This
  is the analogue of [`hessian-theory.md`](hessian-theory.md), §6.2's
  off-eigenstate third term ([`gradient-theory.md`](gradient-theory.md),
  §4.5).
- J. W. Park, *Analytical gradient theory for resolvent-fitted second-order
  extended multiconfiguration perturbation theory (XMCQDPT2)*, J. Chem.
  Theory Comput. **17**, 6122 (2021).
  [doi:10.1021/acs.jctc.1c00613](https://doi.org/10.1021/acs.jctc.1c00613),
  [arXiv:2106.10817](https://arxiv.org/abs/2106.10817). Traces a spurious
  acrolein frequency to the overlap threshold.

## Derivative theory: Lagrangians, and what order of response a derivative needs

The framework [`hessian-theory.md`](hessian-theory.md), §7–§8 rests on. Park's
Eq. 31 Lagrangian is an instance of it; what these add is the **rule for which
responses a given derivative order needs**, which is what separates a gradient
(one Z-vector solve) from a Hessian (`3N` response solves).

- T. Helgaker, P. Jørgensen, *Analytical calculation of geometrical
  derivatives in molecular electronic structure theory*, Adv. Quantum Chem.
  **19**, 183 (1988).
  [doi:10.1016/S0065-3276(08)60616-4](https://doi.org/10.1016/S0065-3276(08)60616-4)
- T. Helgaker, P. Jørgensen, *Configuration-interaction energy derivatives in
  a fully variational formulation*, Theor. Chim. Acta **75**, 111 (1989).
  [doi:10.1007/BF00527713](https://doi.org/10.1007/BF00527713). The
  fully-variational Lagrangian in which the state and orbital response
  parameters obey the **2n+1** rule and the Lagrange multipliers the stronger
  **2n+2** rule. That pair is what fixes
  [`hessian-theory.md`](hessian-theory.md), §8's cost: a second derivative
  needs the wavefunction parameters only to **first** order and the multipliers
  only to **zeroth** — the gradient's own `Z`, undifferentiated.
- P. Jørgensen, T. Helgaker, *A numerically stable procedure for calculating
  Møller–Plesset energy derivatives, derived using the theory of Lagrangians*,
  Theor. Chim. Acta (1988).
  [doi:10.1007/BF00532006](https://doi.org/10.1007/BF00532006)

- R. Shepard, H. Lischka, *Analytic energy second derivatives for general
  MCSCF wave functions*, J. Chem. Phys. **80**, 2660 (1984).
  [doi:10.1063/1.447061](https://doi.org/10.1063/1.447061). The general MCSCF
  second-derivative formulation, including the redundant variables that the
  active–active gauge ([`gradient-theory.md`](gradient-theory.md), §1.3) puts
  into the coupled-perturbed equations.
- T. J. Dudley, R. M. Olson, M. W. Schmidt, M. S. Gordon, *Parallel coupled
  perturbed CASSCF equations and analytic CASSCF second derivatives*,
  J. Comput. Chem. **27**, 352 (2006).
  [doi:10.1002/jcc.20350](https://doi.org/10.1002/jcc.20350). An implemented
  CP-CASSCF / analytic CASSCF second derivative. Cited by
  [`hessian-theory.md`](hessian-theory.md), §1.1 and §8 as the evidence that
  the reference-side half of a nuclear NEVPT2 Hessian — the `3N` response
  solves — is established machinery rather than an open problem.

## Numerical linear algebra

- G. H. Golub, V. Pereyra, *The differentiation of pseudo-inverses and
  nonlinear least squares problems whose variables separate*, SIAM J. Numer.
  Anal. **10**, 413 (1973).
  [doi:10.1137/0710036](https://doi.org/10.1137/0710036). The derivative of a
  Moore–Penrose pseudo-inverse at locally constant rank, which
  [`hessian-theory.md`](hessian-theory.md), §6 differentiates a second time.
  Note the two range-complement terms carry `A^+ A^+`, **not** `A^+`.
- G. W. Stewart, *On the continuity of the generalized inverse*, SIAM J.
  Appl. Math. **17**, 33 (1969).
  [doi:10.1137/0117004](https://doi.org/10.1137/0117004). The pseudo-inverse
  is continuous only at constant rank. Cited by
  [`hessian-theory.md`](hessian-theory.md), §6.2.
- T. Kato, *Perturbation Theory for Linear Operators*, 2nd ed. (Springer,
  1995). Rellich–Kato analytic perturbation theory: how a zero eigenvalue
  of an analytic family moves, and the second-order degenerate (Schur
  complement) correction [`hessian-theory.md`](hessian-theory.md), §6.2's
  `L_uv` is an instance of.

## Related work: the two "no Hessian here" findings

The literature [`hessian-theory.md`](hessian-theory.md), §5.6 and §6.2
place their findings against, and [`gradient-theory.md`](gradient-theory.md),
§4.5 at first order. These come from a search on 2026-10-20 that was not
exhaustive. The papers it did not read are listed in those sections. Park
2019, Nishimoto 2019 and 2023, Park 2021 (under "Analytic gradients"),
Golub–Pereyra, Stewart and Kato are cited there too.

- C. Angeli, M. Pastore, R. Cimiraglia, *New perspectives in multireference
  perturbation theory: the n-electron valence state approach*, Theor. Chem.
  Acc. **117**, 743 (2007).
  [doi:10.1007/s00214-006-0207-0](https://doi.org/10.1007/s00214-006-0207-0).
  SC-NEVPT2's non-invariance to inactive and virtual rotations. Not read in
  full.
- S. Li, Z. Zhao, F. A. Evangelista, *Equation-of-motion internally
  contracted multireference unitary coupled-cluster theory*, J. Chem. Phys.
  **162**, 154112 (2025).
  [doi:10.1063/5.0261000](https://doi.org/10.1063/5.0261000),
  [arXiv:2501.17421](https://arxiv.org/abs/2501.17421). Potential energy
  curves that are discontinuous where the number of linearly independent
  operators changes.
- L. Joubert-Doriol, *A variational approach for linearly dependent moving
  bases in quantum dynamics: application to Gaussian functions*, J. Chem.
  Theory Comput. **18**, 5799 (2022).
  [doi:10.1021/acs.jctc.2c00461](https://doi.org/10.1021/acs.jctc.2c00461),
  [arXiv:2205.02358](https://arxiv.org/abs/2205.02358). The same
  discontinuity in a moving basis of changing dimension, outside electronic
  structure.
- A. A. Safari, R. J. Anderson, G. Li Manni, *Toward a stochastic complete
  active space second-order perturbation theory*, J. Phys. Chem. A **128**,
  281 (2024).
  [doi:10.1021/acs.jpca.3c05109](https://doi.org/10.1021/acs.jpca.3c05109).
  RDM noise propagating non-linearly through the inversion of the CASPT2
  metric.
- S. Battaglia, L. Fransén, I. Fdez. Galván, R. Lindh, *Regularized CASPT2:
  an intruder-state-free approach*, J. Chem. Theory Comput. **18**, 4814
  (2022). [doi:10.1021/acs.jctc.2c00368](https://doi.org/10.1021/acs.jctc.2c00368).
  Background on CASPT2's denominators.
- OpenMolcas CASPT2 manual, keyword `THRESHOLDS`:
  <https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html>.
  Separate thresholds for zero-norm components and for near linear
  dependencies.
- F. A. Evangelista, J. Gauss, *An orbital-invariant internally contracted
  multireference coupled cluster approach*, J. Chem. Phys. **134**, 114102
  (2011). [doi:10.1063/1.3559149](https://doi.org/10.1063/1.3559149).
  Listed because it was **not** read. It is on §6.2's list of papers to
  check.

## Related work

- A. Y. Sokolov, G. K.-L. Chan, *A time-dependent formulation of
  multi-reference perturbation theory*, J. Chem. Phys. **144**, 064102 (2016).
  [doi:10.1063/1.4941606](https://doi.org/10.1063/1.4941606)
- S. Guo, M. A. Watson, W. Hu, Q. Sun, G. K.-L. Chan, *N-Electron valence
  state perturbation theory based on a density matrix renormalization group
  reference function, with applications to the chromium dimer and a trimer
  model of poly(p-phenylenevinylene)*, J. Chem. Theory Comput. **12**, 1583
  (2016). [doi:10.1021/acs.jctc.5b01225](https://doi.org/10.1021/acs.jctc.5b01225)
- Y. Guo, K. Sivalingam, F. Neese, *Approximations of density matrices in
  N-electron valence state second-order perturbation theory (NEVPT2).
  I. Revisiting the NEVPT2 construction*, J. Chem. Phys. **154**, 214111
  (2021). [doi:10.1063/5.0051211](https://doi.org/10.1063/5.0051211)
- Y. Guo, K. Sivalingam, C. Kollmar, F. Neese, *Approximations of density
  matrices in N-electron valence state second-order perturbation theory
  (NEVPT2). II. The full rank NEVPT2 (FR-NEVPT2) formulation*, J. Chem. Phys.
  **154**, 214113 (2021).
  [doi:10.1063/5.0051218](https://doi.org/10.1063/5.0051218)
- C. Kollmar, K. Sivalingam, Y. Guo, F. Neese, *An efficient implementation
  of the NEVPT2 and CASPT2 methods avoiding higher-order density matrices*,
  J. Chem. Phys. **155**, 234104 (2021).
  [doi:10.1063/5.0072129](https://doi.org/10.1063/5.0072129)

## Configuration interaction

- P. J. Knowles, N. C. Handy, *A new determinant-based full configuration
  interaction method*, Chem. Phys. Lett. **111**, 315 (1984).
  [doi:10.1016/0009-2614(84)85513-X](https://doi.org/10.1016/0009-2614(84)85513-X)

## Basis sets

- T. H. Dunning Jr., *Gaussian basis sets for use in correlated molecular
  calculations. I. The atoms boron through neon and hydrogen*, J. Chem. Phys.
  **90**, 1007 (1989). [doi:10.1063/1.456153](https://doi.org/10.1063/1.456153)
- F. Weigend, *A fully direct RI-HF algorithm: Implementation, optimised
  auxiliary basis sets, demonstration of accuracy and efficiency*,
  Phys. Chem. Chem. Phys. **4**, 4285 (2002).
  [doi:10.1039/b204199p](https://doi.org/10.1039/b204199p)

## Floating-point emulation

- K. Ozaki, T. Ogita, S. Oishi, S. M. Rump, *Error-free transformations of
  matrix multiplication by using fast routines of matrix multiplication and
  its applications*, Numer. Algorithms **59**, 95 (2012).
  [doi:10.1007/s11075-011-9478-1](https://doi.org/10.1007/s11075-011-9478-1)
- Y. Uchino, K. Ozaki, T. Imamura, *Performance enhancement of the Ozaki
  Scheme on integer matrix multiplication unit*, Int. J. High Perform. Comput.
  Appl. **39**, 462 (2025).
  [doi:10.1177/10943420241313064](https://doi.org/10.1177/10943420241313064)
- Y. Uchino, K. Ozaki, T. Imamura, *High-Performance and Power-Efficient
  Emulation of Matrix Multiplication using INT8 Matrix Engines*, Proc. SC '25
  Workshops, 1824 (2025).
  [doi:10.1145/3731599.3767539](https://doi.org/10.1145/3731599.3767539)

## Reference software

- Q. Sun *et al.*, *PySCF: the Python-based simulations of chemistry
  framework*, WIREs Comput. Mol. Sci. **8**, e1340 (2018).
  [doi:10.1002/wcms.1340](https://doi.org/10.1002/wcms.1340)
- Q. Sun *et al.*, *Recent developments in the PySCF program package*,
  J. Chem. Phys. **153**, 024109 (2020).
  [doi:10.1063/5.0006074](https://doi.org/10.1063/5.0006074)
- H. Zhai *et al.*, *Block2: A comprehensive open source framework to develop
  and apply state-of-the-art DMRG algorithms in electronic structure and
  beyond*, J. Chem. Phys. **159**, 234801 (2023).
  [doi:10.1063/5.0180424](https://doi.org/10.1063/5.0180424)

## Software

| software | version pinned by |
|---|---|
| [PySCF](https://pyscf.org) | `uv.lock` |
| [block2](https://github.com/block-hczhai/block2-preview) | `uv.lock` |
| [NumPy](https://numpy.org) | `uv.lock` |
| [WarpWraps](https://github.com/whatDiracIsCooking/WarpWraps) | the `deps/WarpWraps` submodule commit |
| CUDA toolkit, cuBLAS | `CUDA_VERSION` in `docker/Dockerfile.cuda` |
| ROCm, hipBLAS | `ROCM_VERSION` in `docker/Dockerfile.hip` and `docker/Dockerfile.combined` |
| clang, libc++ | `LLVM_VERSION` in `docker/Dockerfile.base` |
| CMake | `CMAKE_VERSION` in `docker/Dockerfile.base`; `cmake/nevpt2_toolchain.cmake` pins the `import std` gate UUID per release |
