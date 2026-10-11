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
