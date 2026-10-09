# nevpt2.cli -- the command-line front end both demos share: the flags every
# demo takes (--golden, --tiles, the digest family, --pc, --profile,
# --pool-threshold), their integer parsing, and the checks that span flags.
# Host only; every bad argument is a value-tier Error each demo's main()
# reports once. An app's own flags (density_fit's --batch, --check-blocks)
# stay in its main.cppm.
add_cxx_module_library(
  NAME nevpt2.cli
  PRIMARY_INTERFACE cli.cppm
  IMPLEMENTATION cli.cpp
  # rdm_build: RdmBuildOptions, the struct the digest flags fill.
  # device_resources: parsePoolThreshold and the pool default.
  # cublas_emul: requireCublasEmul (a refusing stub on HIP).
  LINK_PUBLIC nevpt2::rdm_build nevpt2::device_resources nevpt2::cublas_emul
)
