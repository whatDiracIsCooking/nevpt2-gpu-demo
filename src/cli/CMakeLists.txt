# nevpt2.cli -- the command line of all four golden-reading binaries: the
# common flags (--golden, --tiles, the digest family, --pc, --profile,
# --pool-threshold), each binary's own (--sc-tangent, --batch, --kappa-full,
# --fd-steps, ...), the binary/flag matrix that refuses a flag a binary does
# not implement, their integer parsing, and the checks that span flags. Host
# only; every bad argument is a value-tier Error each main() reports once.
add_cxx_module_library(
  NAME nevpt2.cli
  PRIMARY_INTERFACE cli.cppm
  # :binary_flags -- the binary/flag matrix and each binary's own flags.
  PARTITIONS cli_binary_flags.cppm
  IMPLEMENTATION cli.cpp
  # rdm_build: RdmBuildOptions, the struct the digest flags fill.
  # device_resources: parsePoolThreshold and the pool default.
  # cublas_emul: requireCublasEmul (a refusing stub on HIP).
  LINK_PUBLIC nevpt2::rdm_build nevpt2::device_resources nevpt2::cublas_emul
)
