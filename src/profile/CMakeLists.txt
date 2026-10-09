# nevpt2.profile -- the one --profile recorder: labelled event
# pairs on the caller's stream, grouped into sections (RDM build, DF
# integrals, energy) and reported per section. Replaced einsum's global
# registry and rdm_build's PhaseTimer.
add_cxx_module_library(
  NAME nevpt2.profile
  PRIMARY_INTERFACE profile.cppm
  PARTITIONS profile_spans.cppm
  IMPLEMENTATION profile.cpp
  # PUBLIC: time() takes a wwrStream_t and is a template its callers
  # instantiate.
  LINK_PUBLIC nevpt2::wwr nevpt2::common
)
