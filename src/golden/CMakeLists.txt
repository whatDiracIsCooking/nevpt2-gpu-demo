# nevpt2.golden -- the readers for generate_golden.py's two outputs: the
# .nevpt2gold golden (golden.cpp) and the .gradgold gradient sidecar beside it
# (gradgold.cpp, a separate file so the committed goldens' bytes never change;
# docs/gradient-theory.md, section 3 is the inventory it carries). Host only,
# no GPU. A missing or malformed file of either kind is a value-tier IO Error
# (nevpt2.error_handling), reported once by each demo's main().
# golden_reader.cppm is the internal partition nevpt2.golden:reader -- the
# bounded file cursor and array parser both loaders share, which no importer
# sees.
add_cxx_module_library(
  NAME nevpt2.golden
  PRIMARY_INTERFACE golden.cppm
  PARTITIONS golden_reader.cppm
  IMPLEMENTATION golden.cpp gradgold.cpp
  LINK_PUBLIC nevpt2::tensor nevpt2::error_handling
)
