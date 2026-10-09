# nevpt2.golden -- the .nevpt2gold reader (generate_golden.py's output). Host
# only, no GPU. A missing or malformed file is a value-tier IO Error
# (nevpt2.error_handling), reported once by each demo's main().
# golden_reader.cppm is the internal partition nevpt2.golden:reader -- the
# bounded file cursor and array parser loadGolden uses, which no importer sees.
add_cxx_module_library(
  NAME nevpt2.golden
  PRIMARY_INTERFACE golden.cppm
  PARTITIONS golden_reader.cppm
  IMPLEMENTATION golden.cpp
  LINK_PUBLIC nevpt2::tensor nevpt2::error_handling
)
