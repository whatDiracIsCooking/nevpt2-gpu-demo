#!/usr/bin/env bash
# Build and install OpenMolcas AT AN EXPLICIT TAG AND COMMIT.
#
# Called by docker/Dockerfile.{cuda,hip,combined} as a leaf layer beside
# install-claude-code.sh and install-agy.sh, and kept out of Dockerfile.base
# for their reason: base is the parent of every GPU image, so a bump there
# would invalidate the CUDA toolkit and the ~19GB ROCm install below it.
#
# WHAT IT IS FOR. One offline script, reference_data/molcas_casscf_hessian.py
# (#223), which checks reference_data/casscf_nuclear_hessian.py's analytic
# CASSCF nuclear Hessian against OpenMolcas's own analytic one (the McKinley
# module, over MCLR's CP-CASSCF response). Nothing in the build, ctest or the
# golden tier calls it; docs/reference-data.md, "The CASSCF Hessian against
# OpenMolcas", has what it checked.
#
# WHY FROM SOURCE. OpenMolcas is not pip-installable (so it is not in uv.lock)
# and has no conda-forge or Ubuntu package. Upstream publishes tagged sources
# on GitLab; this clones the pinned tag and REFUSES unless the tag resolves to
# the pinned commit, so a moved tag cannot change what lands in the image.
# A git commit pins the bytes the way a sha512 pins a tarball, and unlike a
# GitLab archive tarball its hash does not depend on GitLab's archiver.
#
# WHAT IT BUILDS, and why each choice:
#   * LINALG=Internal -- OpenMolcas's bundled reference BLAS/LAPACK (the
#     External/lapack submodule, pinned by the superproject's gitlink).
#     Upstream's OpenBLAS route needs an ILP64 (INTERFACE64=1) OpenBLAS, which
#     Ubuntu's libopenblas-dev is not; and the molecules this checks are a few
#     dozen basis functions, so the slow BLAS costs seconds.
#   * gfortran + gcc + g++, named explicitly: the image's CC/CXX are clang, and
#     OpenMolcas's Fortran/C interop is tested upstream with the GNU trio.
#   * Libxc, which OpenMolcas always links: its CMake clones it during the
#     build (ExternalProject, ~1GB of history) at the commit hash
#     OpenMolcas's own CMakeLists.txt names, so the commit pin covers it too.
#   * MPI, OpenMP, HDF5 and the unit tests OFF -- none is needed here, and each
#     is one more thing that could differ between two builds.
#   * the pymolcas driver, which needs pyparsing at configure time and at run
#     time. pyparsing is installed from ONE pinned wheel, sha256-checked, into
#     a private directory of this prefix -- not into /opt/venv, whose contents
#     are uv.lock's alone, and not with pip, which the image's venv lacks.
#
# The installed tree is $OPENMOLCAS_PREFIX (default /opt/openmolcas), and the
# entry point is a wrapper at $OPENMOLCAS_BIN (default /usr/local/bin/pymolcas)
# that sets MOLCAS and the driver's PYTHONPATH. $OPENMOLCAS_PREFIX/VERSION holds
# "<tag> <commit>", which devtools/doctor.sh compares against the pin.
#
# Run by hand (no root, no apt) with a scratch prefix to check this script
# without an image build:
#   INSTALL_OPENMOLCAS=1 OPENMOLCAS_VERSION=v26.06 OPENMOLCAS_COMMIT=<sha> \
#   OPENMOLCAS_PREFIX=/scratch/om OPENMOLCAS_BIN=/scratch/om/pymolcas-wrapper \
#   OPENMOLCAS_FC=... OPENMOLCAS_JOBS=6 bash docker/install-openmolcas.sh
set -euo pipefail

# OPT-IN AT THIS LAYER, exactly as the two agents are: the ARG's default is 0,
# so a plain `docker build` of a GPU Dockerfile gets no OpenMolcas; THIS
# project's builds get it because devtools/config.sh defaults
# INSTALL_OPENMOLCAS to 1 and docker/build.sh forwards it.
if [ "${INSTALL_OPENMOLCAS:-0}" != 1 ]; then
  echo "=== OpenMolcas not installed (INSTALL_OPENMOLCAS=${INSTALL_OPENMOLCAS:-0}; set 1 to opt in) ==="
  if [ -f "${OPENMOLCAS_PREFIX:-/opt/openmolcas}/VERSION" ]; then
    echo "    (the parent image's copy is still at ${OPENMOLCAS_PREFIX:-/opt/openmolcas} -- inherited, not installed here)"
  fi
  exit 0
fi

: "${OPENMOLCAS_VERSION:?set by the ARG of the same name in the Dockerfile}"
: "${OPENMOLCAS_COMMIT:?set by the ARG of the same name in the Dockerfile}"

OPENMOLCAS_REPO=${OPENMOLCAS_REPO:-https://gitlab.com/Molcas/OpenMolcas.git}
OPENMOLCAS_PREFIX=${OPENMOLCAS_PREFIX:-/opt/openmolcas}
OPENMOLCAS_BIN=${OPENMOLCAS_BIN:-/usr/local/bin/pymolcas}
OPENMOLCAS_JOBS=${OPENMOLCAS_JOBS:-$(nproc)}
OPENMOLCAS_FC=${OPENMOLCAS_FC:-gfortran}
OPENMOLCAS_CC=${OPENMOLCAS_CC:-gcc}
OPENMOLCAS_CXX=${OPENMOLCAS_CXX:-g++}
OPENMOLCAS_PYTHON=${OPENMOLCAS_PYTHON:-/usr/bin/python3}

# The driver's one third-party dependency, pinned by file and hash.
PYPARSING_WHEEL_URL=${PYPARSING_WHEEL_URL:-https://files.pythonhosted.org/packages/05/e7/df2285f3d08fee213f2d041540fa4fc9ca6c2d44cf36d3a035bf2a8d2bcc/pyparsing-3.2.3-py3-none-any.whl}
PYPARSING_WHEEL_SHA256=${PYPARSING_WHEEL_SHA256:-a749938e02d6fd0b59b356ca504a24982314bb090c383e3cf201c95ef7e2bfcf}

pin="${OPENMOLCAS_VERSION} ${OPENMOLCAS_COMMIT}"

# Dockerfile.combined re-runs this on top of the `cuda` image, which already
# built it; a rebuild is many minutes of Fortran for the same bytes. Inherit
# when the installed tree IS the pin, rebuild when it is not.
if [ -f "$OPENMOLCAS_PREFIX/VERSION" ] && [ "$(cat "$OPENMOLCAS_PREFIX/VERSION")" = "$pin" ] \
   && [ -x "$OPENMOLCAS_BIN" ]; then
  echo "=== OpenMolcas ${pin} already at ${OPENMOLCAS_PREFIX} (inherited from the parent image) ==="
  exit 0
fi

# gfortran is the one compiler the images lack (Dockerfile.base installs
# build-essential, which is gcc and g++). Installed here, only when opted in,
# so an image without OpenMolcas carries no Fortran either.
if ! command -v "$OPENMOLCAS_FC" >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends gfortran
    rm -rf /var/lib/apt/lists/*
  else
    echo "install-openmolcas: no Fortran compiler '$OPENMOLCAS_FC' and cannot apt-get it" >&2
    echo "  set OPENMOLCAS_FC (and OPENMOLCAS_CC / OPENMOLCAS_CXX) to a GNU toolchain" >&2
    exit 1
  fi
fi
for tool in "$OPENMOLCAS_CC" "$OPENMOLCAS_CXX" "$OPENMOLCAS_PYTHON" git cmake curl; do
  command -v "$tool" >/dev/null 2>&1 || { echo "install-openmolcas: '$tool' not found" >&2; exit 1; }
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# --- the source, at the pinned commit ---------------------------------------
echo "install-openmolcas: cloning ${OPENMOLCAS_REPO} at ${OPENMOLCAS_VERSION}"
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$OPENMOLCAS_VERSION" \
  "$OPENMOLCAS_REPO" "$tmp/src"
got=$(git -C "$tmp/src" rev-parse HEAD)
if [ "$got" != "$OPENMOLCAS_COMMIT" ]; then
  echo "install-openmolcas: tag ${OPENMOLCAS_VERSION} is ${got}, the pin says ${OPENMOLCAS_COMMIT}" >&2
  echo "  a moved tag is not the source that was reviewed; bump both ARGs on purpose" >&2
  exit 1
fi
# Only the submodule LINALG=Internal needs; its commit is the superproject's
# gitlink, so the pin above covers it too.
git -C "$tmp/src" submodule update --quiet --init --depth 1 External/lapack

# --- the driver's dependency -------------------------------------------------
mkdir -p "$OPENMOLCAS_PREFIX/driver-lib"
curl -fsSL -o "$tmp/pyparsing.whl" "$PYPARSING_WHEEL_URL"
printf '%s  %s\n' "$PYPARSING_WHEEL_SHA256" "$tmp/pyparsing.whl" | sha256sum -c - >/dev/null || {
  echo "install-openmolcas: sha256 mismatch for $PYPARSING_WHEEL_URL" >&2
  echo "  got $(sha256sum "$tmp/pyparsing.whl" | cut -d' ' -f1)" >&2
  exit 1
}
# A wheel is a zip; this one is pure Python, so unpacking it is installing it.
"$OPENMOLCAS_PYTHON" -I -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
  "$tmp/pyparsing.whl" "$OPENMOLCAS_PREFIX/driver-lib"
export PYTHONPATH="$OPENMOLCAS_PREFIX/driver-lib"

# --- configure, build, install ----------------------------------------------
cmake -S "$tmp/src" -B "$tmp/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$OPENMOLCAS_PREFIX" \
  -DCMAKE_Fortran_COMPILER="$OPENMOLCAS_FC" \
  -DCMAKE_C_COMPILER="$OPENMOLCAS_CC" \
  -DCMAKE_CXX_COMPILER="$OPENMOLCAS_CXX" \
  -DPython_EXECUTABLE="$OPENMOLCAS_PYTHON" \
  -DLINALG=Internal -DMPI=OFF -DOPENMP=OFF -DHDF5=OFF -DBUILD_TESTING=OFF \
  >"$tmp/configure.log" 2>&1 || { tail -40 "$tmp/configure.log" >&2; exit 1; }
cmake --build "$tmp/build" -j "$OPENMOLCAS_JOBS" >"$tmp/build.log" 2>&1 \
  || { tail -60 "$tmp/build.log" >&2; exit 1; }
cmake --install "$tmp/build" >"$tmp/install.log" 2>&1 \
  || { tail -40 "$tmp/install.log" >&2; exit 1; }

for need in pymolcas bin/seward.exe bin/rasscf.exe bin/mclr.exe bin/mckinley.exe basis_library/CC-PVDZ basis_library/DEF2-SVP; do
  [ -e "$OPENMOLCAS_PREFIX/$need" ] || { echo "install-openmolcas: $need missing from $OPENMOLCAS_PREFIX" >&2; exit 1; }
done

# The wrapper: pymolcas finds its tree through MOLCAS and pyparsing through
# PYTHONPATH, and both are this install's, not the caller's.
mkdir -p "$(dirname "$OPENMOLCAS_BIN")"
cat >"$OPENMOLCAS_BIN" <<EOF
#!/bin/sh
# Written by docker/install-openmolcas.sh: OpenMolcas ${pin}.
MOLCAS=${OPENMOLCAS_PREFIX}
export MOLCAS
PYTHONPATH=${OPENMOLCAS_PREFIX}/driver-lib\${PYTHONPATH:+:\$PYTHONPATH}
export PYTHONPATH
exec ${OPENMOLCAS_PYTHON} ${OPENMOLCAS_PREFIX}/pymolcas "\$@"
EOF
chmod 0755 "$OPENMOLCAS_BIN"

# --- the guard that matters: it runs ------------------------------------------
# A build that links but cannot run a job would leave an image whose one
# OpenMolcas consumer fails at first use. An H2 SCF exercises the driver, the
# basis library and the integral and SCF modules; McKinley is the cross-check's
# own business (reference_data/molcas_casscf_hessian.py).
# (pymolcas makes $MOLCAS_WORKDIR/<project> but not $MOLCAS_WORKDIR itself.)
mkdir -p "$tmp/smoke/work"
cat >"$tmp/smoke/h2.input" <<'EOF'
&GATEWAY
  Coord
  2

  H 0.0 0.0 0.0
  H 0.0 0.0 0.74
  Basis = STO-3G
  Group = NoSym
&SEWARD
&SCF
EOF
( cd "$tmp/smoke" && MOLCAS_WORKDIR="$tmp/smoke/work" "$OPENMOLCAS_BIN" -np 1 h2.input >h2.log 2>&1 ) || true
if ! grep -q "Happy landing" "$tmp/smoke/h2.log"; then
  echo "install-openmolcas: the H2 smoke job did not finish" >&2
  tail -40 "$tmp/smoke/h2.log" >&2
  exit 1
fi

printf '%s\n' "$pin" >"$OPENMOLCAS_PREFIX/VERSION"
echo "=== OpenMolcas ${pin} installed at ${OPENMOLCAS_PREFIX} (${OPENMOLCAS_BIN}) ==="
