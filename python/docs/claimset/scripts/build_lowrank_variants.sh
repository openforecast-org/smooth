#!/usr/bin/env bash
# Builds the two variants of the Python extension _adamCore that lowrank_timing.py times:
#   lowrank: the filter keeps the covariance of the states as its low-rank factor P = U U'
#   full:    the filter keeps the whole covariance
# In the package the choice is internal (src/headers/adamCore.h, in adamCore::filter():
# state.lowRank = 4 * gaps * (...) <= state.cells.n_elem;) and there is no switch. The script
# copies src/ and python/CMakeLists.txt of the repository into a directory OUTSIDE the repository,
# patches the copy of that line to read the macro SMOOTH_FORCE_LOWRANK, and builds each variant with
# the CMakeLists of the package (Release: -O3 -DNDEBUG -flto, as the package's own build), adding
# only two definitions to the target _adamCore:
#   SMOOTH_FORCE_LOWRANK=1 or 0               (the patched choice)
#   PYBIND11_INTERNALS_KIND="_smooth_<variant>" (separate pybind11 registries, so both variants and
#                                               the package's own _adamCore load in one process)
# The repository itself is not modified.
#
# Usage: bash build_lowrank_variants.sh [OUTPUT_DIR]
#   OUTPUT_DIR defaults to $LOWRANK_DIR, else to smooth-lowrank in the temporary directory. The variants land
#   in OUTPUT_DIR/variants/{lowrank,full}/_adamCore*.so, where lowrank_timing.py looks for them
#   (it reads the same LOWRANK_DIR).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DEFAULT_DIR="${TMPDIR:-/tmp}/smooth-lowrank"
OUT="${1:-${LOWRANK_DIR:-$DEFAULT_DIR}}"
PYTHON="$REPO/python/.venv/bin/python"
case "$OUT/" in "$REPO"/*) echo "OUTPUT_DIR must be outside the repository" >&2; exit 1;; esac

# The sources: a fresh copy, without the objects of the R build
rm -rf "$OUT/src" "$OUT/python"
mkdir -p "$OUT/src" "$OUT/python"
cp -r "$REPO/src/headers" "$REPO/src/python" "$REPO/src/libs" "$OUT/src/"
cp "$REPO/python/CMakeLists.txt" "$OUT/python/"

# The patch of the choice between the factor and the whole covariance
"$PYTHON" - "$OUT/src/headers/adamCore.h" <<'EOF'
import sys
path = sys.argv[1]
text = open(path).read()
old = (
    "        state.lowRank = 4 * gaps * (noiseMean.n_elem == 0 ? 1 : nComponents)"
    " <= state.cells.n_elem;\n"
)
new = (
    "#ifdef SMOOTH_FORCE_LOWRANK\n"
    "        state.lowRank = (SMOOTH_FORCE_LOWRANK != 0);\n"
    "        (void)gaps;\n"
    "#else\n" + old + "#endif\n"
)
if text.count(old) != 1:
    sys.exit("adamCore.h: the line of the choice was not found exactly once")
open(path, "w").write(text.replace(old, new))
EOF
cat >> "$OUT/python/CMakeLists.txt" <<'EOF'

# lowrank_timing.py: the forced choice of the filter and a separate pybind11 registry
target_compile_definitions(_adamCore PRIVATE SMOOTH_FORCE_LOWRANK=${SMOOTH_FORCE_LOWRANK}
                           PYBIND11_INTERNALS_KIND="_smooth_${SMOOTH_VARIANT}")
EOF

# pybind11 from the package's own build directory when it is there (no download), else fetched
EXTRA=()
if [ -d "$REPO/python/build/_deps/pybind11-src" ]; then
    EXTRA+=("-DFETCHCONTENT_SOURCE_DIR_PYBIND11=$REPO/python/build/_deps/pybind11-src")
fi
EXTRA+=("-DFETCHCONTENT_SOURCE_DIR_CARMAARMADILLO=$OUT/src/libs/carma/extern/armadillo-code")

for variant in lowrank full; do
    force=$([ "$variant" = lowrank ] && echo 1 || echo 0)
    build="$OUT/build_$variant"
    rm -rf "$build"
    cmake -S "$OUT/python" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE="$PYTHON" -DPYTHON_EXECUTABLE="$PYTHON" \
        -DSMOOTH_FORCE_LOWRANK="$force" -DSMOOTH_VARIANT="$variant" "${EXTRA[@]}" > "$build.log"
    cmake --build "$build" --target _adamCore -v >> "$build.log"
    mkdir -p "$OUT/variants/$variant"
    rm -f "$OUT/variants/$variant"/_adamCore*.so
    cp "$build"/_adamCore*.so "$OUT/variants/$variant/"
    grep -m1 -o -- "-O3 -DNDEBUG[^ ]*.*-flto -fno-fat-lto-objects" "$build.log" | head -1 \
        | sed "s/^/$variant flags: /"
done
git -C "$REPO" rev-parse HEAD > "$OUT/variants/commit.txt"
git -C "$REPO" status --porcelain -- src python/CMakeLists.txt | wc -l | sed 's/^/dirty files in src: /' \
    >> "$OUT/variants/commit.txt"
echo "built: $OUT/variants (commit $(head -1 "$OUT/variants/commit.txt"))"
