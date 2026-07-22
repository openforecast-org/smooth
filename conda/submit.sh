#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGING_DIR="${HOME}/conda-forge-staging"

usage() {
    echo "Usage: $0 <r-smooth|python|tag>"
    echo ""
    echo "  tag       Create v1.0.0 tag and compute SHA256 hash"
    echo "  python    Submit Python smooth recipe (can run anytime after tag)"
    echo "  r-smooth  Submit r-smooth recipe (run after r-greybox is merged)"
    exit 1
}

[ $# -lt 1 ] && usage

ensure_staging_repo() {
    if [ -d "${STAGING_DIR}" ]; then
        cd "${STAGING_DIR}"
        git checkout main
        git pull upstream main 2>/dev/null || git pull origin main
    else
        gh repo fork conda-forge/staged-recipes --clone -- "${STAGING_DIR}"
        cd "${STAGING_DIR}"
    fi
}

case "$1" in
    tag)
        echo "=== Creating v1.0.0 tag ==="
        cd "${REPO_ROOT}"
        git tag v1.0.0
        git push origin v1.0.0
        echo "Waiting 5s for GitHub to process the tag..."
        sleep 5

        echo "=== Computing SHA256 ==="
        HASH=$(curl -sL "https://github.com/openforecast-org/smooth/archive/refs/tags/v1.0.0.tar.gz" | sha256sum | awk '{print $1}')
        echo "SHA256: ${HASH}"

        echo "=== Updating meta.yaml ==="
        sed -i "s/REPLACE_WITH_ACTUAL_HASH/${HASH}/" "${SCRIPT_DIR}/smooth/meta.yaml"
        echo "Updated conda/smooth/meta.yaml with hash ${HASH}"
        ;;

    python)
        # Verify hash has been set
        if grep -q "REPLACE_WITH_ACTUAL_HASH" "${SCRIPT_DIR}/smooth/meta.yaml"; then
            echo "ERROR: Run '$0 tag' first to set the source hash."
            exit 1
        fi

        echo "=== Submitting Python smooth recipe ==="
        ensure_staging_repo

        git checkout -b smooth-python main 2>/dev/null || git checkout smooth-python
        cp -r "${SCRIPT_DIR}/smooth" recipes/smooth

        git add recipes/smooth
        git commit -m "Add smooth recipe (Python v1.0.0)"
        git push -u origin smooth-python

        gh pr create \
            --repo conda-forge/staged-recipes \
            --title "Add smooth (Python)" \
            --body "$(cat <<'EOF'
## Summary
- Python package `smooth` v1.0.0
- ADAM (Augmented Dynamic Adaptive Model) for time series forecasting
- State space models: ETS, ARIMA, and regression in a unified framework
- License: LGPL-2.1
- Source: GitHub release + carma submodule (v0.6.7)

## Build notes
- C++ extensions via pybind11 (conda-provided, FetchContent patched out)
- Requires BLAS/LAPACK and Armadillo
- Python >= 3.10

## Checklist
- [x] Dual source entries (main repo + carma submodule)
- [x] CMake FetchContent patched to use conda pybind11
- [x] Build scripts for Linux/macOS and Windows
- [x] Test: import + fit/predict smoke test
EOF
)"

        echo "=== Done! Python smooth PR created. ==="
        ;;

    r-smooth)
        echo "=== Submitting r-smooth recipe ==="
        echo "Make sure r-greybox has been merged on conda-forge first!"
        read -rp "Has r-greybox been merged? [y/N] " confirm
        [[ "${confirm}" =~ ^[Yy]$ ]] || { echo "Aborting."; exit 0; }

        ensure_staging_repo

        git checkout -b r-smooth main 2>/dev/null || git checkout r-smooth
        cp -r "${SCRIPT_DIR}/r-smooth" recipes/r-smooth

        git add recipes/r-smooth
        git commit -m "Add r-smooth recipe (v4.4.0 from CRAN)"
        git push -u origin r-smooth

        gh pr create \
            --repo conda-forge/staged-recipes \
            --title "Add r-smooth" \
            --body "$(cat <<'EOF'
## Summary
- CRAN package `smooth` v4.4.0
- Forecasting using Single Source of Error state space models
- License: LGPL-2.1
- Depends on `r-greybox` (now available on conda-forge)

## Checklist
- [x] Source from CRAN with archive fallback
- [x] All dependencies available on conda-forge (including r-greybox)
- [x] Build scripts for Linux/macOS and Windows
- [x] Test: `library('smooth')`
EOF
)"

        echo "=== Done! r-smooth PR created. ==="
        ;;

    *)
        usage
        ;;
esac
