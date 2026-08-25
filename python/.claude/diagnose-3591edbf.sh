#!/bin/bash
# Isolate R code from compiled library for the 3591edbf slowdown.
# Usage: diagnose.sh <lib_parent> <lib_child>
P=$1; C=$2
cat > /tmp/_t.R <<'EOF'
.libPaths(c(commandArgs(trailingOnly=TRUE)[1], .libPaths()))
suppressMessages(library(smooth)); y <- as.numeric(AirPassengers)
invisible(adam(y, model="MAM", lags=12))
cat(sprintf("%.4f\n", median(replicate(60, system.time(adam(y, model="MAM", lags=12))[["elapsed"]]))))
EOF
echo "R:        $(Rscript -e 'cat(R.version.string)')"
echo "BLAS:     $(Rscript -e 'cat(extSoftVersion()[["BLAS"]])')"
echo "CXXFLAGS: $(R CMD config CXXFLAGS)"
echo "Makevars: $(ls -l ~/.R/Makevars 2>/dev/null || echo none)"
for L in "$P" "$C"; do
  echo "--- $L"
  echo "  smooth.so: $(stat -c%s $L/smooth/libs/smooth.so) bytes"
  echo "  median:    $(OMP_NUM_THREADS=1 Rscript /tmp/_t.R $L 2>/dev/null | tail -1) s"
done
# THE decisive test: child's R code + parent's compiled library
cp -r "$C" "${C}_swap" && cp "$P/smooth/libs/smooth.so" "${C}_swap/smooth/libs/smooth.so"
echo "--- child R code + PARENT smooth.so"
echo "  median:    $(OMP_NUM_THREADS=1 Rscript /tmp/_t.R ${C}_swap 2>/dev/null | tail -1) s"
echo
echo "If the swap is fast  -> the compiled library differs; the R commit is not the cause."
echo "If the swap is slow  -> the R code is the cause."
